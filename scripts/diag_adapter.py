# scripts/diag_adapter.py
import numpy as np
from occupancy.data.nuscenes_adapter import NuScenesAdapter
from occupancy.labels import OCC3D_CLASS_NAMES

a = NuScenesAdapter("/data/nuscenes", "v1.0-mini", window_keyframes=4)
tok = a.sample_tokens()[200]
print("window size:", len(a._keyframe_window(tok)))

c = a.get(tok).cloud
p, lab = c.points_m, c.labels

# 1. Ground plane. driveable_surface should sit tightly around -1.8 m.
road = p[lab == 11, 2]
print(f"road z: {np.percentile(road,1):.2f} .. {np.percentile(road,99):.2f} "
      f"(median {np.median(road):.2f})")

# 2. Where do the extremes live?
for name, m in [("z < -5", p[:,2] < -5), ("z > 8", p[:,2] > 8)]:
    idx = np.where(m)[0]
    print(f"\n{name}: {len(idx):,} points")
    if len(idx):
        cls, cnt = np.unique(lab[idx], return_counts=True)
        for k, n in sorted(zip(cls, cnt), key=lambda t: -t[1])[:5]:
            print(f"   {OCC3D_CLASS_NAMES[k]:<20} {n:,}  "
                  f"dyn={c.in_annotated_box[idx][lab[idx]==k].mean():.0%}")

# 3. Does the correction cause it? Compare with it disabled.
b = NuScenesAdapter("/data/nuscenes", "v1.0-mini", window_keyframes=4,
                    separate_dynamic=False)
q = b.get(tok).cloud.points_m
print(f"\nwith    correction: z {p[:,2].min():.1f} .. {p[:,2].max():.1f}")
print(f"without correction: z {q[:,2].min():.1f} .. {q[:,2].max():.1f}")

cs = a.nusc.get('calibrated_sensor',
                a.nusc.get('sample_data',
                    a.nusc.get('sample', tok)['data']['LIDAR_TOP'])['calibrated_sensor_token'])
print(cs['translation'])     # expect z around 1.8

# The six cameras are not simultaneous with each other. 
# Each fires as the spinning LiDAR crosses its FOV, so CAM_FRONT_LEFT fires first and the order is constant across keyframes. 
# The spread is bounded by one LiDAR rotation — 50 ms at 20 Hz.
# Nobody models this — every nuScenes occupancy paper ignores it. But naming it as a known limitation, with a measured number rather than a hand-wave, is better than nothing. 
# Let's see what the numbers are for this scene.
s = a.nusc.get('sample', tok)
t0 = a.nusc.get('sample_data', s['data']['LIDAR_TOP'])['timestamp']
for name in ('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT'):
    dt = a.nusc.get('sample_data', s['data'][name])['timestamp'] - t0
    print(f"{name:<18} {dt/1000:+7.1f} ms")