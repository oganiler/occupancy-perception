# scripts/check_adapter.py
from occupancy.data.nuscenes_adapter import NuScenesAdapter

a = NuScenesAdapter("/data/nuscenes", "v1.0-mini", window_keyframes=4,
                    camera_names=("CAM_FRONT",))
s = a.get(a.sample_tokens()[200])          # mid-scene, avoids the boundary case
c = s.cloud
print(f"{s.scene_name}  points={len(c):,}  dynamic={c.in_annotated_box.sum():,}")
print(f"labels present: {sorted(set(c.labels.tolist()))}")
print(f"x range: {c.points_m[:,0].min():.1f} .. {c.points_m[:,0].max():.1f}")
print(f"z range: {c.points_m[:,2].min():.1f} .. {c.points_m[:,2].max():.1f}")