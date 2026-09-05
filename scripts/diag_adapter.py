"""Adapter diagnostics: ground plane, z-extremes, and dynamic-correction effect.

Run with optional overrides, e.g.
    python scripts/diag_adapter.py gt.window_keyframes=8
"""
import sys

import numpy as np

from occupancy.config import load_config
from occupancy.factory import build_adapter
from occupancy.labels import OCC3D_CLASS_NAMES

SAMPLE_INDEX = 200
DRIVEABLE_SURFACE = 11

cfg = load_config(overrides=sys.argv[1:])
adapter = build_adapter(cfg)

tok = adapter.sample_tokens()[SAMPLE_INDEX]
sample = adapter.get(tok)
c = sample.cloud
p, lab = c.points_m, c.labels

print(f"{sample.scene_name}  window={cfg.gt.window_keyframes}  points={len(c):,}")

# 1. Ground plane. In the EGO frame the road sits near z = 0: the LIDAR_TOP
#    extrinsic lifts the sensor ~1.84 m, and that offset is removed by the
#    lidar -> ego transform. A road near -1.8 would mean the extrinsic was
#    skipped.
road = p[lab == DRIVEABLE_SURFACE, 2]
print(f"\nroad z: {np.percentile(road, 1):.2f} .. {np.percentile(road, 99):.2f} "
      f"(median {np.median(road):+.2f})")

# 2. Where the z-extremes live.
for name, mask in [("z < -5", p[:, 2] < -5), ("z > 8", p[:, 2] > 8)]:
    idx = np.flatnonzero(mask)
    print(f"\n{name}: {len(idx):,} points")
    for k, n in sorted(zip(*np.unique(lab[idx], return_counts=True)),
                       key=lambda t: -t[1])[:5]:
        print(f"   {OCC3D_CLASS_NAMES[k]:<20} {n:,}")

# 3. Does dynamic correction change the extent? Reloads NuScenes, so this
#    costs a few seconds on mini.
q = build_adapter(load_config(overrides=["gt.separate_dynamic=false"])) \
        .get(tok).cloud.points_m
print(f"\nwith    correction: z {p[:, 2].min():6.1f} .. {p[:, 2].max():5.1f}")
print(f"without correction: z {q[:, 2].min():6.1f} .. {q[:, 2].max():5.1f}")