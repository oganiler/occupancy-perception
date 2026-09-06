# scripts/check_gt.py
import time
import numpy as np

from occupancy.config import load_config
from occupancy.factory import build_adapter, build_grid_spec, build_voxelizer

import sys
cfg = load_config(overrides=sys.argv[1:])
spec = build_grid_spec(cfg)
adapter = build_adapter(cfg)
vox = build_voxelizer(cfg, spec)

print(f"grid {spec.dims} = {spec.num_voxels:,} voxels, "
      f"stride={cfg.gt.free_ray_stride}")

t0 = time.perf_counter()
s = adapter.get(adapter.sample_tokens()[200])
t_adapt = time.perf_counter() - t0

c = s.cloud
cast_free = (~c.in_annotated_box) | c.from_reference     # D-003

t0 = time.perf_counter()
state, sem = vox.build(c.points_m, c.labels, c.sensor_origins_m, cast_free)
t_vox = time.perf_counter() - t0

n = state.size
u, f, o = [(state == k).sum() for k in (0, 1, 2)]
print(f"\nadapter  {t_adapt:6.2f} s   ({len(c):,} points)")
print(f"voxelize {t_vox:6.2f} s   ({cast_free.sum():,} rays cast)")
print(f"\nunobserved {u/n:6.1%}   free {f/n:6.1%}   occupied {o/n:6.1%}")
print(f"semantics present: {sorted(np.unique(sem[state == 2]).tolist())}")

#occupied at 0.8% (5,120 voxels) from 157k points is ~30 points per voxel, which suggests a lot of points fall outside the grid. Expected — your adapter reported x reaching ±96 m — but confirm the fraction rather than assume it:
idx = np.floor(spec.to_voxel_coords(c.points_m.astype(np.float64))).astype(np.int64)
print(f"in grid: {spec.in_bounds(idx).mean():.1%}")