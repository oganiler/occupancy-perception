# scripts/check_adapter.py
from occupancy.config import load_config
from occupancy.factory import build_adapter, build_grid_spec, build_voxelizer

cfg = load_config()
spec = build_grid_spec(cfg)
adapter = build_adapter(cfg)

print(f"grid {spec.dims}  = {spec.num_voxels:,} voxels")

s = adapter.get(adapter.sample_tokens()[200])
c = s.cloud

print(f"{s.scene_name}  points={len(c):,}  dynamic={c.in_annotated_box.sum():,}")
print(f"labels present: {sorted(set(c.labels.tolist()))}")
print(f"x range: {c.points_m[:,0].min():.1f} .. {c.points_m[:,0].max():.1f}")
print(f"z range: {c.points_m[:,2].min():.1f} .. {c.points_m[:,2].max():.1f}")