# scripts/profile_forward.py
"""Where does the forward pass spend its time?

Answers whether the cost is the backbone, grid_sample, or the elementwise
accumulate in the view transform. The last would indicate a memory-bandwidth
bound rather than a compute bound.

    python scripts/profile_forward.py
    python scripts/profile_forward.py --batch-size 2 data.image_size=[144,256]
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.profiler as prof
from torch.utils.data import DataLoader

from occupancy.config import load_config
from occupancy.data.dataset import collate
from occupancy.factory import build_dataset, build_grid_spec, build_model

p = argparse.ArgumentParser()
p.add_argument("--batch-size", type=int, default=1)
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("overrides", nargs="*")
args = p.parse_args()

cfg = load_config(args.config, args.overrides)
spec = build_grid_spec(cfg)
ds = build_dataset(cfg, split="val")
dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
model = build_model(cfg, spec)
model.eval()

batch = next(iter(dl))

# Peak memory: the view transform materialises (B, N, C, V) floats, which at
# C=256 and V=640k is 655 MB per camera.
B, T, N = batch["imgs"].shape[:3]
V = spec.num_voxels
C_backbone = model.backbone.out_channels
C_lift = model.neck.out_channels          # what the view transform carries
print(f"batch {B}  cameras {N}  voxels {V:,}")
print(f"channels: backbone {C_backbone} -> neck {C_lift}")
print(f"one (C, V) tensor: {C_lift * V * 4 / 1e9:.2f} GB")
print(f"all cameras at once: {B * N * C_lift * V * 4 / 1e9:.2f} GB\n")

with torch.no_grad():
    model(batch)                                  # warm up, exclude one-off costs

t0 = time.perf_counter()
with torch.no_grad():
    model(batch)
print(f"forward {time.perf_counter() - t0:.2f} s\n")

with prof.profile(
    activities=[prof.ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prf:
    with torch.no_grad():
        model(batch)

print(prf.key_averages().table(
    sort_by="self_cpu_time_total", row_limit=15))