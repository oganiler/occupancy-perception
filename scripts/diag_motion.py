# scripts/diag_motion.py
"""Test whether GT coverage is driven by ego motion during the aggregation window."""
import numpy as np
from collections import defaultdict
from pathlib import Path

from occupancy.config import load_config
from occupancy.gt.shard import read_shard
from occupancy.labels import UNOBSERVED_LABEL

cfg = load_config()
W = cfg.gt.window_keyframes

rows = []
for p in Path(cfg.gt.output_dir).glob("*.npz"):
    s = read_shard(p)
    rows.append((s.meta["scene_name"], s.meta["timestamp_us"],
                 s.T_global_from_ego[:3, 3],
                 float((s.target == UNOBSERVED_LABEL).mean())))

by_scene = defaultdict(list)
for scene, ts, xyz, unobs in rows:
    by_scene[scene].append((ts, xyz, unobs))

disp, unobs_all = [], []
print(f"{'scene':<12}{'displacement':>14}{'unobs':>9}")
print("-" * 36)
for scene in sorted(by_scene):
    v = sorted(by_scene[scene])
    d_scene, u_scene = [], []
    for i in range(len(v)):
        lo, hi = max(0, i - W), min(len(v) - 1, i + W)
        d = float(np.linalg.norm(v[hi][1] - v[lo][1]))
        d_scene.append(d); u_scene.append(v[i][2])
        disp.append(d); unobs_all.append(v[i][2])
    print(f"{scene:<12}{np.mean(d_scene):>11.1f} m{np.mean(u_scene):>8.0%}")

r = np.corrcoef(disp, unobs_all)[0, 1]
print(f"\ncorrelation(window displacement, unobserved) = {r:+.3f}  n={len(disp)}")