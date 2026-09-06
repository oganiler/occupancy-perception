"""Command-line entrypoints. `gt` generates shards; `infer` runs the pipeline."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from occupancy.config import load_config
from occupancy.factory import build_adapter, build_grid_spec, build_voxelizer
from occupancy.gt.backends.base import to_target
from occupancy.gt.shard import write_shard


def cmd_gt(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    spec = build_grid_spec(cfg)
    adapter = build_adapter(cfg)
    vox = build_voxelizer(cfg, spec)

    out_dir = Path(cfg.gt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.snapshot.yaml")

    tokens = adapter.sample_tokens()
    tokens = tokens[args.start:]
    if args.num_samples:
        tokens = tokens[: args.num_samples]

    print(f"grid {spec.dims} = {spec.num_voxels:,} voxels -> {out_dir}")

    total_bytes, t_start = 0, time.perf_counter()
    stats: list[tuple[float, float, float]] = []

    for i, token in enumerate(tokens, 1):
        sample = adapter.get(token)
        c = sample.cloud
        state, sem = vox.build(
            c.points_m, c.labels, c.sensor_origins_m,
            (~c.in_annotated_box) | c.from_reference,      # D-003
            extra_free_points_m=c.free_only_points_m,
            extra_free_origins_m=c.free_only_origins_m,
        )
        path = write_shard(out_dir, sample, to_target(state, sem), cfg.dataset.root)
        total_bytes += path.stat().st_size

        u, f, o = (float(np.mean(state == k)) for k in (0, 1, 2))
        stats.append((u, f, o))
        print(f"[{i}/{len(tokens)}] {sample.scene_name} {token[:8]} "
              f"kf={sample.num_keyframes} unobs={u:.0%} free={f:.0%} "
              f"occ={o:.1%} {path.stat().st_size/1024:.0f} KiB")

    dt = time.perf_counter() - t_start
    raw = spec.num_voxels * len(tokens)
    arr = np.array(stats)

    print(f"\n{len(tokens)} shards in {dt:.1f} s ({dt/len(tokens):.2f} s/sample)")
    print(f"{total_bytes/1e6:.1f} MB on disk, {raw/total_bytes:.0f}x compression")
    print("\nvoxel state distribution across samples (mean / min / max):")
    for name, col in zip(("unobserved", "free", "occupied"), arr.T):
        print(f"  {name:<11} {col.mean():5.1%}   {col.min():5.1%}   {col.max():5.1%}")
    return 0

def cmd_infer(args: argparse.Namespace) -> int:
    raise NotImplementedError("DL pipeline not yet implemented")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="occupancy")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gt", help="generate occupancy ground-truth shards")
    g.add_argument("--start", type=int, default=0,
                   help="index of the first sample to process")    
    g.add_argument("--num-samples", type=int, default=None)
    g.add_argument("overrides", nargs="*", help="e.g. gt.window_keyframes=8")
    g.set_defaults(func=cmd_gt)

    i = sub.add_parser("infer", help="forward pass and loss")
    i.add_argument("overrides", nargs="*")
    i.set_defaults(func=cmd_infer)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())