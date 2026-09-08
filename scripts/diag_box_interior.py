"""Are vehicle interiors marked FREE in the generated ground truth?

Pass 1 casts free rays from every keyframe in the aggregation window. A vehicle
occupying volume B at t_ref was elsewhere earlier in the window, so beams from
those keyframes passed through B and marked it FREE. Pass 2 then marks only the
near surface OCCUPIED, because that is where the returns landed. The result is a
vehicle shell containing free space.

Neither D-003 nor the pass-3 guard prevents this: the offending rays terminate
on static geometry so they carry no annotation flag, and pass 1 runs before
there is anything to guard against. D-011 corrects it after pass 2.

This reads the generated shards rather than re-deriving the grid, so it
measures the artifact that actually ships.

    python scripts/diag_box_interior.py
    python scripts/diag_box_interior.py --start 200 --num-samples 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from occupancy.config import load_config
from occupancy.factory import build_adapter, build_grid_spec
from occupancy.gt.backends.base import FREE, OCCUPIED, UNOBSERVED
from occupancy.gt.box_fill import VEHICLE_PREFIXES, _voxel_centers_ego
from occupancy.gt.shard import read_shard
from occupancy.labels import FREE_CLASS, UNOBSERVED_LABEL

STATE_NAMES = {UNOBSERVED: "unobs", FREE: "free", OCCUPIED: "occ"}

# Below this a box is too small at 0.4 m resolution to have a meaningful
# interior: a distant car may occupy two voxels, both of them surface.
MIN_INTERIOR_VOXELS = 4


def states_from_target(target: np.ndarray) -> np.ndarray:
    """Recover the three-state view from the stored single-array target.

    The shard folds state and semantics into one uint8 (D-005): 255 unobserved,
    17 free, 0..16 occupied with a class.
    """
    return np.where(
        target == UNOBSERVED_LABEL, UNOBSERVED,
        np.where(target == FREE_CLASS, FREE, OCCUPIED),
    ).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--start", type=int, default=200)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    spec = build_grid_spec(cfg)
    adapter = build_adapter(cfg)
    shard_dir = Path(cfg.gt.output_dir)

    print(f"box_fill = {cfg.gt.box_fill}   shards from {shard_dir}\n")

    # Voxel centres are fixed by the grid spec, so build them once.
    centers = _voxel_centers_ego(spec)                       # (3, V)
    centers_h = np.vstack([centers, np.ones((1, centers.shape[1]))])

    tokens = adapter.sample_tokens()[args.start:args.start + args.num_samples]
    grand = np.zeros(3, dtype=int)
    n_boxes = 0

    for token in tokens:
        path = shard_dir / f"{token}.npz"
        if not path.exists():
            print(f"skip {token[:8]}: no shard. Generate it with\n"
                  f"  python -m occupancy.cli gt --start {args.start} "
                  f"--num-samples {args.num_samples}")
            continue

        flat = states_from_target(read_shard(path).target).reshape(-1)
        sample = adapter.get(token)

        print(f"{sample.scene_name}  {token[:8]}")
        totals = np.zeros(3, dtype=int)

        for box in sample.boxes:
            if not box.name.startswith(VEHICLE_PREFIXES):
                continue

            local = (np.linalg.inv(box.T_ego_from_box) @ centers_h)[:3]
            inside = np.all(np.abs(local) <= box.half_extent_m[:, None], axis=0)
            n = int(inside.sum())
            if n < MIN_INTERIOR_VOXELS:
                continue

            counts = np.array(
                [int((flat[inside] == k).sum())
                 for k in (UNOBSERVED, FREE, OCCUPIED)]
            )
            totals += counts
            n_boxes += 1
            print(f"  {box.name.split('.')[-1]:<14} {n:5d} voxels   "
                  f"unobs {counts[0]:5d}   free {counts[1]:5d}   "
                  f"occ {counts[2]:5d}")

        if totals.sum():
            print(f"  {'sample':<14} {totals.sum():5d} voxels   "
                  f"free {totals[1] / totals.sum():5.1%}\n")
        grand += totals

    if not grand.sum():
        print("no vehicle boxes with enough interior voxels")
        return 1

    print(f"{n_boxes} vehicle boxes, {grand.sum():,} interior voxels")
    print(f"  unobserved {grand[0] / grand.sum():6.1%}")
    print(f"  free       {grand[1] / grand.sum():6.1%}"
          f"{'   <- the defect' if grand[1] else '   <- corrected'}")
    print(f"  occupied   {grand[2] / grand.sum():6.1%}")

    if cfg.gt.box_fill != "none" and grand[1]:
        print("\nbox_fill is enabled but free interior voxels remain. Either "
              "the shards predate the change, or the box transform is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())