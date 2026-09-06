"""Correlate ground-truth coverage with what the scene actually looks like.

Parses a `gt` run log, groups coverage statistics by scene, joins them with the
human-written nuScenes scene descriptions, and renders camera views alongside
the stored occupancy target for the best- and worst-covered scenes.

The BEV panels are read from the shards on disk, not regenerated, so this also
demonstrates that a shard is self-contained: images are located via the paths
it stores, with no devkit lookup.

Usage:
    python scripts/diag_scenes.py
    python scripts/diag_scenes.py --log outputs/gt_full.log --scenes 3
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from occupancy.config import load_config
from occupancy.factory import build_adapter
from occupancy.gt.shard import read_shard
from occupancy.labels import FREE_CLASS, UNOBSERVED_LABEL

LOG_LINE = re.compile(
    r"(?P<scene>scene-\d+)\s+(?P<token>\w+)\s+kf=(?P<kf>\d+)\s+"
    r"unobs=(?P<unobs>\d+)%\s+free=(?P<free>\d+)%\s+occ=(?P<occ>[\d.]+)%"
)

# Panels chosen to show the environment ahead and behind: coverage depends on
# how far rays travel before leaving the slab, which is a property of what
# surrounds the ego, not of what is in front of it alone.
CAMERAS = ("CAM_FRONT", "CAM_BACK")

STATE_CMAP = ListedColormap(["#202020", "#5ec8ff", "#ff8c00"])  # unobs/free/occ


def parse_log(path: Path) -> dict[str, list[dict]]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        m = LOG_LINE.search(line)
        if m:
            by_scene[m["scene"]].append(
                {
                    "kf": int(m["kf"]),
                    "unobs": int(m["unobs"]),
                    "free": int(m["free"]),
                    "occ": float(m["occ"]),
                }
            )
    return dict(by_scene)


def scene_sample_tokens(nusc, scene_name: str) -> list[str]:
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    tokens, token = [], scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def bev_state(target: np.ndarray) -> np.ndarray:
    """Collapse the stored target to a 2D state map: 0 unobs, 1 free, 2 occ."""
    occupied = target < FREE_CLASS
    free = target == FREE_CLASS
    bev = np.zeros(target.shape[:2], dtype=np.uint8)
    bev[free.any(axis=2)] = 1
    bev[occupied.any(axis=2)] = 2   # occupied wins, as in the grid itself
    return bev


def render_one(row: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for c, cam in enumerate(CAMERAS):
        if row["images"].get(cam) is not None:
            axes[c].imshow(row["images"][cam])
        axes[c].set_title(cam, fontsize=10)
        axes[c].axis("off")

    axes[2].imshow(bev_state(row["target"]).T, origin="lower",
                   cmap=STATE_CMAP, vmin=0, vmax=2)
    axes[2].plot(row["target"].shape[0] // 2, row["target"].shape[1] // 2,
                 "w+", ms=12, mew=2)
    axes[2].set_title("stored target (unobserved / free / occupied)", fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle(
        f"#{row['rank']:02d}  {row['scene']}   "
        f"unobserved {row['unobs']:.1f}%  occupied {row['occ']:.1f}%\n"
        f"{row['description']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="outputs/gt_full.log")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--scenes", type=int, default=2,
                   help="how many scenes from each extreme to render")
    p.add_argument("--out", default="outputs/scene_coverage.png")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(
            f"{log_path} not found. Generate it with:\n"
            f"  python -m occupancy.cli gt 2>&1 | tee {log_path}"
        )

    by_scene = parse_log(log_path)
    if not by_scene:
        raise SystemExit(f"no per-sample lines matched in {log_path}")

    cfg = load_config(args.config)
    adapter = build_adapter(cfg)
    nusc = adapter.nusc
    shard_dir = Path(cfg.gt.output_dir)

    descriptions = {s["name"]: s["description"] for s in nusc.scene}

    summary = sorted(
        (
            {
                "scene": name,
                "n": len(v),
                "unobs": st.mean(x["unobs"] for x in v),
                "unobs_min": min(x["unobs"] for x in v),
                "unobs_max": max(x["unobs"] for x in v),
                "occ": st.mean(x["occ"] for x in v),
                "description": descriptions.get(name, "(no description)"),
            }
            for name, v in by_scene.items()
        ),
        key=lambda d: d["unobs"],
    )

    for rank, s in enumerate(summary, 1):      # add this line after the sort
        s["rank"] = rank

    print(f"{'scene':<12}{'n':>4}{'unobs':>8}{'range':>12}{'occ':>7}  description")
    print("-" * 110)
    for s in summary:
        rng = f"{s['unobs_min']:.0f}-{s['unobs_max']:.0f}%"
        print(f"{s['scene']:<12}{s['n']:>4}{s['unobs']:>7.1f}%{rng:>12}"
              f"{s['occ']:>6.1f}%  {s['description'][:60]}")

    k = min(args.scenes, len(summary) // 2)
    chosen = summary[:k] + summary[-k:]

    print(f"\nrendering {len(chosen)} scenes "
          f"(best and worst {k} by coverage)")

    rows = []
    for s in chosen:
        tokens = scene_sample_tokens(nusc, s["scene"])
        token = tokens[len(tokens) // 2]           # mid-scene, window saturated
        shard_path = shard_dir / f"{token}.npz"
        if not shard_path.exists():
            print(f"  skip {s['scene']}: {shard_path.name} not found")
            continue

        shard = read_shard(shard_path)
        images = {}
        for cam in CAMERAS:
            if cam in shard.camera_names:
                rel = shard.image_paths[shard.camera_names.index(cam)]
                images[cam] = mpimg.imread(Path(cfg.dataset.root) / rel)
            else:
                images[cam] = None

        rows.append({**s, "target": shard.target, "images": images})
        print(f"  {s['scene']}  {token[:8]}  unobs {s['unobs']:.0f}%")

    out = Path(args.out)
    print()
    for row in rows:
        render_one(row, out.with_name(
            f"{out.stem}_{row['rank']:02d}_{row['scene']}{out.suffix}"
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())