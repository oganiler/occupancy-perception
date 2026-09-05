"""Per-sample ground-truth shards.

A shard is self-contained: it holds the target grid, the camera calibration,
and the ego pose. The DL pipeline reads shards and image files only -- it never
imports nuscenes. That keeps the two halves of the project independently
testable and keeps the devkit out of the dataloader workers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from occupancy.data.nuscenes_adapter import SampleData


@dataclass(frozen=True)
class Shard:
    target: np.ndarray             # (nx, ny, nz) uint8; 255 = unobserved
    image_paths: list[str]         # relative to the dataset root
    camera_names: list[str]
    intrinsics: np.ndarray         # (6, 3, 3)
    T_ego_from_cam: np.ndarray     # (6, 4, 4)
    T_global_from_ego: np.ndarray  # (4, 4)
    meta: dict


def write_shard(
    out_dir: Path, sample: SampleData, target: np.ndarray, dataroot: str
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sample.sample_token}.npz"

    root = str(Path(dataroot))
    np.savez_compressed(
        path,
        target=target,
        # Stored relative so a shard directory is portable across machines;
        # the Dataset re-joins with its own configured dataroot.
        image_paths=np.array(
            [str(Path(c.image_path).relative_to(root)) for c in sample.cameras]
        ),
        camera_names=np.array([c.name for c in sample.cameras]),
        intrinsics=np.stack([c.intrinsic for c in sample.cameras]),
        T_ego_from_cam=np.stack([c.T_ego_from_cam for c in sample.cameras]),
        # Kept so ego motion between samples can be derived later, which is
        # what temporal fusion needs. Cheap now, expensive to retrofit.
        T_global_from_ego=sample.T_global_from_ego,
        meta=json.dumps(
            {
                "sample_token": sample.sample_token,
                "scene_name": sample.scene_name,
                "timestamp_us": sample.timestamp_us,
                "num_keyframes": sample.num_keyframes,
                "num_points": len(sample.cloud),
            }
        ),
    )
    return path


def read_shard(path: Path) -> Shard:
    with np.load(path, allow_pickle=False) as z:
        return Shard(
            target=z["target"],
            image_paths=[str(s) for s in z["image_paths"]],
            camera_names=[str(s) for s in z["camera_names"]],
            intrinsics=z["intrinsics"],
            T_ego_from_cam=z["T_ego_from_cam"],
            T_global_from_ego=z["T_global_from_ego"],
            meta=json.loads(str(z["meta"])),
        )