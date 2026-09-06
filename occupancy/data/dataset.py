"""Torch Dataset over generated GT shards.

Imports numpy, torch and PIL only. The nuScenes devkit is deliberately absent:
everything this needs was captured into the shards at generation time, so the
devkit never enters a dataloader worker and this module is testable against
synthetic shards with no dataset mounted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from occupancy.gt.shard import read_shard

# Native nuScenes camera resolution.
NATIVE_H, NATIVE_W = 900, 1600

# ImageNet statistics: the backbone is initialised from torchvision weights
# trained under this normalisation, so transfer learning requires matching it.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OccupancyDataset(Dataset):
    """One item = one keyframe: six images, their calibration, and the target.

    The temporal axis is present with T = 1 (D-007 note: `num_frames`). It
    exists now so that adding temporal fusion later does not require changing
    the batch contract, the collate function and every model signature at once.
    """

    def __init__(
        self,
        shard_dir: str,
        dataroot: str,
        image_size: tuple[int, int],   # (H, W)
        cameras: tuple[str, ...],
        num_frames: int,
        split: str = "train",
        val_scenes: int = 2,
    ) -> None:
        if num_frames != 1:
            raise NotImplementedError(
                f"num_frames={num_frames}: temporal fusion is not implemented. "
                "The batch already carries the T axis; see docs/decisions.md."
            )

        self.dataroot = Path(dataroot)
        self.image_size = image_size
        self.cameras = cameras
        self.num_frames = num_frames

        # A pure scale, so one correction covers both focal length and
        # principal point (D-007). Verify the config preserves aspect, or the
        # single scale would be wrong on one axis.
        self.scale = image_size[1] / NATIVE_W
        if abs(image_size[0] / NATIVE_H - self.scale) > 1e-6:
            raise ValueError(
                f"image_size {image_size} does not preserve the native "
                f"{NATIVE_H}x{NATIVE_W} aspect ratio. D-007 assumes a pure "
                f"scale; a crop would need a principal-point offset too."
            )

        self.paths = self._select_split(Path(shard_dir), split, val_scenes)
        if not self.paths:
            raise RuntimeError(
                f"no shards for split '{split}' in {shard_dir}. "
                "Generate them with: python -m occupancy.cli gt"
            )

    # -- split -------------------------------------------------------------

    @staticmethod
    def _select_split(
        shard_dir: Path, split: str, val_scenes: int
    ) -> list[Path]:
        """Split by scene, never by sample.

        Samples within a scene are 0.5 s apart and highly correlated. A random
        sample-level split would leak near-duplicate frames across the
        boundary and report a meaningless validation score.
        """
        by_scene: dict[str, list[Path]] = {}
        for p in sorted(shard_dir.glob("*.npz")):
            scene = read_shard(p).meta["scene_name"]
            by_scene.setdefault(scene, []).append(p)

        scenes = sorted(by_scene)
        val = set(scenes[-val_scenes:])            # deterministic, no shuffle
        chosen = val if split == "val" else set(scenes) - val
        return [p for s in sorted(chosen) for p in by_scene[s]]

    # -- torch api ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        shard = read_shard(self.paths[idx])
        H, W = self.image_size

        imgs, intrinsics, extrinsics = [], [], []
        for cam in self.cameras:
            i = shard.camera_names.index(cam)

            img = Image.open(self.dataroot / shard.image_paths[i]).convert("RGB")
            img = img.resize((W, H), Image.Resampling.BILINEAR)

            arr = np.asarray(img, dtype=np.float32) / 255.0      # (H, W, 3)
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            imgs.append(arr.transpose(2, 0, 1))                  # (3, H, W)

            # The single intrinsic correction. Scaling rows 0 and 1 scales
            # fx, fy, cx and cy together, which is exactly right for a pure
            # resize and exactly wrong if a crop were also applied.
            K = shard.intrinsics[i].astype(np.float32).copy()
            K[:2, :] *= self.scale
            intrinsics.append(K)

            extrinsics.append(shard.T_ego_from_cam[i].astype(np.float32))

        def stack_t(xs):                     # add the T axis, currently 1
            return torch.from_numpy(np.stack(xs)).unsqueeze(0)

        return {
            "imgs": stack_t(imgs),                     # (T, 6, 3, H, W)
            "intrinsics": stack_t(intrinsics),         # (T, 6, 3, 3)
            "T_ego_from_cam": stack_t(extrinsics),     # (T, 6, 4, 4)
            # int64 for cross_entropy; 255 stays as the ignore_index sentinel.
            "target": torch.from_numpy(shard.target.astype(np.int64)),
            "sample_token": shard.meta["sample_token"],
            "scene_name": shard.meta["scene_name"],
        }


def collate(batch: list[dict]) -> dict:
    """Stack tensors, keep string metadata as lists."""
    out: dict = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        out[key] = (
            torch.stack(values) if torch.is_tensor(values[0]) else values
        )
    return out