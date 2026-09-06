"""Evaluation metrics.

Voxel mIoU only. RayIoU -- which casts rays into predicted and ground-truth
volumes and compares first-hit depth -- is the metric current work prefers,
because voxel mIoU cannot see depth error and rewards thickened surfaces. Not
implemented: no model is trained here, so there is nothing to evaluate.
"""
from __future__ import annotations

import torch


def voxel_miou(pred: torch.Tensor, target: torch.Tensor,
               num_classes: int, ignore_index: int) -> dict[str, float]:
    valid = target != ignore_index
    pred, target = pred[valid], target[valid]

    ious: list[float] = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        if union:
            ious.append((p & t).sum().item() / union)

    return {"mIoU": sum(ious) / len(ious) if ious else 0.0,
            "classes_present": len(ious)}