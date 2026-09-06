"""Masked cross-entropy over the occupancy grid.

The stored target is consumed directly: 0..16 semantic, 17 free, 255
unobserved. ignore_index=255 means unobserved voxels contribute no gradient
(D-005). They are not a class the model predicts -- the head has 18 logits and
none of them means "unobserved".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropy(nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """logits (B, K, X, Y, Z), target (B, X, Y, Z) -> scalar, stats."""
        loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index)

        with torch.no_grad():
            supervised = (target != self.ignore_index)
            stats = {
                "supervised_frac": supervised.float().mean().item(),
                # Same logits, every voxel supervised. The gap is the mask's
                # effect, not a different model. See D-005.
                "loss_unmasked": F.cross_entropy(
                    logits, target.clamp(max=logits.shape[1] - 1)
                ).item(),
            }
        return loss, stats