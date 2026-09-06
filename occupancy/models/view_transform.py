"""Lifting image features into the 3D voxel grid.

Backward projection: voxel centres are projected into each image and features
are sampled where they land. This avoids predicting depth -- the 3D location is
chosen, not inferred -- at the cost of depth ambiguity, since every voxel along
a camera ray samples the same pixel.

Forward projection (LSS) and deformable cross-attention (BEVFormer) are the
alternatives; both implement this same protocol.
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

# A projection is only meaningful at least this far in front of the lens.
# Below it the perspective divide is unstable and the point is inside the
# vehicle body. A near-plane epsilon, not a tuned parameter.
NEAR_PLANE_M = 0.1


class ViewTransform(Protocol):
    def forward(
        self,
        feats: torch.Tensor,            # (B, N, C, h, w)
        intrinsics: torch.Tensor,       # (B, N, 3, 3), scaled to image_hw
        T_ego_from_cam: torch.Tensor,   # (B, N, 4, 4)
        voxel_centers: torch.Tensor,    # (V, 3) ego frame
    ) -> torch.Tensor:                  # (B, C, V)
        ...


class NaiveProjection(nn.Module):
    """Project voxel centres into every view and average the samples."""

    def __init__(self, image_hw: tuple[int, int]) -> None:
        super().__init__()
        self.image_hw = image_hw

    def forward(self, feats, intrinsics, T_ego_from_cam, voxel_centers):
        B, N, C = feats.shape[:3]
        H, W = self.image_hw
        V = voxel_centers.shape[0]

        ones = voxel_centers.new_ones(V, 1)
        voxels_h = torch.cat([voxel_centers, ones], dim=1).T      # (4, V)

        acc = feats.new_zeros(B, C, V)
        hits = feats.new_zeros(B, 1, V)

        T_cam_from_ego = torch.inverse(T_ego_from_cam)            # (B, N, 4, 4)

        for b in range(B):
            for n in range(N):
                voxels_cam = (T_cam_from_ego[b, n] @ voxels_h)[:3]    # (3, V)
                uvw = intrinsics[b, n] @ voxels_cam                   # (3, V)
                z = uvw[2]
                u, v = uvw[0] / z, uvw[1] / z

                # z > 0 rejects voxels BEHIND the camera. Without it they
                # project to valid-looking pixels, because the perspective
                # divide cancels the sign, and the grid picks up mirrored
                # content from behind each view.
                ok = ((z > NEAR_PLANE_M)
                      & (u >= 0) & (u < W) & (v >= 0) & (v < H))

                # Normalise against the IMAGE size, not the feature size:
                # grid_sample maps [-1, 1] onto the full feature extent, which
                # corresponds to the full image extent whatever the stride.
                grid = torch.stack([u / W * 2 - 1, v / H * 2 - 1], dim=-1)
                sampled = F.grid_sample(
                    feats[b, n].unsqueeze(0),
                    grid.view(1, 1, V, 2),
                    mode="bilinear", padding_mode="zeros",
                    align_corners=False,
                ).view(C, V)

                acc[b] += sampled * ok
                hits[b] += ok

        # Voxels seen by no camera keep zero features rather than NaN.
        return acc / hits.clamp(min=1.0)