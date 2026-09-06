"""Assembled placeholder network.

All four stages are injected. Replacing any one -- a different backbone, LSS
instead of naive projection, a deeper head -- touches only the factory.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from typing import Protocol, runtime_checkable

from occupancy.gt.grid import GridSpec
from occupancy.labels import NUM_CLASSES

@runtime_checkable
class FeatureExtractor(Protocol):
    """What OccupancyNet needs from a backbone.

    nn.Module alone is too weak: its __getattr__ returns Tensor | Module, so
    the checker cannot see out_channels, and nothing stops a module without it
    being passed in.
    """

    out_channels: int
    out_stride: int

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor: ...

def voxel_centers(spec: GridSpec) -> torch.Tensor:
    """(V, 3) ego-frame centres, ordered to match the target's flattening."""
    nx, ny, nz = spec.dims
    o, s = spec.origin_m, spec.voxel_m
    ax = [torch.arange(n) * si + oi + si / 2.0
          for n, si, oi in zip((nx, ny, nz), s, o)]
    gx, gy, gz = torch.meshgrid(*ax, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).float()


class OccupancyNet(nn.Module):
    def __init__(
        self,
        backbone: FeatureExtractor,
        view_transform: nn.Module,
        spec: GridSpec,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.view_transform = view_transform
        self.spec = spec
        self.register_buffer("voxel_centers", voxel_centers(spec))

        self.voxel_encoder = nn.Sequential(
            nn.Conv3d(backbone.out_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(hidden_channels, NUM_CLASSES, 1)

        self.last_visible_frac: float = 0.0 #initialized to 0.0, updated after each forward pass

    def forward(self, batch: dict) -> torch.Tensor:
        imgs = batch["imgs"]                       # (B, T, N, 3, H, W)
        B, T, N = imgs.shape[:3]
        if T != 1:
            raise NotImplementedError("temporal fusion is not implemented")

        # Fold T and cameras into the batch: one shared backbone, N passes.
        feats = self.backbone(imgs.reshape(B * T * N, *imgs.shape[3:]))
        feats = feats.reshape(B, N, *feats.shape[1:])              # (B,N,C,h,w)

        voxels = self.view_transform(
            feats,
            batch["intrinsics"][:, 0],
            batch["T_ego_from_cam"][:, 0],
            self.voxel_centers,
        )                                                          # (B, C, V)

        # Fraction of voxels that projected into at least one camera. Voxels
        # seen by none receive zero features and send no gradient to the
        # backbone -- a structural property of backward projection.
        self.last_visible_frac = float(
            (voxels.abs().sum(dim=1) > 0).float().mean()
        )

        nx, ny, nz = self.spec.dims
        voxels = voxels.reshape(B, -1, nx, ny, nz)
        return self.head(self.voxel_encoder(voxels))    # (B, 18, nx, ny, nz)