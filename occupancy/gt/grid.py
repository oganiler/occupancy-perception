"""Voxel grid geometry. Dimensions are derived, never configured."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Axis-aligned voxel grid in the ego frame of the reference keyframe.

    `dims` is deliberately absent from the config: it is a function of range
    and voxel size, so the two cannot disagree.
    """

    point_cloud_range: tuple[float, float, float, float, float, float]
    voxel_size: tuple[float, float, float]

    @property
    def origin_m(self) -> np.ndarray:
        return np.array(self.point_cloud_range[:3], dtype=np.float64)

    @property
    def extent_m(self) -> np.ndarray:
        return np.array(self.point_cloud_range[3:], dtype=np.float64) - self.origin_m

    @property
    def voxel_m(self) -> np.ndarray:
        return np.array(self.voxel_size, dtype=np.float64)

    @property
    def dims(self) -> tuple[int, int, int]:
        d = np.round(self.extent_m / self.voxel_m).astype(int)
        return int(d[0]), int(d[1]), int(d[2])

    @property
    def num_voxels(self) -> int:
        nx, ny, nz = self.dims
        return nx * ny * nz

    def to_voxel_coords(self, points_m: np.ndarray) -> np.ndarray:
        """(N, 3) metres -> (N, 3) continuous voxel coordinates."""
        return (points_m - self.origin_m) / self.voxel_m

    def in_bounds(self, idx: np.ndarray) -> np.ndarray:
        """(N, 3) integer voxel indices -> (N,) bool."""
        d = np.array(self.dims)
        return np.all((idx >= 0) & (idx < d), axis=1)