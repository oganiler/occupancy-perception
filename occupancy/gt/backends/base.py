"""Contract every voxelizer backend must satisfy.

Backends are interchangeable and MUST produce bit-identical output for
identical input. This is asserted by tests/test_parity.py.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

UNOBSERVED: int = 0
FREE: int = 1
OCCUPIED: int = 2


class VoxelizerBackend(Protocol):
    def build(
        self,
        points_m: np.ndarray,          # (N, 3) float32, ego frame
        labels: np.ndarray,            # (N,)   uint8, Occ3D 0..16
        sensor_origins_m: np.ndarray,  # (N, 3) float32, ego frame
        cast_free_ray: np.ndarray,     # (N,)   bool, see D-003
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (state, semantics), each (nx, ny, nz) uint8.

        state:     UNOBSERVED | FREE | OCCUPIED
        semantics: Occ3D class per voxel; FREE_CLASS where state is FREE,
                   undefined where state is UNOBSERVED.
        """
        ...