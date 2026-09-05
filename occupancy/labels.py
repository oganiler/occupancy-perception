"""Mapping from nuScenes-lidarseg fine classes to the Occ3D-nuScenes class set.

The devkit's LidarsegClassMapper already collapses the 32 fine lidarseg
classes onto indices 0..16, which coincide exactly with Occ3D-nuScenes
classes 0..16. We therefore reuse it rather than maintaining our own table.

Occ3D class 17 (`free`) is NOT a point label: it is assigned by ray traversal
in the voxelizer. Point labels produced here are always in [0, 16].
"""
from __future__ import annotations

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper

NUM_CLASSES: int = 18       # 0..16 semantic + 17 free
FREE_CLASS: int = 17
OTHERS_CLASS: int = 0       # devkit calls this 'ignore'
UNOBSERVED_LABEL: int = 255   # sentinel in the stored target; not a class

OCC3D_CLASS_NAMES: tuple[str, ...] = (
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation", "free",
)


def build_fine_to_occ3d_lut(nusc: NuScenes) -> np.ndarray:
    """Build a lookup table from lidarseg fine index to Occ3D class index.

    Returns
    -------
    (32,) uint8 array. Index it with raw lidarseg labels to get Occ3D classes.
    """
    mapper = LidarsegClassMapper(nusc)
    fine2coarse: dict[int, int] = mapper.get_fine_idx_2_coarse_idx()

    lut = np.zeros(max(fine2coarse) + 1, dtype=np.uint8)
    for fine_idx, coarse_idx in fine2coarse.items():
        lut[fine_idx] = coarse_idx

    assert lut.max() <= 16, "point labels must never be 17 (free)"
    return lut


def remap_labels(raw_labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map raw lidarseg labels to Occ3D classes.

    Parameters
    ----------
    raw_labels : (N,) uint8, values in [0, 31]
    lut        : (32,) uint8, from build_fine_to_occ3d_lut

    Returns
    -------
    (N,) uint8, values in [0, 16]
    """
    return lut[raw_labels]