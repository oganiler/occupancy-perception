"""Correct voxels inside reference-time vehicle boxes. See D-011.

Pass 1 casts free rays from every keyframe in the aggregation window. A vehicle
occupying volume B at t_ref was elsewhere earlier in the window, so beams from
those keyframes passed through B and marked it FREE. Pass 2 marks only the near
surface OCCUPIED, because that is where the returns landed. The result is a
vehicle shell containing free space.

Measured on one sample over six vehicle boxes and 2,761 interior voxels:
unobserved 56.6%, free 26.9%, occupied 16.5%. The free fraction split by object
(6-8% for two boxes, 54-65% for two others) is consistent with parked versus
moving vehicles, which is the mechanism above.

Neither D-003 nor the pass-3 guard covers this. The offending rays terminate on
static geometry, so they carry no annotation flag, and pass 1 runs before there
is anything to guard against.
"""
from __future__ import annotations

import numpy as np

from occupancy.data.nuscenes_adapter import Box3D
from occupancy.gt.backends.base import FREE, OCCUPIED, UNOBSERVED
from occupancy.gt.grid import GridSpec

# Box-to-volume fill ratio is high enough only for vehicles. A pedestrian's
# box is mostly air; a traffic cone's almost entirely so.
VEHICLE_PREFIXES = (
    "vehicle.car", "vehicle.truck", "vehicle.bus",
    "vehicle.trailer", "vehicle.construction",
)


def _voxel_centers_ego(spec: GridSpec) -> np.ndarray:
    """(3, V) voxel centres in the ego frame, matching the grid's flattening."""
    nx, ny, nz = spec.dims
    o, s = spec.origin_m, spec.voxel_m
    ax = [np.arange(n) * si + oi + si / 2.0
          for n, si, oi in zip((nx, ny, nz), s, o)]
    g = np.meshgrid(*ax, indexing="ij")
    return np.stack([a.reshape(-1) for a in g])


def apply_box_fill(
    state: np.ndarray,          # (nx, ny, nz) uint8, modified in place
    semantics: np.ndarray,      # (nx, ny, nz) uint8, modified in place
    boxes: tuple[Box3D, ...],
    spec: GridSpec,
    mode: str,                  # none | correct | assert
) -> dict[str, int]:
    """Returns counts of voxels changed, for reporting."""
    if mode == "none" or not boxes:
        return {"corrected": 0, "asserted": 0, "interior": 0}

    centers = _voxel_centers_ego(spec)                      # (3, V)
    ones = np.ones((1, centers.shape[1]))
    centers_h = np.vstack([centers, ones])                  # (4, V)

    flat_state = state.reshape(-1)
    flat_sem = semantics.reshape(-1)
    corrected = asserted = interior = 0

    for box in boxes:
        if not box.name.startswith(VEHICLE_PREFIXES):
            continue

        # ego -> box-local, then an axis-aligned test against the half extents.
        local = (np.linalg.inv(box.T_ego_from_box) @ centers_h)[:3]
        inside = np.all(np.abs(local) <= box.half_extent_m[:, None], axis=0)
        if not inside.any():
            continue
        interior += int(inside.sum())

        # (a) correction: remove free space the data does not support at t_ref.
        free_inside = inside & (flat_state == FREE)
        flat_state[free_inside] = UNOBSERVED
        corrected += int(free_inside.sum())

        if mode == "assert":
            # (b) assertion: adds a claim no sensor made, and approximates the
            # vehicle as a cuboid. Off by default.
            unk = inside & (flat_state == UNOBSERVED)
            flat_state[unk] = OCCUPIED
            flat_sem[unk] = box.label
            asserted += int(unk.sum())

    return {"corrected": corrected, "asserted": asserted, "interior": interior}