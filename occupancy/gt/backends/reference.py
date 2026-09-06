"""Pure-Python reference voxelizer. Correctness over speed.

Two passes, which makes the D-003 precedence rule (occupied beats free)
a structural property rather than an ordering accident:

  pass 1  free-space ray traversal          -> FREE
  pass 2  endpoint marking and label votes  -> OCCUPIED, overwriting FREE

Ray traversal is the Amanatides-Woo 3D DDA, run in voxel coordinates so the
voxel size becomes 1 and the arithmetic stays in units of "voxels crossed".
"""
from __future__ import annotations

import numpy as np

from occupancy.gt.grid import GridSpec
from occupancy.gt.backends.base import FREE, OCCUPIED, UNOBSERVED
from occupancy.labels import FREE_CLASS, NUM_CLASSES

_NUM_SEMANTIC = NUM_CLASSES - 1   # 0..16; 17 (free) is not a point label


class ReferenceVoxelizer:
    def __init__(self, spec: GridSpec, free_ray_stride: int) -> None:
        self.spec = spec
        self.free_ray_stride = free_ray_stride

    def build(
        self,
        points_m: np.ndarray,
        labels: np.ndarray,
        sensor_origins_m: np.ndarray,
        cast_free_ray: np.ndarray,
        extra_free_points_m: np.ndarray | None = None,      # (M, 3)
        extra_free_origins_m: np.ndarray | None = None,     # (M, 3)        
    ) -> tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = self.spec.dims
        state = np.full(self.spec.num_voxels, UNOBSERVED, dtype=np.uint8)

        starts = self.spec.to_voxel_coords(sensor_origins_m.astype(np.float64))
        ends = self.spec.to_voxel_coords(points_m.astype(np.float64))

        # ---- pass 1: free space -----------------------------------------
        ray_idx = np.flatnonzero(cast_free_ray)[:: self.free_ray_stride]
        max_steps = 2 * (nx + ny + nz)
        for i in ray_idx:
            _dda(starts[i], ends[i], state, nx, ny, nz, max_steps)

        # ---- pass 2: endpoints, vectorised ------------------------------
        vox = np.floor(ends).astype(np.int64)                  # (N, 3) all points at once
        ok = self.spec.in_bounds(vox)                          # (N,)    all bounds checks
        vox, lab = vox[ok], labels[ok]
        flat = (vox[:, 0] * ny + vox[:, 1]) * nz + vox[:, 2]    # (M,)  3D index -> 1D

        state[flat] = OCCUPIED # one scattered write

        votes = np.zeros((self.spec.num_voxels, _NUM_SEMANTIC), dtype=np.uint16)
        np.add.at(votes, (flat, lab), 1) # one scattered accumulate

        semantics = np.full(self.spec.num_voxels, FREE_CLASS, dtype=np.uint8)
        occ = np.flatnonzero(state == OCCUPIED)
        # np.argmax breaks ties toward the lowest class index -> deterministic.
        semantics[occ] = votes[occ].argmax(axis=1).astype(np.uint8)

        # ---- pass 3: non-keyframe free space (D-001 refinement) ----------
        # Additive by construction: _dda only fills UNOBSERVED voxels, so this
        # cannot contradict pass 1 or pass 2 regardless of ordering.
        if extra_free_points_m is not None or extra_free_origins_m is not None:
            if extra_free_points_m is None or extra_free_origins_m is None:
                raise ValueError(
                    "extra_free_points_m and extra_free_origins_m must be "
                    "given together"
                )
            if len(extra_free_points_m) != len(extra_free_origins_m):
                raise ValueError(
                    f"length mismatch: {len(extra_free_points_m)} points vs "
                    f"{len(extra_free_origins_m)} origins"
                )

            s2 = self.spec.to_voxel_coords(extra_free_origins_m.astype(np.float64))
            e2 = self.spec.to_voxel_coords(extra_free_points_m.astype(np.float64))
            for i in range(len(e2)):
                _dda(s2[i], e2[i], state, nx, ny, nz, max_steps)    

        return (
            state.reshape(nx, ny, nz),
            semantics.reshape(nx, ny, nz),
        )


def _dda(o: np.ndarray, e: np.ndarray, state: np.ndarray,
         nx: int, ny: int, nz: int, max_steps: int) -> None:
    """Amanatides-Woo traversal from o to e, in voxel coordinates.

    Marks every traversed voxel FREE, excluding the endpoint voxel, which
    pass 2 will claim. `t` is parameterised on [0, 1] because the direction
    vector is (e - o), so t == 1 is exactly the endpoint.
    """
    d = e - o
    ix, iy, iz = int(np.floor(o[0])), int(np.floor(o[1])), int(np.floor(o[2]))
    ex, ey, ez = int(np.floor(e[0])), int(np.floor(e[1])), int(np.floor(e[2]))

    step = [0, 0, 0]
    t_max = [np.inf, np.inf, np.inf]
    t_delta = [np.inf, np.inf, np.inf]
    cur = (o[0], o[1], o[2])

    for k in range(3):
        if d[k] > 0.0:
            step[k] = 1
            t_max[k] = (np.floor(cur[k]) + 1.0 - cur[k]) / d[k]
            t_delta[k] = 1.0 / d[k]
        elif d[k] < 0.0:
            step[k] = -1
            t_max[k] = (cur[k] - np.floor(cur[k])) / -d[k]
            t_delta[k] = 1.0 / -d[k]

    for _ in range(max_steps):
        if ix == ex and iy == ey and iz == ez:
            return
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            flat = (ix * ny + iy) * nz + iz
            if state[flat] == UNOBSERVED:
                state[flat] = FREE

        axis = 0 if t_max[0] <= t_max[1] and t_max[0] <= t_max[2] else (
            1 if t_max[1] <= t_max[2] else 2
        )
        if t_max[axis] > 1.0:
            return
        t_max[axis] += t_delta[axis]
        if axis == 0:
            ix += step[0]
        elif axis == 1:
            iy += step[1]
        else:
            iz += step[2]