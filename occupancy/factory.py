"""The single place where configuration becomes objects.

Every other module takes explicit parameters and knows nothing about the
config format.
"""
from __future__ import annotations

from omegaconf import DictConfig

from occupancy.data.nuscenes_adapter import NuScenesAdapter
from occupancy.gt.backends.base import VoxelizerBackend
from occupancy.gt.grid import GridSpec


def build_grid_spec(cfg: DictConfig) -> GridSpec:
    return GridSpec(
        point_cloud_range=tuple(cfg.grid.point_cloud_range),
        voxel_size=tuple(cfg.grid.voxel_size),
    )


def build_adapter(cfg: DictConfig) -> NuScenesAdapter:
    if cfg.gt.aggregation_mode != "keyframes":
        raise NotImplementedError(
            f"aggregation_mode={cfg.gt.aggregation_mode!r} not implemented; "
            "see docs/decisions.md D-001"
        )
    return NuScenesAdapter(
        dataroot=cfg.dataset.root,
        version=cfg.dataset.version,
        window_keyframes=cfg.gt.window_keyframes,
        min_distance_m=cfg.gt.min_distance_m,
        separate_dynamic=cfg.gt.separate_dynamic,
        camera_names=tuple(cfg.data.cameras),
    )


def build_voxelizer(cfg: DictConfig, spec: GridSpec) -> VoxelizerBackend:
    if cfg.gt.backend == "python":
        from occupancy.gt.backends.reference import ReferenceVoxelizer
        return ReferenceVoxelizer(spec, free_ray_stride=cfg.gt.free_ray_stride)

    if cfg.gt.backend == "cpp":
        # Not built. The native backend plugs in here behind the same
        # VoxelizerBackend protocol, with no change to any caller.
        raise NotImplementedError(
            "gt.backend='cpp' is not available in this build. "
            "Use gt.backend='python'."
        )

    raise ValueError(f"unknown backend: {cfg.gt.backend!r}")