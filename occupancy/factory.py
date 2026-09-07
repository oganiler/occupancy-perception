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
        free_sweep_stride=cfg.gt.free_sweep_stride,
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

def build_dataset(cfg: DictConfig, split: str = "train"):
    from occupancy.data.dataset import OccupancyDataset
    return OccupancyDataset(
        shard_dir=cfg.gt.output_dir,
        dataroot=cfg.dataset.root,
        image_size=tuple(cfg.data.image_size),
        cameras=tuple(cfg.data.cameras),
        num_frames=cfg.data.num_frames,
        split=split,
    )

def build_model(cfg: DictConfig, spec: GridSpec):
    from occupancy.models.backbone import ImageBackbone
    from occupancy.models.occupancy_net import OccupancyNet
    from occupancy.models.view_transform import NaiveProjection

    backbone = ImageBackbone(
        name=cfg.model.backbone.name,
        pretrained=cfg.model.backbone.pretrained,
        out_stride=cfg.model.backbone.out_stride,
    )
    if cfg.model.view_transform != "naive":
        raise NotImplementedError(
            f"view_transform={cfg.model.view_transform!r} not implemented; "
            "'lss' and 'deformable_attention' plug in behind the same protocol"
        )
    return OccupancyNet(
        backbone=backbone,
        view_transform=NaiveProjection(tuple(cfg.data.image_size)),
        spec=spec,
        lift_channels=cfg.model.lift_channels,
        hidden_channels=cfg.model.hidden_channels,
    )


def build_loss(cfg: DictConfig):
    from occupancy.models.loss import MaskedCrossEntropy
    return MaskedCrossEntropy(ignore_index=cfg.loss.ignore_index)