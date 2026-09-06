"""Typed configuration. Loaded once at the entrypoint, never inside a module.

The structured schema means a mistyped key or a wrong type fails at load
rather than silently falling back to a default. Modules take explicit
parameters; only the factory sees this object.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass
class DatasetConfig:
    root: str = "/data/nuscenes"
    version: str = "v1.0-mini"


@dataclass
class GridConfig:
    point_cloud_range: list[float] = field(
        default_factory=lambda: [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    )
    voxel_size: list[float] = field(default_factory=lambda: [0.4, 0.4, 0.4])


@dataclass
class GTConfig:
    aggregation_mode: str = "keyframes"      # keyframes | sweeps  (D-001)
    window_keyframes: int = 4
    separate_dynamic: bool = True
    min_distance_m: float = 1.0
    free_ray_stride: int = 1                 # 1 = every ray; >1 subsamples
    free_sweep_stride: int = 0
    backend: str = "python"                  # python | cpp
    generate_camera_mask: bool = False
    output_dir: str = "/workspace/gt_shards"


@dataclass
class DataConfig:
    num_frames: int = 1
    image_size: list[int] = field(default_factory=lambda: [256, 704])
    cameras: list[str] = field(
        default_factory=lambda: [
            "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
        ]
    )


@dataclass
class LossConfig:
    use_camera_mask: bool = False
    ignore_index: int = 255


@dataclass
class BackboneConfig:
    name: str = "resnet18"          # resnet18 | resnet34 | resnet50
    pretrained: bool = True         # ImageNet init; Dataset normalisation assumes this
    out_stride: int = 16            # 8 | 16 | 32


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    view_transform: str = "naive"   # naive | lss | deformable_attention
    hidden_channels: int = 64

@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    gt: GTConfig = field(default_factory=GTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)


DEFAULT_CONFIG_PATH = "configs/default.yaml"


def load_config(
    path: str = DEFAULT_CONFIG_PATH,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Merge schema <- yaml <- CLI dotlist overrides.

    Overrides look like ["gt.window_keyframes=8", "gt.backend=cpp"].
    """
    file_cfg = OmegaConf.load(path)
    if not isinstance(file_cfg, DictConfig):
        raise TypeError(
            f"{path}: top level must be a mapping, "
            f"got {type(file_cfg).__name__}"
        )

    merged = OmegaConf.merge(OmegaConf.structured(Config), file_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    # merge() is annotated ListConfig | DictConfig because YAML permits a list
    # at the top level. Merging two DictConfigs always yields a DictConfig, so
    # this narrows the type rather than testing anything that can fail.
    assert isinstance(merged, DictConfig)

    _validate(merged)
    return merged


def _validate(cfg: DictConfig) -> None:
    lo = cfg.grid.point_cloud_range[:3]
    hi = cfg.grid.point_cloud_range[3:]
    for axis, (a, b, v) in enumerate(zip(lo, hi, cfg.grid.voxel_size)):
        if b <= a:
            raise ValueError(f"grid axis {axis}: max {b} must exceed min {a}")
        n = (b - a) / v
        if abs(n - round(n)) > 1e-6:
            raise ValueError(
                f"grid axis {axis}: extent {b - a} is not an integer multiple "
                f"of voxel size {v}"
            )
    if cfg.gt.free_ray_stride < 1:
        raise ValueError("gt.free_ray_stride must be >= 1")
    if cfg.loss.use_camera_mask and not cfg.gt.generate_camera_mask:
        raise ValueError(
            "loss.use_camera_mask requires gt.generate_camera_mask"
        )

    from occupancy.labels import UNOBSERVED_LABEL
    if cfg.loss.ignore_index != UNOBSERVED_LABEL:
        raise ValueError(
            f"loss.ignore_index ({cfg.loss.ignore_index}) must equal the shard "
            f"sentinel ({UNOBSERVED_LABEL})"
        )

    if cfg.gt.free_sweep_stride < 0:
        raise ValueError("gt.free_sweep_stride must be >= 0 (0 disables)")    