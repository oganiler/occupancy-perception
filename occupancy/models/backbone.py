"""Shared image backbone.

One set of weights processes all six views: the camera axis is folded into the
batch, so the backbone never knows how many cameras there are. Initialised from
ImageNet weights, which is why the Dataset applies ImageNet normalisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

# Output channels at each stride. ResNet18/34 use BasicBlock (64..512);
# ResNet50+ use Bottleneck (256..2048), so the widths differ by 4x and the
# voxel encoder must be told which it is getting.
_CHANNELS: dict[str, dict[int, int]] = {
    "resnet18": {8: 128, 16: 256, 32: 512},
    "resnet34": {8: 128, 16: 256, 32: 512},
    "resnet50": {8: 512, 16: 1024, 32: 2048},
}

# ResNet is stride-32 by default, so a smaller stride means truncating layers.
_KEEP_LAYERS = {8: 6, 16: 7, 32: 8}


class ImageBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool, out_stride: int) -> None:
        super().__init__()
        if name not in _CHANNELS:
            raise ValueError(
                f"unsupported backbone {name!r}; known: {sorted(_CHANNELS)}"
            )
        if out_stride not in _KEEP_LAYERS:
            raise ValueError(f"out_stride must be 8, 16 or 32, got {out_stride}")

        net = getattr(torchvision.models, name)(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        layers = [net.conv1, net.bn1, net.relu, net.maxpool,
                  net.layer1, net.layer2, net.layer3, net.layer4]
        self.net = nn.Sequential(*layers[: _KEEP_LAYERS[out_stride]])
        self.out_channels = _CHANNELS[name][out_stride]
        self.out_stride = out_stride

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, C, H/stride, W/stride). N folds T and cameras."""
        return self.net(imgs)