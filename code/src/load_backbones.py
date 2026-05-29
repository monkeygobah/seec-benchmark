from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from src.backbones.vit import VIT_BACKBONES, load_timm_vit_backbone


def _ensure_torch_home() -> None:
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        Path(torch_home).mkdir(parents=True, exist_ok=True)
        return

    default_cache = Path.home() / ".cache" / "torch"
    try:
        default_cache.mkdir(parents=True, exist_ok=True)
        if os.access(default_cache, os.W_OK):
            os.environ["TORCH_HOME"] = str(default_cache)
            return
    except OSError:
        pass

    fallback_cache = Path(tempfile.gettempdir()) / "torch_cache"
    fallback_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(fallback_cache)



def load_segmentation_encoder(ckpt_path="hp_tune.pth"):
    from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
    from torchvision.models.segmentation import deeplabv3_resnet101

    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 6, kernel_size=1)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    encoder = model.backbone
    return encoder

def load_resnet101_encoder(pretrained: bool = False):
    from torchvision.models import ResNet101_Weights, resnet101

    weights = None
    if pretrained:
        _ensure_torch_home()
        weights = ResNet101_Weights.IMAGENET1K_V1
    model = resnet101(weights=weights)
    return nn.Sequential(*list(model.children())[:-2])


def load_encoder_backbone(init, seg_ckpt=None, backbone="resnet101"):
    backbone = "resnet101" if backbone is None else str(backbone)
    if backbone in VIT_BACKBONES:
        return load_timm_vit_backbone(backbone=backbone, init=init)
    if backbone != "resnet101":
        raise ValueError(f"Unknown encoder backbone: {backbone}")
    if init == "seg_init":
        return load_segmentation_encoder(seg_ckpt)
    if init == "imagenet":
        return load_resnet101_encoder(pretrained=True)
    if init == "random":
        return load_resnet101_encoder(pretrained=False)
    raise ValueError(f"Unknown encoder init mode: {init}")
