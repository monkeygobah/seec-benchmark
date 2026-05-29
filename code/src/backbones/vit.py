from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ViTBackboneSpec:
    model_name: str
    imagenet_model_name: str
    feat_dim: int
    patch_size: int


VIT_BACKBONES: dict[str, ViTBackboneSpec] = {
    "vit_base_patch16_224": ViTBackboneSpec(
        model_name="vit_base_patch16_224",
        imagenet_model_name="vit_base_patch16_224.augreg_in1k",
        feat_dim=768,
        patch_size=16,
    ),
    "vit_large_patch16_224": ViTBackboneSpec(
        model_name="vit_large_patch16_224",
        imagenet_model_name="vit_large_patch16_224.augreg_in1k",
        feat_dim=1024,
        patch_size=16,
    ),
}


def patch_tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Patch tokens must have shape [B, N, D], got {tuple(tokens.shape)}")

    batch_size, num_tokens, dim = tokens.shape
    grid_size = int(num_tokens**0.5)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Patch token count must form a square grid, got {num_tokens}")

    return tokens.transpose(1, 2).reshape(batch_size, dim, grid_size, grid_size).contiguous()


def _tokens_from_forward_features(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
            value = features.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"Unsupported timm ViT feature output type: {type(features)}")


class TimmViTPatchMap(nn.Module):
    """Adapter returning patch tokens as a CNN-like feature map [B, D, H, W]."""

    def __init__(self, model: nn.Module, patch_size: int):
        super().__init__()
        self.model = model
        self.patch_size = int(patch_size)
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"ViT inputs must be divisible by patch_size={self.patch_size}, "
                f"got spatial shape {tuple(x.shape[-2:])}"
            )

        tokens = _tokens_from_forward_features(self.model.forward_features(x))
        if tokens.ndim != 3:
            raise ValueError(f"ViT forward_features must return [B, N, D], got {tuple(tokens.shape)}")

        expected_patch_tokens = (x.shape[-2] // self.patch_size) * (x.shape[-1] // self.patch_size)
        if tokens.shape[1] == expected_patch_tokens + self.num_prefix_tokens:
            tokens = tokens[:, self.num_prefix_tokens :]
        elif tokens.shape[1] != expected_patch_tokens:
            raise ValueError(
                f"Unexpected ViT token count {tokens.shape[1]} for input {tuple(x.shape[-2:])}; "
                f"expected {expected_patch_tokens} patch tokens"
            )

        return patch_tokens_to_feature_map(tokens)


def load_timm_vit_backbone(backbone: str, init: str = "random") -> nn.Module:
    if backbone not in VIT_BACKBONES:
        raise ValueError(f"Unknown ViT backbone: {backbone}")
    if init not in {"random", "imagenet"}:
        raise ValueError(
            f"ViT backbone {backbone} supports init='random' or 'imagenet', got {init!r}"
        )

    try:
        import timm
    except ImportError as exc:
        raise ImportError("Training ViT backbones requires timm==1.0.26") from exc

    spec = VIT_BACKBONES[backbone]
    model_name = spec.model_name if init == "random" else spec.imagenet_model_name
    pretrained = init == "imagenet"
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
    except TypeError:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

    return TimmViTPatchMap(model=model, patch_size=spec.patch_size)
