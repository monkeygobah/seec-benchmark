from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

MAE_VITB16_PRETRAIN_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"


@dataclass(frozen=True)
class ExternalModelInfo:
    model_family: str
    model_arch: str
    patch_size: int
    pretrain_data: str
    feat_dim: int

    @property
    def init_mode(self) -> str:
        return self.model_family


EXTERNAL_MODEL_INFO: dict[str, ExternalModelInfo] = {
    "dinov2_vitb14": ExternalModelInfo(
        model_family="dinov2",
        model_arch="vitb14",
        patch_size=14,
        pretrain_data="LVD-142M",
        feat_dim=768,
    ),
    "mae_vitb16_in1k_pretrain": ExternalModelInfo(
        model_family="mae",
        model_arch="vitb16",
        patch_size=16,
        pretrain_data="ImageNet-1K",
        feat_dim=768,
    ),
}


def external_model_info(model_name: str) -> ExternalModelInfo:
    try:
        return EXTERNAL_MODEL_INFO[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported external model: {model_name}") from exc


def _freeze_eval(model: nn.Module) -> nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def patch_tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Patch tokens must have shape [B, N, C], got {tuple(tokens.shape)}")
    batch_size, num_tokens, dim = tokens.shape
    grid_size = int(num_tokens**0.5)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Patch token count must be a square grid, got {num_tokens}")
    return tokens.transpose(1, 2).reshape(batch_size, dim, grid_size, grid_size).contiguous()


class DINOv2PatchFeatureMap(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(x)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise ValueError("DINOv2 model did not return x_norm_patchtokens from forward_features")
        return patch_tokens_to_feature_map(features["x_norm_patchtokens"])


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height != self.img_size or width != self.img_size:
            raise ValueError(f"MAE ViT-B/16 expects {self.img_size}x{self.img_size} inputs, got {height}x{width}")
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, 768), requires_grad=False)
        self.blocks = nn.ModuleList(Block() for _ in range(12))
        self.norm = nn.LayerNorm(768, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return patch_tokens_to_feature_map(x[:, 1:])


def _load_checkpoint(path_or_url: Path | None) -> dict[str, Any]:
    if path_or_url is None:
        return torch.hub.load_state_dict_from_url(MAE_VITB16_PRETRAIN_URL, map_location="cpu", check_hash=False)
    return torch.load(path_or_url, map_location="cpu")


def load_mae_vitb16_encoder(checkpoint_path: Path | None) -> nn.Module:
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("MAE checkpoint must contain a state dict or a 'model' state dict")
    encoder = MAEViTEncoder()
    ignored_prefixes = ("decoder_", "mask_token")
    encoder_state = {k: v for k, v in state_dict.items() if not k.startswith(ignored_prefixes)}
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(ignored_prefixes)]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "MAE checkpoint is not compatible with the ViT-B/16 encoder: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    return _freeze_eval(encoder)


def load_external_vit_feature_model(model_name: str, checkpoint_path: Path | None = None) -> nn.Module:
    external_model_info(model_name)
    if model_name == "dinov2_vitb14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict, strict=True)
        return _freeze_eval(DINOv2PatchFeatureMap(model))
    if model_name == "mae_vitb16_in1k_pretrain":
        return load_mae_vitb16_encoder(checkpoint_path)
    raise ValueError(f"Unsupported external model: {model_name}")
