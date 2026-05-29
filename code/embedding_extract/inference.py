from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.load_backbones import load_encoder_backbone
from src.projectors import MLPProjector, ProjectorCfg, gap_pool


def _strip_module_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _get_feat_out(y: Any) -> torch.Tensor:
    return y["out"] if isinstance(y, Mapping) else y


def _disable_running_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None


def _extract_projector_state(obj_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("projector.", "proj."):
        proj_sd = {k[len(prefix):]: v for k, v in obj_sd.items() if k.startswith(prefix)}
        if proj_sd:
            return _strip_module_prefix(proj_sd)
    raise KeyError(
        "No projector weights found in ckpt['objective']; expected keys starting with 'projector.'"
    )


@dataclass(frozen=True)
class InferenceBundle:
    encoder: nn.Module
    projector: nn.Module
    method: str
    train_cfg: dict[str, Any]


class EmbeddingModel(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module, method: str):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.method = method

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = _get_feat_out(self.encoder(x))
        emb = gap_pool(feat)
        proj = self.projector(emb)
        # if self.method == "infonce":
        #     proj = F.normalize(proj, dim=1, eps=1e-8)
        return {"emb": emb, "proj": proj}


def build_inference_bundle(train_cfg: dict[str, Any], checkpoint_path: str | Path) -> InferenceBundle:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    encoder = load_encoder_backbone(
        init=train_cfg["model"]["init"],
        seg_ckpt=train_cfg["model"].get("seg_ckpt"),
    )
    # Match the training-time encoder state structure: backbone BatchNorm
    # running stats were disabled before checkpoints were saved.
    _disable_running_stats(encoder)
    encoder.load_state_dict(_strip_module_prefix(ckpt["encoder"]), strict=True)

    proj_cfg = ProjectorCfg(
        in_dim=int(train_cfg["model"].get("feat_dim", 2048)),
        proj_dim=int(train_cfg["model"]["proj_dim"]),
        hidden_dim=int(train_cfg["model"]["proj_hidden"]),
        layers=int(train_cfg["model"]["proj_layers"]),
    )
    projector = MLPProjector(proj_cfg)
    projector.load_state_dict(_extract_projector_state(ckpt["objective"]), strict=True)

    encoder.eval()
    projector.eval()
    for module in (encoder, projector):
        for param in module.parameters():
            param.requires_grad_(False)

    return InferenceBundle(
        encoder=encoder,
        projector=projector,
        method=str(train_cfg["ssl"]["method"]),
        train_cfg=train_cfg,
    )


def create_embedding_model(bundle: InferenceBundle, device: torch.device) -> EmbeddingModel:
    model = EmbeddingModel(bundle.encoder, bundle.projector, bundle.method).to(device)
    model.eval()
    return model
