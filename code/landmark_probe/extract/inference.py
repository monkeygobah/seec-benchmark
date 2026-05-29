from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from landmark_probe.constants import POOL_G4, REPRESENTATION_PATCH_TOKENS
from landmark_probe.config import RunSpec
from landmark_probe.extract.external_vit import external_model_info, load_external_vit_feature_model
import yaml

POOLING_AREA = {
    "gap": 1,
    "g2": 4,
    "g4": 16,
}


def _strip_module_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _disable_running_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None


def load_training_config_for_run(run: RunSpec) -> dict[str, Any]:
    if run.external_model is not None:
        info = external_model_info(run.external_model)
        return {
            "run": {
                "name": run.run_name,
                "seed": run.baseline_seed,
            },
            "model": {
                "backbone": info.model_arch,
                "init": info.init_mode,
                "feat_dim": info.feat_dim,
                "model_family": info.model_family,
                "model_arch": info.model_arch,
                "patch_size": info.patch_size,
                "pretrain_data": info.pretrain_data,
                "feature_kind": f"{REPRESENTATION_PATCH_TOKENS}_{POOL_G4}",
                "external_model": run.external_model,
            },
            "ssl": {
                "method": "external_baseline",
            },
        }
    if run.baseline_init is not None:
        seg_ckpt = run.baseline_seg_ckpt or Path("/workspace/models/hp_tune.pth")
        return {
            "run": {
                "name": run.run_name,
                "seed": run.baseline_seed,
            },
            "model": {
                "backbone": "resnet101",
                "init": run.baseline_init,
                "seg_ckpt": str(seg_ckpt),
                "feat_dim": 2048,
            },
            "ssl": {
                "method": "baseline",
            },
        }
    if run.run_dir is None:
        raise ValueError(f"Run {run.run_name} has no run_dir")
    with (run.run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expected_embedding_dim(train_cfg: dict[str, Any], pooling: str) -> int:
    if pooling not in POOLING_AREA:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    feat_dim = int(train_cfg.get("model", {}).get("feat_dim", 2048))
    return feat_dim * POOLING_AREA[pooling]


def checkpoint_path_for_run(run: RunSpec) -> Path:
    if run.external_model is not None:
        return Path(f"external://{run.external_model}")
    if run.baseline_init is not None:
        raise ValueError(f"Baseline run {run.run_name} does not use a checkpoint")
    if run.run_dir is None:
        raise ValueError(f"Run {run.run_name} has no run_dir")
    return run.checkpoint_path or (run.run_dir / "checkpoints" / f"ckpt_step_{run.checkpoint_step:07d}.pth")


def load_feature_model_for_run(run: RunSpec) -> tuple[nn.Module, dict[str, Any], Path]:
    train_cfg = load_training_config_for_run(run)
    if run.external_model is not None:
        model = load_external_vit_feature_model(run.external_model, run.checkpoint_path)
        return model, train_cfg, run.checkpoint_path or checkpoint_path_for_run(run)

    if run.baseline_init is not None:
        from src.load_backbones import load_encoder_backbone

        torch.manual_seed(run.baseline_seed)
        seg_ckpt = run.baseline_seg_ckpt or Path("/workspace/models/hp_tune.pth")
        encoder = load_encoder_backbone(init=run.baseline_init, seg_ckpt=str(seg_ckpt))
        _disable_running_stats(encoder)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
        return encoder, train_cfg, Path(f"baseline://{run.baseline_init}")

    checkpoint_path = checkpoint_path_for_run(run)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    init_mode = str(train_cfg["model"]["init"])
    backbone = str(train_cfg.get("model", {}).get("backbone", "resnet101"))
    if backbone != "resnet101":
        from src.load_backbones import load_encoder_backbone

        encoder = load_encoder_backbone(
            backbone=backbone,
            init=init_mode,
            seg_ckpt=train_cfg.get("model", {}).get("seg_ckpt"),
        )
    elif init_mode == "seg_init":
        from torchvision.models.segmentation import deeplabv3_resnet101

        model = deeplabv3_resnet101(weights=None, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(256, 6, kernel_size=1, stride=1)
        encoder = model.backbone
    elif init_mode in {"imagenet", "random"}:
        from torchvision.models import resnet101

        model = resnet101(weights=None)
        encoder = nn.Sequential(*list(model.children())[:-2])
    else:
        raise ValueError(f"Unsupported init mode for landmark probe extraction: {init_mode}")
    _disable_running_stats(encoder)
    encoder.load_state_dict(_strip_module_prefix(ckpt["encoder"]), strict=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder, train_cfg, checkpoint_path


@torch.inference_mode()
def pooled_feature_map_embeddings(model: nn.Module, x: torch.Tensor, pooling: str) -> torch.Tensor:
    feat = model(x)
    if isinstance(feat, dict):
        feat = feat["out"]
    if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
        raise ValueError(f"Feature extractor must return a [B, C, H, W] tensor, got {type(feat)}")
    if pooling == "gap":
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
    elif pooling == "g2":
        pooled = F.adaptive_avg_pool2d(feat, (2, 2))
    elif pooling == "g4":
        pooled = F.adaptive_avg_pool2d(feat, (4, 4))
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    return pooled.flatten(1)


def load_backbone_for_run(run: RunSpec) -> tuple[nn.Module, dict[str, Any], Path]:
    return load_feature_model_for_run(run)


def pooled_backbone_embeddings(encoder: nn.Module, x: torch.Tensor, pooling: str) -> torch.Tensor:
    return pooled_feature_map_embeddings(encoder, x, pooling)
