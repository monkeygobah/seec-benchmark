from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

try:
    from torch import autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast  # type: ignore

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable

from disease_embeddings.config import ModelSpec, StudyConfig
from disease_embeddings.datasets import build_dataloader
from disease_embeddings.paths import embedding_artifact_path
from landmark_probe.config import RunSpec
from landmark_probe.extract.inference import (
    expected_embedding_dim,
    load_feature_model_for_run,
    pooled_feature_map_embeddings,
)


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_spec)


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _to_landmark_run(model: ModelSpec) -> RunSpec:
    if model.source == "external":
        return RunSpec(
            run_name=model.run_name,
            run_dir=None,
            checkpoint_step=model.checkpoint_step,
            checkpoint_path=model.checkpoint_path,
            external_model=model.external_model,
        )
    return RunSpec(
        run_name=model.run_name,
        run_dir=model.run_dir,
        checkpoint_step=model.checkpoint_step,
        checkpoint_path=model.checkpoint_path,
    )


def extract_model_embeddings(cfg: StudyConfig, model_spec: ModelSpec, overwrite: bool = False) -> Path:
    out_path = embedding_artifact_path(cfg, model_spec)
    allow_overwrite = overwrite or cfg.extraction.overwrite
    if out_path.exists() and not allow_overwrite:
        return out_path

    dl = build_dataloader(cfg.dataset, cfg.extraction)
    run = _to_landmark_run(model_spec)
    model, train_cfg, checkpoint_path = load_feature_model_for_run(run)
    expected_dim = expected_embedding_dim(train_cfg, cfg.pooling)

    device = _resolve_device(cfg.extraction.device)
    model = model.to(device)
    model.eval()
    autocast_dtype = _autocast_dtype(cfg.extraction.precision)
    amp_enabled = device.type == "cuda" and autocast_dtype is not None

    emb_batches: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    desc = f"extract {model_spec.label}"
    for xs, records in tqdm(dl, desc=desc):
        xs = xs.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            emb = pooled_feature_map_embeddings(model, xs, cfg.pooling)
        if emb.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch for {model_spec.model_id}: got {emb.shape[1]}, expected {expected_dim}"
            )
        emb_batches.append(emb.detach().cpu().to(torch.float32))
        for record in records:
            row = dict(record.metadata)
            row["manifest_row_index"] = int(record.row_index)
            metadata.append(row)

    embeddings = torch.cat(emb_batches, dim=0)
    if embeddings.shape[0] != len(metadata):
        raise ValueError(f"Embedding row count mismatch for {model_spec.model_id}")

    payload: dict[str, Any] = {
        "embeddings": embeddings,
        "metadata": metadata,
        "model": {
            "model_id": model_spec.model_id,
            "label": model_spec.label,
            "source": model_spec.source,
            "external_model": model_spec.external_model,
            "run_name": model_spec.run_name,
            "run_dir": str(model_spec.run_dir) if model_spec.run_dir is not None else None,
        },
        "checkpoint": {
            "checkpoint_step": model_spec.checkpoint_step,
            "checkpoint_path": str(checkpoint_path),
        },
        "dataset": {
            "root": str(cfg.dataset.root),
            "manifest_csv": str(cfg.dataset.manifest_csv),
            "image_size": cfg.dataset.image_size,
            "normalize_imagenet": cfg.dataset.normalize_imagenet,
        },
        "embedding_key": cfg.embedding_key,
        "pooling": cfg.pooling,
        "embedding_dim": int(embeddings.shape[1]),
        "num_rows": int(embeddings.shape[0]),
        "train_cfg": train_cfg,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def extract_study(cfg: StudyConfig, overwrite: bool = False) -> list[Path]:
    cfg = replace(cfg, extraction=replace(cfg.extraction, overwrite=cfg.extraction.overwrite or overwrite))
    return [extract_model_embeddings(cfg, model, overwrite=overwrite) for model in cfg.models]

