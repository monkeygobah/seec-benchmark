from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable, **_: object):
        return iterable

try:
    from torch import autocast
except ImportError:  # pragma: no cover - older torch
    from torch.cuda.amp import autocast  # type: ignore

from landmark_probe.constants import REPRESENTATION_BACKBONE, REPRESENTATION_PATCH_TOKENS
from landmark_probe.config import DatasetSpec, RepresentationSpec, RunSpec, StudyConfig, TaskSplitSpec
from landmark_probe.extract.datasets import build_dataloader, load_split_records
from landmark_probe.extract.inference import (
    expected_embedding_dim,
    load_feature_model_for_run,
    load_training_config_for_run,
    pooled_feature_map_embeddings,
)
from landmark_probe.paths import embedding_artifact_path


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


def _validate_embedding_payload(
    payload: dict[str, Any],
    out_path: Path,
    run: RunSpec,
    split_spec: TaskSplitSpec,
    representation: RepresentationSpec,
    expected_dim: int,
    expected_rows: int,
) -> None:
    embeddings = payload.get("embeddings")
    sample_ids = payload.get("sample_ids")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"Embedding artifact has invalid embeddings tensor: {out_path}")
    if not isinstance(sample_ids, list):
        raise ValueError(f"Embedding artifact has invalid sample_ids: {out_path}")
    if embeddings.shape[0] != len(sample_ids) or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding artifact row count mismatch at {out_path}: "
            f"tensor_rows={embeddings.shape[0]}, sample_ids={len(sample_ids)}, expected_rows={expected_rows}"
        )
    if int(payload.get("embedding_dim", embeddings.shape[1])) != embeddings.shape[1]:
        raise ValueError(f"Embedding artifact metadata dim does not match tensor shape: {out_path}")
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding artifact dim mismatch at {out_path}: got {embeddings.shape[1]}, expected {expected_dim}. "
            "Re-run extraction with overwrite enabled."
        )
    expected_fields = {
        "run_name": run.run_name,
        "checkpoint_step": run.checkpoint_step,
        "dataset_name": split_spec.dataset_name,
        "split": split_spec.split,
        "embedding_key": representation.embedding_key,
        "pooling": representation.pooling,
    }
    for key, expected_value in expected_fields.items():
        if payload.get(key) != expected_value:
            raise ValueError(
                f"Embedding artifact metadata mismatch at {out_path}: "
                f"{key}={payload.get(key)!r}, expected {expected_value!r}"
            )


def extract_split_embeddings(
    study_cfg: StudyConfig,
    dataset_cfg: DatasetSpec,
    run: RunSpec,
    split_spec: TaskSplitSpec,
    representation: RepresentationSpec,
) -> Path:
    out_path = embedding_artifact_path(study_cfg, run, split_spec, representation)
    allow_overwrite = study_cfg.extraction.overwrite

    expected_key = REPRESENTATION_PATCH_TOKENS if run.external_model is not None else REPRESENTATION_BACKBONE
    if representation.embedding_key != expected_key:
        raise ValueError(
            f"Run {run.run_name} requires {expected_key} embeddings, got {representation.embedding_key}"
        )

    records = load_split_records(dataset_cfg, split_spec)
    train_cfg_for_validation = load_training_config_for_run(run)
    expected_dim = expected_embedding_dim(train_cfg_for_validation, representation.pooling)
    if out_path.exists() and not allow_overwrite:
        payload = torch.load(out_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload at {out_path}")
        _validate_embedding_payload(
            payload,
            out_path,
            run,
            split_spec,
            representation,
            expected_dim=expected_dim,
            expected_rows=len(records),
        )
        return out_path

    dl = build_dataloader(dataset_cfg, split_spec, study_cfg.extraction)
    device = _resolve_device(study_cfg.extraction.device)
    autocast_dtype = _autocast_dtype(study_cfg.extraction.precision)
    amp_enabled = device.type == "cuda" and autocast_dtype is not None

    model, train_cfg, checkpoint_path = load_feature_model_for_run(run)
    model = model.to(device)
    all_embs: list[torch.Tensor] = []
    sample_ids: list[str] = []
    for xs, metas in tqdm(dl, desc=f"extract {run.run_name} {split_spec.dataset_name}:{split_spec.split}:{representation.pooling}"):
        xs = xs.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            emb = pooled_feature_map_embeddings(model, xs, representation.pooling)
        if emb.shape[1] != expected_dim:
            raise ValueError(
                f"Extracted embedding dim mismatch for {run.run_name} {representation.pooling}: "
                f"got {emb.shape[1]}, expected {expected_dim}"
            )
        all_embs.append(emb.detach().cpu().to(torch.float32))
        sample_ids.extend(meta.sample_id for meta in metas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "embeddings": torch.cat(all_embs, dim=0),
        "sample_ids": sample_ids,
        "dataset_name": split_spec.dataset_name,
        "split": split_spec.split,
        "embedding_key": representation.embedding_key,
        "pooling": representation.pooling,
        "embedding_dim": int(all_embs[0].shape[1]) if all_embs else 0,
        "run_name": run.run_name,
        "run_dir": str(run.run_dir),
        "checkpoint_step": run.checkpoint_step,
        "checkpoint_path": str(checkpoint_path),
        "train_cfg": train_cfg,
        "ssl_method": str(train_cfg.get("ssl", {}).get("method", "unknown")),
        "init_mode": str(train_cfg.get("model", {}).get("init", "unknown")),
        "model_family": train_cfg.get("model", {}).get("model_family"),
        "model_arch": train_cfg.get("model", {}).get("model_arch"),
        "patch_size": train_cfg.get("model", {}).get("patch_size"),
        "pretrain_data": train_cfg.get("model", {}).get("pretrain_data"),
        "feature_kind": train_cfg.get("model", {}).get("feature_kind"),
        "num_rows": len(sample_ids),
        "manifest_count": len(records),
    }
    if payload["num_rows"] != payload["manifest_count"]:
        raise ValueError(f"Embedding row count mismatch for {out_path}")
    torch.save(payload, out_path)
    return out_path


def extract_study(study_cfg: StudyConfig, dataset_cfg: DatasetSpec) -> list[Path]:
    written: list[Path] = []
    unique_splits: dict[tuple[str, str], TaskSplitSpec] = {}
    for task in study_cfg.tasks:
        for split in (task.train_split, task.val_split, task.test_split):
            unique_splits[(split.dataset_name, split.split)] = split

    for run in study_cfg.runs:
        for representation in study_cfg.representations:
            for split_spec in unique_splits.values():
                written.append(extract_split_embeddings(study_cfg, dataset_cfg, run, split_spec, representation))
    return written
