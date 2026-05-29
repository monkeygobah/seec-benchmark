from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

from embedding_extract.datasets import build_dataloader
from embedding_extract.inference import build_inference_bundle, create_embedding_model
from embedding_extract.paths import embedding_artifact_path
from embedding_extract.pipeline_config import DatasetSpec, StudyConfig
from embedding_extract.runtime import load_training_config_for_run, resolve_checkpoint_path


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _meta_record(meta: Any, dataset: DatasetSpec) -> dict[str, Any]:
    return {
        "filename": meta.filename,
        "rel_path": str(meta.rel_path),
        "path": str(meta.path),
        "dataset_name": dataset.dataset_name,
        "split_label": dataset.split_label,
    }


def extract_study(cfg: StudyConfig, overwrite: bool = False) -> list[Path]:
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.extraction.device)
    autocast_dtype = _autocast_dtype(cfg.extraction.precision)
    amp_enabled = device.type == "cuda" and autocast_dtype is not None

    written: list[Path] = []

    for run in cfg.runs:
        train_cfg = load_training_config_for_run(run)
        checkpoint_path = resolve_checkpoint_path(run)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        bundle = build_inference_bundle(train_cfg, checkpoint_path)
        model = create_embedding_model(bundle, device)

        for dataset in cfg.datasets:
            out_path = embedding_artifact_path(cfg, run.run_name, run.checkpoint_step, dataset)
            allow_overwrite = overwrite or cfg.extraction.overwrite
            if out_path.exists() and not allow_overwrite:
                raise FileExistsError(f"Embedding artifact already exists: {out_path}")

            dl = build_dataloader(dataset, cfg.extraction)
            emb_batches: list[torch.Tensor] = []
            proj_batches: list[torch.Tensor] = []
            metas: list[dict[str, Any]] = []

            desc = f"extract {run.run_name} [{dataset.dataset_name}:{dataset.split_label}]"
            iterator = tqdm(dl, desc=desc)
            for imgs, batch_meta in iterator:
                imgs = imgs.to(device, non_blocking=True)
                with autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=amp_enabled,
                ):
                    outputs = model(imgs)

                if cfg.artifact.save_backbone:
                    emb_batches.append(outputs["emb"].detach().cpu().to(torch.float32))
                if cfg.artifact.save_projected:
                    proj_batches.append(outputs["proj"].detach().cpu().to(torch.float32))
                metas.extend(_meta_record(meta, dataset) for meta in batch_meta)

            payload: dict[str, Any] = {
                "meta": metas,
                "run": {
                    "run_name": run.run_name,
                    "run_dir": str(run.run_dir),
                },
                "checkpoint": {
                    "checkpoint_step": run.checkpoint_step,
                    "checkpoint_path": str(checkpoint_path),
                },
                "dataset": {
                    "dataset_name": dataset.dataset_name,
                    "split_label": dataset.split_label,
                    "root": str(dataset.root),
                },
                "config_snapshot": {
                    "training_cfg": train_cfg,
                    "extraction": {
                        "batch_size": cfg.extraction.batch_size,
                        "num_workers": cfg.extraction.num_workers,
                        "device": str(device),
                        "precision": cfg.extraction.precision,
                    },
                    "artifact": {
                        "save_backbone": cfg.artifact.save_backbone,
                        "save_projected": cfg.artifact.save_projected,
                    },
                },
                "num_rows": len(metas),
                "meta_count": len(metas),
            }
            if cfg.artifact.save_backbone:
                payload["emb"] = torch.cat(emb_batches, dim=0)
            if cfg.artifact.save_projected:
                payload["proj"] = torch.cat(proj_batches, dim=0)

            if "emb" in payload and payload["emb"].shape[0] != len(metas):
                raise ValueError(f"emb rows do not match metadata count for {out_path}")
            if "proj" in payload and payload["proj"].shape[0] != len(metas):
                raise ValueError(f"proj rows do not match metadata count for {out_path}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out_path)
            written.append(out_path)

    return written
