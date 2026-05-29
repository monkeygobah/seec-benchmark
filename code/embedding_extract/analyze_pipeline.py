from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from embedding_extract.isotropy_analysis import compute_isotropy_metrics
from embedding_extract.paths import embedding_artifact_path, isotropy_metric_path
from embedding_extract.pipeline_config import StudyConfig


def _load_artifact(path: Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected dict artifact at {path}, got {type(artifact)}")
    return artifact


def analyze_study(cfg: StudyConfig, overwrite: bool = False) -> list[Path]:
    if not cfg.isotropy.enabled:
        return []

    written: list[Path] = []
    metrics_root = cfg.metrics_dir / "isotropy"
    metrics_root.mkdir(parents=True, exist_ok=True)

    for run in cfg.runs:
        for dataset in cfg.datasets:
            artifact_path = embedding_artifact_path(cfg, run.run_name, run.checkpoint_step, dataset)
            if not artifact_path.exists():
                raise FileNotFoundError(f"Embedding artifact missing: {artifact_path}")

            artifact = _load_artifact(artifact_path)
            for embedding_key in cfg.isotropy.embedding_keys:
                if embedding_key not in artifact:
                    continue

                out_path = isotropy_metric_path(cfg, run.run_name, run.checkpoint_step, dataset, embedding_key)
                allow_overwrite = overwrite or cfg.extraction.overwrite
                if out_path.exists() and not allow_overwrite:
                    raise FileExistsError(f"Metric output already exists: {out_path}")

                metrics = compute_isotropy_metrics(
                    artifact[embedding_key],
                    num_pairs=cfg.isotropy.num_pairs,
                    seed=cfg.isotropy.seed,
                )
                metrics.update(
                    {
                        "analysis": "isotropy",
                        "embedding_key": embedding_key,
                        "run_name": artifact["run"]["run_name"],
                        "run_dir": artifact["run"]["run_dir"],
                        "checkpoint_step": artifact["checkpoint"]["checkpoint_step"],
                        "checkpoint_path": artifact["checkpoint"]["checkpoint_path"],
                        "dataset_name": artifact["dataset"]["dataset_name"],
                        "split_label": artifact["dataset"]["split_label"],
                        "dataset_root": artifact["dataset"]["root"],
                        "artifact_path": str(artifact_path),
                        "num_rows": int(artifact["num_rows"]),
                    }
                )

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                written.append(out_path)

    return written
