from __future__ import annotations

from pathlib import Path

from embedding_extract.pipeline_config import DatasetSpec, StudyConfig
from embedding_extract.runtime import resolve_checkpoint_path


def artifact_stem(run_name: str, checkpoint_step: int, dataset: DatasetSpec) -> str:
    return (
        f"{run_name}__step_{checkpoint_step:07d}"
        f"__{dataset.dataset_name}__{dataset.split_label}"
    )


def embedding_artifact_path(cfg: StudyConfig, run_name: str, checkpoint_step: int, dataset: DatasetSpec) -> Path:
    return cfg.embeddings_dir / f"{artifact_stem(run_name, checkpoint_step, dataset)}.pt"


def isotropy_metric_path(cfg: StudyConfig, run_name: str, checkpoint_step: int, dataset: DatasetSpec, embedding_key: str) -> Path:
    return cfg.metrics_dir / "isotropy" / f"{artifact_stem(run_name, checkpoint_step, dataset)}__{embedding_key}.json"
