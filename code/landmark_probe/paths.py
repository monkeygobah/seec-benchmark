from __future__ import annotations

from pathlib import Path

from landmark_probe.config import DatasetSpec, RepresentationSpec, RunSpec, StudyConfig, TaskSplitSpec


def embedding_artifact_path(
    cfg: StudyConfig,
    run: RunSpec,
    split: TaskSplitSpec,
    representation: RepresentationSpec,
) -> Path:
    stem = (
        f"{run.run_name}__step_{run.checkpoint_step:07d}"
        f"__{split.dataset_name}__{split.split}"
        f"__{representation.embedding_key}__{representation.pooling}"
    )
    return cfg.embeddings_dir / f"{stem}.pt"


def probe_run_dir(
    cfg: StudyConfig,
    task_name: str,
    run: RunSpec,
    representation: RepresentationSpec,
) -> Path:
    return cfg.probe_runs_dir / task_name / run.run_name / f"{representation.embedding_key}__{representation.pooling}"
