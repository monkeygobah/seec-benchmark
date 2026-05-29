from __future__ import annotations

from pathlib import Path

from disease_embeddings.config import ModelSpec, StudyConfig


def embedding_artifact_path(cfg: StudyConfig, model: ModelSpec) -> Path:
    return cfg.embeddings_dir / f"{model.model_id}__{cfg.pooling}.pt"


def reduction_csv_path(cfg: StudyConfig, model: ModelSpec, method: str) -> Path:
    return cfg.reductions_dir / f"{model.model_id}__{cfg.pooling}__{method}.csv"


def figure_path(cfg: StudyConfig, model: ModelSpec, method: str) -> Path:
    return cfg.figures_dir / f"{model.model_id}__{cfg.pooling}__{method}.png"


def split_csv_path(cfg: StudyConfig) -> Path:
    return cfg.splits_dir / f"{cfg.name}__seed_{cfg.supervised.seed}.csv"


def linear_probe_dir(cfg: StudyConfig, model: ModelSpec) -> Path:
    return cfg.linear_probe_dir / model.model_id


def finetune_dir(cfg: StudyConfig, model: ModelSpec) -> Path:
    return cfg.finetune_dir / model.model_id


def finetune_tag(cfg: StudyConfig) -> str:
    return f"finetune_{cfg.supervised.finetune_epochs}epochs"


def adapted_reduction_csv_path(cfg: StudyConfig, model: ModelSpec, method: str, split: str | None = None) -> Path:
    split_part = f"__{split}" if split else ""
    return cfg.reductions_dir / f"{model.model_id}__{cfg.pooling}__{finetune_tag(cfg)}{split_part}__{method}.csv"


def adapted_figure_path(cfg: StudyConfig, model: ModelSpec, method: str, split: str | None = None) -> Path:
    split_part = f"__{split}" if split else ""
    return cfg.figures_dir / f"{model.model_id}__{cfg.pooling}__{finetune_tag(cfg)}{split_part}__{method}.png"
