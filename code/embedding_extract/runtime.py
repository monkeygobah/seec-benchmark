from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from embedding_extract.pipeline_config import RunSpec, merge_training_config


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_checkpoint_path(run: RunSpec) -> Path:
    if run.checkpoint_path is not None:
        return run.checkpoint_path
    return run.run_dir / "checkpoints" / f"ckpt_step_{run.checkpoint_step:07d}.pth"


def load_training_config_for_run(run: RunSpec) -> dict[str, Any]:
    cfg = load_yaml(run.run_dir / "config.yaml")
    if run.config_overrides:
        cfg = merge_training_config(cfg, run.config_overrides)
    return cfg
