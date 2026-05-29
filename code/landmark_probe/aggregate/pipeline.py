from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from landmark_probe.config import StudyConfig


def aggregate_study(study_cfg: StudyConfig) -> tuple[Path, Path]:
    overall_rows = []
    per_landmark_rows = []
    for metrics_path in sorted(study_cfg.probe_runs_dir.glob("**/test_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            overall_rows.append(json.load(f))
        per_landmark_path = metrics_path.parent / "per_landmark.csv"
        if per_landmark_path.exists():
            per_landmark_rows.append(pd.read_csv(per_landmark_path))

    if not overall_rows:
        raise FileNotFoundError(f"No probe result files found in {study_cfg.probe_runs_dir}")

    study_cfg.summaries_dir.mkdir(parents=True, exist_ok=True)
    overall_out = study_cfg.summaries_dir / "overall_summary.csv"
    per_landmark_out = study_cfg.summaries_dir / "per_landmark_summary.csv"
    pd.DataFrame(overall_rows).sort_values(["task_name", "run_name", "pooling"]).to_csv(overall_out, index=False)
    pd.concat(per_landmark_rows, ignore_index=True).sort_values(["task_name", "run_name", "pooling", "landmark"]).to_csv(
        per_landmark_out, index=False
    )

    with (study_cfg.summaries_dir / "study_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"study_cfg": str(study_cfg.path)}, f, sort_keys=False)
    return overall_out, per_landmark_out
