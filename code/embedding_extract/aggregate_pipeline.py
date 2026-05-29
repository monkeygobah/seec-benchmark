from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from embedding_extract.pipeline_config import StudyConfig


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_study(cfg: StudyConfig) -> Path:
    metrics_dir = cfg.metrics_dir / "isotropy"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory does not exist: {metrics_dir}")

    rows = [_load_json(path) for path in sorted(metrics_dir.glob("*.json"))]
    if not rows:
        raise FileNotFoundError(f"No isotropy metric files found in {metrics_dir}")

    default_columns = list(cfg.aggregation.row_keys)
    for extra in [
        "N",
        "D",
        "mean_norm",
        "erank",
        "erank_over_d",
        "ev1",
        "ev5",
        "ev10",
        "ev20",
        "cond_1_med",
        "cos_mean",
        "cos_std",
        "cos_std_expected_sphere",
        "cos_frac_abs_gt_0.2",
        "cos_frac_abs_gt_0.3",
        "cos_frac_abs_gt_0.4",
        "num_pairs_used",
        "artifact_path",
        "checkpoint_path",
    ]:
        if extra not in default_columns:
            default_columns.append(extra)

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.tables_dir / cfg.aggregation.summary_csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=default_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in default_columns})

    return out_path
