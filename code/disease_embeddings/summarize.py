from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from disease_embeddings.config import StudyConfig
from disease_embeddings.paths import embedding_artifact_path, finetune_dir, finetune_tag, linear_probe_dir
from disease_embeddings.supervised import _split_indices, label_space_from_records, load_manifest_records, load_or_create_split


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict JSON payload at {path}")
    return payload


def _summary_paths(cfg: StudyConfig) -> dict[str, Path]:
    return {
        "headline": cfg.summaries_dir / "linear_probe_summary.csv",
        "per_class": cfg.summaries_dir / "linear_probe_per_class.csv",
        "confusion": cfg.summaries_dir / "linear_probe_confusion_long.csv",
        "knn": cfg.summaries_dir / "embedding_knn5_summary.csv",
    }


def _finetune_summary_paths(cfg: StudyConfig) -> dict[str, Path]:
    tag = finetune_tag(cfg)
    return {
        "headline": cfg.summaries_dir / f"{tag}_summary.csv",
        "per_class": cfg.summaries_dir / f"{tag}_per_class.csv",
        "confusion": cfg.summaries_dir / f"{tag}_confusion_long.csv",
        "knn": cfg.summaries_dir / f"{tag}_knn5_summary.csv",
    }


def _load_embedding_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict embedding artifact at {path}")
    return payload


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return x / norm


def _majority_vote(neighbor_labels: np.ndarray) -> np.ndarray:
    preds = []
    for row in neighbor_labels:
        labels, counts = np.unique(row, return_counts=True)
        preds.append(int(labels[np.argmax(counts)]))
    return np.asarray(preds, dtype=np.int64)


def knn5_metrics_for_model(cfg: StudyConfig, model) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    split_df = load_or_create_split(cfg)
    payload = _load_embedding_payload(embedding_artifact_path(cfg, model))
    metadata = payload["metadata"]
    label_space = label_space_from_records(load_manifest_records(cfg.dataset), cfg.supervised.label_column)
    labels = label_space.encode([str(row[cfg.supervised.label_column]) for row in metadata]).numpy()
    train_idx = _split_indices(metadata, split_df, "train")
    test_idx = _split_indices(metadata, split_df, "test")
    if not train_idx or not test_idx:
        raise ValueError(f"kNN requires non-empty train/test splits for {model.model_id}")

    x = payload["embeddings"].float().numpy()
    x_train = _normalize_rows(x[train_idx].astype(np.float64, copy=False))
    x_test = _normalize_rows(x[test_idx].astype(np.float64, copy=False))
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    k = min(5, len(train_idx))
    similarities = x_test @ x_train.T
    neighbor_idx = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    neighbor_labels = y_train[neighbor_idx]
    y_pred = _majority_vote(neighbor_labels)
    class_labels = list(range(len(label_space.classes)))
    return {
        "model_id": model.model_id,
        "model_label": model.label,
        "run_name": model.run_name,
        "checkpoint_step": model.checkpoint_step,
        "embedding_dim": int(payload["embeddings"].shape[1]),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "knn_k": k,
        "knn_metric": "cosine",
        "knn5_accuracy": float(accuracy_score(y_test, y_pred)),
        "knn5_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "knn5_macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=class_labels, zero_division=0)),
        "knn5_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", labels=class_labels, zero_division=0)),
    }


def _knn5_metrics_from_payload(cfg: StudyConfig, model, payload: dict[str, Any], source: str) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    metadata = payload["metadata"]
    labels = payload["labels"].numpy() if isinstance(payload["labels"], torch.Tensor) else np.asarray(payload["labels"])
    split_by_row = {int(k): v for k, v in payload["split_by_manifest_row_index"].items()}
    train_idx = [idx for idx, row in enumerate(metadata) if split_by_row[int(row["manifest_row_index"])] == "train"]
    test_idx = [idx for idx, row in enumerate(metadata) if split_by_row[int(row["manifest_row_index"])] == "test"]
    if not train_idx or not test_idx:
        raise ValueError(f"Adapted kNN requires non-empty train/test splits for {model.model_id}")

    x = payload["embeddings"].float().numpy()
    x_train = _normalize_rows(x[train_idx].astype(np.float64, copy=False))
    x_test = _normalize_rows(x[test_idx].astype(np.float64, copy=False))
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    k = min(5, len(train_idx))
    similarities = x_test @ x_train.T
    neighbor_idx = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    y_pred = _majority_vote(y_train[neighbor_idx])
    classes = tuple(str(label) for label in payload["classes"])
    class_labels = list(range(len(classes)))
    return {
        "model_id": model.model_id,
        "model_label": model.label,
        "run_name": model.run_name,
        "checkpoint_step": model.checkpoint_step,
        "embedding_source": source,
        "embedding_dim": int(payload["embeddings"].shape[1]),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "knn_k": k,
        "knn_metric": "cosine",
        "knn5_accuracy": float(accuracy_score(y_test, y_pred)),
        "knn5_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "knn5_macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=class_labels, zero_division=0)),
        "knn5_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", labels=class_labels, zero_division=0)),
    }


def adapted_knn5_metrics_for_model(cfg: StudyConfig, model) -> dict[str, Any]:
    path = finetune_dir(cfg, model) / "adapted_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Adapted embeddings missing: {path}")
    return _knn5_metrics_from_payload(cfg, model, _load_embedding_payload(path), source=finetune_tag(cfg))


def summarize_adapted_knn5(cfg: StudyConfig) -> Path:
    rows = [adapted_knn5_metrics_for_model(cfg, model) for model in cfg.models]
    out_path = cfg.summaries_dir / f"{finetune_tag(cfg)}_knn5_summary.csv"
    cfg.summaries_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["knn5_accuracy", "knn5_macro_f1"], ascending=False).to_csv(out_path, index=False)
    return out_path


def _summarize_metrics(
    cfg: StudyConfig,
    metrics_dir_fn,
    paths: dict[str, Path],
    include_adapted_knn: bool,
) -> dict[str, Path]:
    headline_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    knn_rows: list[dict[str, Any]] = []

    for model in cfg.models:
        metrics_path = metrics_dir_fn(cfg, model) / "test_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics missing: {metrics_path}")
        metrics = _load_json(metrics_path)
        knn_metrics = adapted_knn5_metrics_for_model(cfg, model) if include_adapted_knn else knn5_metrics_for_model(cfg, model)
        knn_rows.append(knn_metrics)
        classes = [str(label) for label in metrics["classes"]]
        report = metrics["classification_report"]

        row = {
            "model_id": metrics.get("model_id", model.model_id),
            "model_label": metrics.get("model_label", model.label),
            "run_name": metrics.get("run_name", model.run_name),
            "checkpoint_step": metrics.get("checkpoint_step", model.checkpoint_step),
            "embedding_dim": metrics.get("embedding_dim"),
            "train_rows": metrics.get("train_rows"),
            "test_rows": metrics.get("test_rows"),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "knn5_accuracy": knn_metrics["knn5_accuracy"],
            "knn5_balanced_accuracy": knn_metrics["knn5_balanced_accuracy"],
            "knn5_macro_f1": knn_metrics["knn5_macro_f1"],
            "knn5_weighted_f1": knn_metrics["knn5_weighted_f1"],
        }
        for class_name in classes:
            class_report = report.get(class_name, {})
            row[f"{class_name}__f1"] = class_report.get("f1-score")
            row[f"{class_name}__recall"] = class_report.get("recall")
        headline_rows.append(row)

        for class_name in classes:
            class_report = report.get(class_name, {})
            per_class_rows.append(
                {
                    "model_id": row["model_id"],
                    "model_label": row["model_label"],
                    "class_name": class_name,
                    "precision": class_report.get("precision"),
                    "recall": class_report.get("recall"),
                    "f1": class_report.get("f1-score"),
                    "support": class_report.get("support"),
                }
            )

        matrix = metrics.get("confusion_matrix", [])
        for true_idx, true_label in enumerate(classes):
            for pred_idx, pred_label in enumerate(classes):
                count = matrix[true_idx][pred_idx] if true_idx < len(matrix) and pred_idx < len(matrix[true_idx]) else 0
                confusion_rows.append(
                    {
                        "model_id": row["model_id"],
                        "model_label": row["model_label"],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(count),
                    }
                )

    cfg.summaries_dir.mkdir(parents=True, exist_ok=True)
    headline_df = pd.DataFrame(headline_rows).sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    headline_df.insert(0, "rank_by_macro_f1", range(1, len(headline_df) + 1))
    headline_df.to_csv(paths["headline"], index=False)
    pd.DataFrame(per_class_rows).to_csv(paths["per_class"], index=False)
    pd.DataFrame(confusion_rows).to_csv(paths["confusion"], index=False)
    pd.DataFrame(knn_rows).sort_values(["knn5_accuracy", "knn5_macro_f1"], ascending=False).to_csv(paths["knn"], index=False)
    return paths


def summarize_linear_probe(cfg: StudyConfig) -> dict[str, Path]:
    return _summarize_metrics(cfg, linear_probe_dir, _summary_paths(cfg), include_adapted_knn=False)


def summarize_finetune(cfg: StudyConfig) -> dict[str, Path]:
    return _summarize_metrics(cfg, finetune_dir, _finetune_summary_paths(cfg), include_adapted_knn=True)


def format_headline_table(path: Path) -> str:
    df = pd.read_csv(path)
    cols = [
        "rank_by_macro_f1",
        "model_label",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "knn5_accuracy",
        "test_rows",
    ]
    present = [col for col in cols if col in df.columns]
    return df[present].to_string(index=False)
