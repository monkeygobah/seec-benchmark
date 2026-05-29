from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from torch import autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast  # type: ignore

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable

from disease_embeddings.config import ModelSpec, StudyConfig
from disease_embeddings.datasets import DiseaseImageDataset, DiseaseImageRecord, load_manifest_records
from disease_embeddings.extract import _autocast_dtype, _resolve_device, _to_landmark_run
from disease_embeddings.paths import (
    adapted_figure_path,
    adapted_reduction_csv_path,
    embedding_artifact_path,
    finetune_dir,
    linear_probe_dir,
    split_csv_path,
)
from disease_embeddings.reduce_plot import plot_coordinates, reduce_embeddings
from landmark_probe.extract.inference import expected_embedding_dim, load_feature_model_for_run, pooled_feature_map_embeddings


@dataclass(frozen=True)
class LabelSpace:
    classes: tuple[str, ...]
    label_to_idx: dict[str, int]

    def encode(self, labels: list[str]) -> torch.Tensor:
        return torch.tensor([self.label_to_idx[label] for label in labels], dtype=torch.long)


class FineTuneClassifier(nn.Module):
    def __init__(self, feature_model: nn.Module, embedding_dim: int, num_classes: int, pooling: str = "g4"):
        super().__init__()
        self.feature_model = feature_model
        self.pooling = pooling
        self.head = nn.Linear(embedding_dim, num_classes)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_model(x)
        if isinstance(feat, dict):
            feat = feat["out"]
        if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
            raise ValueError(f"Feature extractor must return a [B, C, H, W] tensor, got {type(feat)}")
        if self.pooling != "g4":
            raise ValueError(f"FineTuneClassifier currently supports pooling='g4', got {self.pooling!r}")
        return F.adaptive_avg_pool2d(feat, (4, 4)).flatten(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embeddings(x)
        return self.head(emb), emb


def label_space_from_records(records: list[DiseaseImageRecord], label_column: str) -> LabelSpace:
    classes = tuple(sorted({str(record.metadata[label_column]) for record in records}))
    return LabelSpace(classes=classes, label_to_idx={label: idx for idx, label in enumerate(classes)})


def _group_table(records: list[DiseaseImageRecord], label_column: str, group_column: str) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "manifest_row_index": record.row_index,
                "label": str(record.metadata[label_column]),
                "group": str(record.metadata[group_column]),
            }
        )
    df = pd.DataFrame(rows)
    group_rows = []
    for group, group_df in df.groupby("group", sort=False):
        labels = group_df["label"].value_counts()
        group_rows.append(
            {
                "group": group,
                "label": str(labels.index[0]),
                "num_rows": int(len(group_df)),
            }
        )
    return pd.DataFrame(group_rows)


def build_grouped_split(records: list[DiseaseImageRecord], cfg: StudyConfig) -> pd.DataFrame:
    label_column = cfg.supervised.label_column
    group_column = cfg.supervised.group_column
    group_df = _group_table(records, label_column, group_column)
    rng = np.random.default_rng(cfg.supervised.seed)
    test_frac = 1.0 - cfg.supervised.train_frac
    test_groups: set[str] = set()

    for label, label_groups in group_df.groupby("label"):
        groups = label_groups["group"].to_numpy()
        shuffled = groups[rng.permutation(len(groups))]
        if len(shuffled) == 1:
            n_test = 0
        else:
            n_test = max(1, int(round(len(shuffled) * test_frac)))
            n_test = min(n_test, len(shuffled) - 1)
        test_groups.update(str(group) for group in shuffled[:n_test])

    rows = []
    for record in records:
        group = str(record.metadata[group_column])
        split = "test" if group in test_groups else "train"
        rows.append(
            {
                "manifest_row_index": record.row_index,
                "split": split,
                "label": str(record.metadata[label_column]),
                "group": group,
                **record.metadata,
            }
        )
    return pd.DataFrame(rows)


def load_or_create_split(cfg: StudyConfig) -> pd.DataFrame:
    path = split_csv_path(cfg)
    if path.exists():
        return pd.read_csv(path)
    records = load_manifest_records(cfg.dataset)
    split_df = build_grouped_split(records, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)
    return split_df


def _class_weights(y_train: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y_train, minlength=num_classes).float()
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return weights


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "classes": list(classes),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_embedding_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict embedding artifact at {path}")
    return payload


def _split_indices(metadata: list[dict[str, Any]], split_df: pd.DataFrame, split: str) -> list[int]:
    split_rows = set(split_df.loc[split_df["split"] == split, "manifest_row_index"].astype(int))
    return [idx for idx, row in enumerate(metadata) if int(row["manifest_row_index"]) in split_rows]


def run_linear_probe_for_model(cfg: StudyConfig, model: ModelSpec) -> Path:
    split_df = load_or_create_split(cfg)
    payload = _load_embedding_payload(embedding_artifact_path(cfg, model))
    metadata = payload["metadata"]
    label_space = label_space_from_records(load_manifest_records(cfg.dataset), cfg.supervised.label_column)

    embeddings = payload["embeddings"].float()
    labels = label_space.encode([str(row[cfg.supervised.label_column]) for row in metadata])
    train_idx = _split_indices(metadata, split_df, "train")
    test_idx = _split_indices(metadata, split_df, "test")
    if not train_idx or not test_idx:
        raise ValueError("Linear probe requires non-empty train and test splits")

    train_ds = TensorDataset(embeddings[train_idx], labels[train_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.supervised.linear_batch_size, shuffle=True)
    device = _resolve_device(cfg.extraction.device)
    head = nn.Linear(embeddings.shape[1], len(label_space.classes)).to(device)
    weights = _class_weights(labels[train_idx], len(label_space.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.supervised.linear_lr, weight_decay=cfg.supervised.weight_decay)

    history = []
    for epoch in range(1, cfg.supervised.linear_epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.shape[0]
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(xb.shape[0])
        history.append({"epoch": epoch, "train_loss": total_loss / max(total, 1), "train_accuracy": total_correct / max(total, 1)})

    head.eval()
    with torch.no_grad():
        logits = head(embeddings[test_idx].to(device)).cpu()
    y_true = labels[test_idx].numpy()
    y_pred = logits.argmax(dim=1).numpy()
    metrics = _metrics(y_true, y_pred, label_space.classes)
    metrics.update(
        {
            "model_id": model.model_id,
            "model_label": model.label,
            "run_name": model.run_name,
            "checkpoint_step": model.checkpoint_step,
            "embedding_dim": int(embeddings.shape[1]),
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
        }
    )

    out_dir = linear_probe_dir(cfg, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": head.state_dict(),
            "in_dim": int(embeddings.shape[1]),
            "classes": label_space.classes,
            "model_id": model.model_id,
            "run_name": model.run_name,
        },
        out_dir / "linear_head.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    _write_json(out_dir / "test_metrics.json", metrics)
    test_metadata = [metadata[idx] for idx in test_idx]
    _prediction_frame(test_metadata, labels[test_idx], split_df, y_pred_by_index={idx: pred for idx, pred in enumerate(y_pred)}, classes=label_space.classes).to_csv(
        out_dir / "predictions.csv", index=False
    )
    return out_dir


def run_linear_probe_study(cfg: StudyConfig) -> list[Path]:
    return [run_linear_probe_for_model(cfg, model) for model in cfg.models]


def _records_for_split(records: list[DiseaseImageRecord], split_df: pd.DataFrame, split: str) -> list[DiseaseImageRecord]:
    wanted = set(split_df.loc[split_df["split"] == split, "manifest_row_index"].astype(int))
    return [record for record in records if record.row_index in wanted]


def _image_loader(cfg: StudyConfig, records: list[DiseaseImageRecord], batch_size: int, shuffle: bool) -> DataLoader:
    ds = DiseaseImageDataset(cfg.dataset, records)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.extraction.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=lambda batch: (torch.stack([item[0] for item in batch], dim=0), [item[1] for item in batch]),
    )


def _unfreeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(True)


def run_finetune_for_model(cfg: StudyConfig, model_spec: ModelSpec, method: str | None = None) -> Path:
    records = load_manifest_records(cfg.dataset)
    split_df = load_or_create_split(cfg)
    label_space = label_space_from_records(records, cfg.supervised.label_column)
    train_loader = _image_loader(cfg, _records_for_split(records, split_df, "train"), cfg.supervised.finetune_batch_size, shuffle=True)
    all_loader = _image_loader(cfg, records, cfg.supervised.finetune_batch_size, shuffle=False)
    test_records = _records_for_split(records, split_df, "test")
    test_loader = _image_loader(cfg, test_records, cfg.supervised.finetune_batch_size, shuffle=False)

    feature_model, train_cfg, checkpoint_path = load_feature_model_for_run(_to_landmark_run(model_spec))
    _unfreeze(feature_model)
    embedding_dim = expected_embedding_dim(train_cfg, cfg.pooling)
    model = FineTuneClassifier(feature_model, embedding_dim=embedding_dim, num_classes=len(label_space.classes), pooling=cfg.pooling)
    device = _resolve_device(cfg.extraction.device)
    model = model.to(device)
    weights = _class_weights(label_space.encode([record.metadata[cfg.supervised.label_column] for record in _records_for_split(records, split_df, "train")]), len(label_space.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(
        [
            {"params": model.feature_model.parameters(), "lr": cfg.supervised.backbone_lr},
            {"params": model.head.parameters(), "lr": cfg.supervised.head_lr},
        ],
        weight_decay=cfg.supervised.weight_decay,
    )
    autocast_dtype = _autocast_dtype(cfg.extraction.precision)
    amp_enabled = device.type == "cuda" and autocast_dtype is not None

    history = []
    for epoch in range(1, cfg.supervised.finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xs, batch_records in tqdm(train_loader, desc=f"finetune {model_spec.label}"):
            xs = xs.to(device, non_blocking=True)
            y = label_space.encode([record.metadata[cfg.supervised.label_column] for record in batch_records]).to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                logits, _ = model(xs)
                loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xs.shape[0]
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(xs.shape[0])
        history.append({"epoch": epoch, "train_loss": total_loss / max(total, 1), "train_accuracy": total_correct / max(total, 1)})

    out_dir = finetune_dir(cfg, model_spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "classes": label_space.classes,
            "embedding_dim": embedding_dim,
            "model_id": model_spec.model_id,
            "run_name": model_spec.run_name,
            "checkpoint_path": str(checkpoint_path),
        },
        out_dir / "finetuned_model.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    test_logits, test_labels, test_meta, _ = _collect_finetune_outputs(model, test_loader, label_space, cfg, device)
    y_true = test_labels.numpy()
    y_pred = test_logits.argmax(dim=1).numpy()
    metrics = _metrics(y_true, y_pred, label_space.classes)
    metrics.update(
        {
            "model_id": model_spec.model_id,
            "model_label": model_spec.label,
            "run_name": model_spec.run_name,
            "checkpoint_step": model_spec.checkpoint_step,
            "checkpoint_path": str(checkpoint_path),
            "embedding_dim": embedding_dim,
            "train_rows": int((split_df["split"] == "train").sum()),
            "test_rows": int((split_df["split"] == "test").sum()),
        }
    )
    _write_json(out_dir / "test_metrics.json", metrics)
    _prediction_frame(test_meta, test_labels, split_df, y_pred_by_index={idx: pred for idx, pred in zip(range(len(test_meta)), y_pred)}, classes=label_space.classes).to_csv(
        out_dir / "predictions.csv", index=False
    )

    all_logits, all_labels, all_meta, all_embeddings = _collect_finetune_outputs(model, all_loader, label_space, cfg, device)
    all_pred = all_logits.argmax(dim=1).numpy()
    split_by_row = {int(row.manifest_row_index): str(row.split) for row in split_df.itertuples(index=False)}
    embedding_payload = {
        "embeddings": all_embeddings,
        "metadata": all_meta,
        "labels": all_labels,
        "predictions": torch.tensor(all_pred, dtype=torch.long),
        "classes": label_space.classes,
        "model": {
            "model_id": model_spec.model_id,
            "label": model_spec.label,
            "source": model_spec.source,
            "external_model": model_spec.external_model,
            "run_name": model_spec.run_name,
            "run_dir": str(model_spec.run_dir) if model_spec.run_dir is not None else None,
        },
        "checkpoint": {
            "checkpoint_step": model_spec.checkpoint_step,
            "checkpoint_path": str(checkpoint_path),
        },
        "embedding_key": "finetuned_g4",
        "pooling": cfg.pooling,
        "embedding_dim": embedding_dim,
        "split_by_manifest_row_index": split_by_row,
        "train_cfg": train_cfg,
    }
    torch.save(embedding_payload, out_dir / "adapted_embeddings.pt")
    _write_adapted_plots(cfg, model_spec, embedding_payload, method or cfg.reduction.method)
    return out_dir


@torch.no_grad()
def _collect_finetune_outputs(
    model: FineTuneClassifier,
    loader: DataLoader,
    label_space: LabelSpace,
    cfg: StudyConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], torch.Tensor]:
    model.eval()
    logits_out = []
    labels_out = []
    metadata = []
    embeddings_out = []
    for xs, records in loader:
        xs = xs.to(device, non_blocking=True)
        logits, embeddings = model(xs)
        logits_out.append(logits.cpu())
        labels_out.append(label_space.encode([record.metadata[cfg.supervised.label_column] for record in records]))
        embeddings_out.append(embeddings.cpu().float())
        for record in records:
            row = dict(record.metadata)
            row["manifest_row_index"] = int(record.row_index)
            metadata.append(row)
    return torch.cat(logits_out, dim=0), torch.cat(labels_out, dim=0), metadata, torch.cat(embeddings_out, dim=0)


def _prediction_frame(
    metadata: list[dict[str, Any]],
    labels: torch.Tensor,
    split_df: pd.DataFrame,
    y_pred_by_index: dict[int, int],
    classes: tuple[str, ...],
) -> pd.DataFrame:
    split_by_row = {int(row.manifest_row_index): str(row.split) for row in split_df.itertuples(index=False)}
    rows = []
    for idx, row in enumerate(metadata):
        true_idx = int(labels[idx].item())
        pred_idx = y_pred_by_index.get(idx)
        out = dict(row)
        out["split"] = split_by_row.get(int(row["manifest_row_index"]), "")
        out["true_label"] = classes[true_idx]
        out["true_label_idx"] = true_idx
        out["pred_label"] = classes[pred_idx] if pred_idx is not None else ""
        out["pred_label_idx"] = pred_idx if pred_idx is not None else ""
        out["correct"] = bool(pred_idx == true_idx) if pred_idx is not None else ""
        rows.append(out)
    return pd.DataFrame(rows)


def _adapted_frame(payload: dict[str, Any], coords: np.ndarray, method: str, split: str | None = None) -> pd.DataFrame:
    metadata = payload["metadata"]
    labels = payload["labels"]
    predictions = payload["predictions"]
    classes = tuple(payload["classes"])
    split_by_row = {int(k): v for k, v in payload["split_by_manifest_row_index"].items()}
    rows = []
    for idx, row in enumerate(metadata):
        row_split = split_by_row[int(row["manifest_row_index"])]
        if split is not None and row_split != split:
            continue
        true_idx = int(labels[idx].item())
        pred_idx = int(predictions[idx].item())
        out = dict(row)
        out["x"] = float(coords[idx, 0])
        out["y"] = float(coords[idx, 1])
        out["reducer"] = method
        out["split"] = row_split
        out["true_label"] = classes[true_idx]
        out["pred_label"] = classes[pred_idx]
        out["correct"] = bool(true_idx == pred_idx)
        out["model_id"] = payload["model"]["model_id"]
        out["model_label"] = payload["model"]["label"]
        out["run_name"] = payload["model"]["run_name"]
        out["checkpoint_step"] = payload["checkpoint"]["checkpoint_step"]
        out["checkpoint_path"] = payload["checkpoint"]["checkpoint_path"]
        out["embedding_key"] = payload["embedding_key"]
        out["pooling"] = payload["pooling"]
        out["embedding_dim"] = payload["embedding_dim"]
        rows.append(out)
    return pd.DataFrame(rows)


def _write_adapted_plots(cfg: StudyConfig, model: ModelSpec, payload: dict[str, Any], method: str) -> None:
    method = method.lower()
    coords = reduce_embeddings(
        payload["embeddings"],
        method=method,
        random_state=cfg.reduction.random_state,
        tsne_perplexity=cfg.reduction.tsne_perplexity,
    )
    disease_order = sorted({str(row[cfg.supervised.label_column]) for row in payload["metadata"]})
    for split in (None, "test"):
        df = _adapted_frame(payload, coords, method, split=split)
        csv_path = adapted_reduction_csv_path(cfg, model, method, split=split)
        fig_path = adapted_figure_path(cfg, model, method, split=split)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        title_suffix = f" - {cfg.supervised.finetune_epochs} Epoch Disease Fine-Tune"
        if split == "test":
            title_suffix += " (Test)"
        plot_coordinates(df, model.label + title_suffix, fig_path, disease_order)


def run_finetune_study(cfg: StudyConfig, method: str | None = None) -> list[Path]:
    return [run_finetune_for_model(cfg, model, method=method) for model in cfg.models]
