from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from landmark_probe.config import DatasetSpec, ProbeConfig, RepresentationSpec, RunSpec, StudyConfig, TaskSpec
from landmark_probe.constants import LANDMARK_KEYS
from landmark_probe.extract.inference import expected_embedding_dim
from landmark_probe.paths import embedding_artifact_path, probe_run_dir
from landmark_probe.probe.datasets import build_dataloader, build_probe_dataset, load_embedding_payload
from landmark_probe.probe.metrics import mean_l2_per_landmark, per_landmark_stats, per_sample_mean_l2
from landmark_probe.probe.model import MLPRegressor


def _run_epoch(model, loader, opt, device, k: int, train: bool) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()
    mae = torch.nn.L1Loss()

    total_mae = 0.0
    total_l2 = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            yhat = model(x)
            loss = mae(yhat, y)
            l2 = mean_l2_per_landmark(yhat, y, k)
            if train:
                loss.backward()
                opt.step()
        batch_size = x.shape[0]
        total_mae += float(loss.item()) * batch_size
        total_l2 += float(l2.item()) * batch_size
        n += batch_size
    return total_mae / max(n, 1), total_l2 / max(n, 1)


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_spec)


@torch.no_grad()
def _collect_predictions(model, loader, device, k: int) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    model.eval()
    ys = []
    yhats = []
    sample_ids: list[str] = []
    for x, y, sid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        yhat = model(x)
        ys.append(y.cpu())
        yhats.append(yhat.cpu())
        sample_ids.extend(str(v) for v in sid)
    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(yhats, dim=0)
    return y_true, y_pred, sample_ids, per_sample_mean_l2(y_pred, y_true, k)


def _ensure_disjoint(task: TaskSpec, dataset_cfg: DatasetSpec) -> None:
    split_df = pd.read_csv(dataset_cfg.metadata.split_csv)

    def ids(spec):
        return set(
            split_df.loc[
                (split_df["dataset_name"] == spec.dataset_name) & (split_df["split"] == spec.split),
                "sample_id",
            ]
        )

    train_ids = ids(task.train_split)
    val_ids = ids(task.val_split)
    test_ids = ids(task.test_split)
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError(f"Split leakage detected for task {task.task_name}")


def run_probe_for_target(
    study_cfg: StudyConfig,
    dataset_cfg: DatasetSpec,
    probe_cfg: ProbeConfig,
    task: TaskSpec,
    run: RunSpec,
    representation: RepresentationSpec,
) -> Path:
    _ensure_disjoint(task, dataset_cfg)

    train_path = embedding_artifact_path(study_cfg, run, task.train_split, representation)
    val_path = embedding_artifact_path(study_cfg, run, task.val_split, representation)
    test_path = embedding_artifact_path(study_cfg, run, task.test_split, representation)
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Embedding artifact missing: {path}")
    train_payload = load_embedding_payload(train_path)
    train_cfg = train_payload.get("train_cfg", {})
    model_cfg = train_cfg.get("model", {})
    ssl_method = str(train_payload.get("ssl_method", train_cfg.get("ssl", {}).get("method", "unknown")))
    init_mode = str(train_payload.get("init_mode", model_cfg.get("init", "unknown")))
    external_metadata = {
        "model_family": train_payload.get("model_family", model_cfg.get("model_family")),
        "model_arch": train_payload.get("model_arch", model_cfg.get("model_arch")),
        "patch_size": train_payload.get("patch_size", model_cfg.get("patch_size")),
        "pretrain_data": train_payload.get("pretrain_data", model_cfg.get("pretrain_data")),
        "feature_kind": train_payload.get("feature_kind", model_cfg.get("feature_kind")),
    }
    expected_dim = expected_embedding_dim(train_cfg, representation.pooling)
    observed_dim = int(train_payload["embeddings"].shape[1])
    if observed_dim != expected_dim:
        raise ValueError(
            f"Embedding dim mismatch at {train_path}: got {observed_dim}, expected {expected_dim}. "
            "Re-run landmark extraction with overwrite enabled."
        )

    train_ds = build_probe_dataset(dataset_cfg, train_path, task.train_split)
    val_ds = build_probe_dataset(dataset_cfg, val_path, task.val_split)
    test_ds = build_probe_dataset(dataset_cfg, test_path, task.test_split)

    train_loader = build_dataloader(train_ds, probe_cfg.batch_size, probe_cfg.num_workers, shuffle=True)
    val_loader = build_dataloader(val_ds, probe_cfg.batch_size, probe_cfg.num_workers, shuffle=False)
    test_loader = build_dataloader(test_ds, probe_cfg.batch_size, probe_cfg.num_workers, shuffle=False)

    device = _resolve_device(study_cfg.extraction.device)
    in_dim = int(train_ds[0][0].shape[0])
    out_dim = 2 * len(dataset_cfg.landmarks)
    k = len(dataset_cfg.landmarks)
    model = MLPRegressor(in_dim=in_dim, out_dim=out_dim, hidden_dims=probe_cfg.hidden_dims, dropout=probe_cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=probe_cfg.lr, weight_decay=probe_cfg.weight_decay)

    history_rows = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improve = 0
    for epoch in range(1, probe_cfg.max_epochs + 1):
        train_mae, train_l2 = _run_epoch(model, train_loader, opt, device, k, train=True)
        val_mae, val_l2 = _run_epoch(model, val_loader, opt, device, k, train=False)
        history_rows.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "train_mean_l2": train_l2,
                "val_mae": val_mae,
                "val_mean_l2": val_l2,
            }
        )

        if val_l2 < best_val:
            best_val = val_l2
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if probe_cfg.early_stopping_enabled and epochs_without_improve >= probe_cfg.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Probe training never produced a best state")
    model.load_state_dict(best_state)

    y_true, y_pred, sample_ids, per_sample_l2 = _collect_predictions(model, test_loader, device, k)
    test_mae = float(torch.nn.functional.l1_loss(y_pred, y_true).item())
    test_mean_l2 = float(mean_l2_per_landmark(y_pred, y_true, k).item())

    out_dir = probe_run_dir(study_cfg, task.task_name, run, representation)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_probe.pt"
    torch.save(
        {
            "model": best_state,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "hidden_dims": probe_cfg.hidden_dims,
            "dropout": probe_cfg.dropout,
            "best_epoch": best_epoch,
            "best_val_mean_l2": best_val,
            "run_name": run.run_name,
            "checkpoint_step": run.checkpoint_step,
            "task_name": task.task_name,
            "dataset_name": task.test_split.dataset_name,
            "split": task.test_split.split,
            "embedding_key": representation.embedding_key,
            "pooling": representation.pooling,
            "embedding_dim": in_dim,
            "ssl_method": ssl_method,
            "init_mode": init_mode,
            **external_metadata,
        },
        best_ckpt_path,
    )

    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)
    with (out_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study_cfg": str(study_cfg.path),
                "probe_cfg": str(probe_cfg.path),
                "task_name": task.task_name,
                "run_name": run.run_name,
                "checkpoint_step": run.checkpoint_step,
                "representation": {
                    "embedding_key": representation.embedding_key,
                    "pooling": representation.pooling,
                },
            },
            f,
            sort_keys=False,
        )

    metrics_payload = {
        "study_name": study_cfg.name,
        "task_name": task.task_name,
        "run_name": run.run_name,
        "checkpoint_step": run.checkpoint_step,
        "embedding_key": representation.embedding_key,
        "pooling": representation.pooling,
        "embedding_dim": in_dim,
        "ssl_method": ssl_method,
        "init_mode": init_mode,
        **external_metadata,
        "train_dataset_name": task.train_split.dataset_name,
        "val_dataset_name": task.val_split.dataset_name,
        "test_dataset_name": task.test_split.dataset_name,
        "test_split": task.test_split.split,
        "best_epoch": best_epoch,
        "best_val_mean_l2": best_val,
        "test_mae": test_mae,
        "test_mean_l2": test_mean_l2,
        "probe_checkpoint_path": str(best_ckpt_path),
    }
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    per_landmark_rows = per_landmark_stats(y_pred, y_true, dataset_cfg.landmarks)
    for row in per_landmark_rows:
        row.update(
            {
                "study_name": study_cfg.name,
                "task_name": task.task_name,
                "run_name": run.run_name,
                "checkpoint_step": run.checkpoint_step,
                "embedding_key": representation.embedding_key,
                "pooling": representation.pooling,
                "embedding_dim": in_dim,
                "ssl_method": ssl_method,
                "init_mode": init_mode,
                **external_metadata,
                "best_epoch": best_epoch,
                "best_val_mean_l2": best_val,
            }
        )
    pd.DataFrame(per_landmark_rows).to_csv(out_dir / "per_landmark.csv", index=False)

    per_sample_rows = []
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    per_sample_l2_np = per_sample_l2.numpy()
    for row_idx, sample_id in enumerate(sample_ids):
        row = {"sample_id": sample_id, "mean_l2": float(per_sample_l2_np[row_idx])}
        for landmark_idx, landmark in enumerate(dataset_cfg.landmarks):
            base = 2 * landmark_idx
            true_x = float(y_true_np[row_idx, base])
            true_y = float(y_true_np[row_idx, base + 1])
            pred_x = float(y_pred_np[row_idx, base])
            pred_y = float(y_pred_np[row_idx, base + 1])
            err_x = pred_x - true_x
            err_y = pred_y - true_y
            row[f"{landmark}_true_x"] = true_x
            row[f"{landmark}_true_y"] = true_y
            row[f"{landmark}_pred_x"] = pred_x
            row[f"{landmark}_pred_y"] = pred_y
            row[f"{landmark}_err_x"] = err_x
            row[f"{landmark}_err_y"] = err_y
            row[f"{landmark}_l2"] = float((err_x**2 + err_y**2) ** 0.5)
        per_sample_rows.append(row)
    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_df.to_csv(out_dir / "per_sample.csv", index=False)
    return out_dir


def run_probe_study(study_cfg: StudyConfig, dataset_cfg: DatasetSpec, probe_cfg: ProbeConfig) -> list[Path]:
    written: list[Path] = []
    for task in study_cfg.tasks:
        for run in study_cfg.runs:
            for representation in study_cfg.representations:
                written.append(run_probe_for_target(study_cfg, dataset_cfg, probe_cfg, task, run, representation))
    return written
