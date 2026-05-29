from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from disease_embeddings.config import ModelSpec, StudyConfig
from disease_embeddings.datasets import load_manifest_records
from disease_embeddings.paths import embedding_artifact_path, figure_path, reduction_csv_path


DISEASE_COLORS = {
    "CAP": "#4E79A7",
    "ptosis": "#F28E2B",
    "TREACHER_COLLINS": "#59A14F",
    "TED": "#E15759",
    "chalazion": "#B07AA1",
    "blind-eye": "#9C755F",
    "ectropion": "#76B7B2",
    "hidrocystoma": "#EDC948",
}


def _load_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict artifact at {path}")
    if "embeddings" not in payload or "metadata" not in payload:
        raise ValueError(f"Artifact is missing embeddings or metadata: {path}")
    return payload


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (x - mean) / std


def reduce_embeddings(embeddings: torch.Tensor, method: str, random_state: int = 0, tsne_perplexity: float = 30.0) -> np.ndarray:
    method = method.lower()
    x = _standardize(embeddings.detach().cpu().numpy())
    if x.shape[0] < 2:
        raise ValueError("At least two rows are required for dimensionality reduction")
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(float(tsne_perplexity), max(1.0, float(x.shape[0] - 1) / 3.0))
        kwargs: dict[str, Any] = {
            "n_components": 2,
            "perplexity": perplexity,
            "random_state": random_state,
            "init": "pca",
            "learning_rate": "auto",
        }
        signature = inspect.signature(TSNE)
        if "max_iter" in signature.parameters:
            kwargs["max_iter"] = 1000
        else:
            kwargs["n_iter"] = 1000
        return TSNE(**kwargs).fit_transform(x)
    raise ValueError(f"Unsupported reduction method: {method}")


def _metadata_frame(payload: dict[str, Any], coords: np.ndarray, method: str) -> pd.DataFrame:
    df = pd.DataFrame(payload["metadata"])
    df.insert(0, "x", coords[:, 0])
    df.insert(1, "y", coords[:, 1])
    df.insert(2, "reducer", method)
    model = payload["model"]
    checkpoint = payload["checkpoint"]
    df["model_id"] = model["model_id"]
    df["model_label"] = model["label"]
    df["model_source"] = model["source"]
    df["external_model"] = model.get("external_model")
    df["run_name"] = model["run_name"]
    df["run_dir"] = model.get("run_dir")
    df["checkpoint_step"] = checkpoint["checkpoint_step"]
    df["checkpoint_path"] = checkpoint["checkpoint_path"]
    df["embedding_key"] = payload["embedding_key"]
    df["pooling"] = payload["pooling"]
    df["embedding_dim"] = payload["embedding_dim"]
    return df


def _disease_order(cfg: StudyConfig) -> list[str]:
    labels = [record.folder_label for record in load_manifest_records(cfg.dataset)]
    return sorted(set(labels))


def plot_coordinates(df: pd.DataFrame, model_label: str, out_path: Path, disease_order: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    fallback_colors = plt.get_cmap("tab20").colors
    for idx, disease in enumerate(disease_order):
        subset = df[df["folder_label"] == disease]
        if subset.empty:
            continue
        color = DISEASE_COLORS.get(disease, fallback_colors[idx % len(fallback_colors)])
        ax.scatter(subset["x"], subset["y"], s=28, alpha=0.82, linewidths=0, label=disease, color=color)

    ax.set_title(model_label, fontsize=15)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def reduce_and_plot_model(cfg: StudyConfig, model: ModelSpec, method: str) -> tuple[Path, Path]:
    method = method.lower()
    payload = _load_artifact(embedding_artifact_path(cfg, model))
    coords = reduce_embeddings(
        payload["embeddings"],
        method=method,
        random_state=cfg.reduction.random_state,
        tsne_perplexity=cfg.reduction.tsne_perplexity,
    )
    df = _metadata_frame(payload, coords, method)
    csv_path = reduction_csv_path(cfg, model, method)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    fig_path = figure_path(cfg, model, method)
    plot_coordinates(df, model.label, fig_path, _disease_order(cfg))
    return csv_path, fig_path


def reduce_and_plot_study(cfg: StudyConfig, method: str | None = None) -> list[tuple[Path, Path]]:
    reducer = (method or cfg.reduction.method).lower()
    return [reduce_and_plot_model(cfg, model, reducer) for model in cfg.models]

