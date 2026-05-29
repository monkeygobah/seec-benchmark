from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from landmark_probe.config import DatasetSpec, TaskSplitSpec


@dataclass(frozen=True)
class ProbeRow:
    sample_id: str
    x: torch.Tensor
    y: torch.Tensor


class ProbeTensorDataset(Dataset):
    def __init__(self, rows: list[ProbeRow]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return row.x, row.y, row.sample_id


def _target_columns(dataset_cfg: DatasetSpec) -> list[str]:
    cols: list[str] = []
    for key in dataset_cfg.landmarks:
        cols.append(f"{key}_x")
        cols.append(f"{key}_y")
    return cols


def load_embedding_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {path}")
    return payload


def build_probe_dataset(dataset_cfg: DatasetSpec, payload_path: Path, split_spec: TaskSplitSpec) -> ProbeTensorDataset:
    payload = load_embedding_payload(payload_path)
    sample_ids = payload["sample_ids"]
    embeddings = payload["embeddings"].float()
    if embeddings.shape[0] != len(sample_ids):
        raise ValueError(f"Embedding payload row mismatch at {payload_path}")

    split_df = pd.read_csv(dataset_cfg.metadata.split_csv)
    split_ids = set(
        split_df.loc[
            (split_df["dataset_name"] == split_spec.dataset_name) & (split_df["split"] == split_spec.split),
            "sample_id",
        ]
    )
    if set(sample_ids) != split_ids:
        raise ValueError(f"Embedding sample IDs do not match canonical split for {split_spec.dataset_name}:{split_spec.split}")

    landmarks_df = pd.read_csv(dataset_cfg.metadata.landmarks_csv)
    target_cols = _target_columns(dataset_cfg)
    target_subset = landmarks_df.loc[
        (landmarks_df["dataset_name"] == split_spec.dataset_name) & (landmarks_df["sample_id"].isin(split_ids)),
        ["sample_id", *target_cols],
    ]
    target_map = {
        str(row.sample_id): torch.tensor([float(getattr(row, col)) for col in target_cols], dtype=torch.float32)
        for row in target_subset.itertuples(index=False)
    }

    rows = []
    for idx, sample_id in enumerate(sample_ids):
        if sample_id not in target_map:
            raise ValueError(f"Missing target row for {sample_id}")
        rows.append(ProbeRow(sample_id=str(sample_id), x=embeddings[idx], y=target_map[sample_id]))
    return ProbeTensorDataset(rows)


def build_dataloader(ds: ProbeTensorDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
