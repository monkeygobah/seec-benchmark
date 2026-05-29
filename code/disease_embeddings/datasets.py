from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from disease_embeddings.config import DatasetSpec, ExtractionSpec


REQUIRED_MANIFEST_COLUMNS = (
    "output_path",
    "source_image_path",
    "folder_label",
    "filename",
    "eye",
    "disease_status",
    "source_mode",
)


@dataclass(frozen=True)
class DiseaseImageRecord:
    row_index: int
    metadata: dict[str, Any]

    @property
    def output_path(self) -> str:
        return str(self.metadata["output_path"])

    @property
    def folder_label(self) -> str:
        return str(self.metadata["folder_label"])


class DiseaseImageDataset(Dataset):
    def __init__(self, cfg: DatasetSpec, records: list[DiseaseImageRecord]):
        self.cfg = cfg
        self.records = records

    def _transform(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.cfg.normalize_imagenet:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image_path = self.cfg.root / record.output_path
        image = Image.open(image_path).convert("RGB")
        return self._transform(image), record


def _stringify_record(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[str(key)] = ""
        else:
            out[str(key)] = str(value)
    return out


def load_manifest_records(cfg: DatasetSpec) -> list[DiseaseImageRecord]:
    df = pd.read_csv(cfg.manifest_csv)
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Disease manifest missing required columns: {missing}")
    records: list[DiseaseImageRecord] = []
    for idx, row in df.iterrows():
        metadata = _stringify_record(row)
        image_path = cfg.root / str(metadata["output_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
        records.append(DiseaseImageRecord(row_index=int(idx), metadata=metadata))
    if not records:
        raise ValueError(f"Disease manifest is empty: {cfg.manifest_csv}")
    return records


def _collate(batch: list[tuple[torch.Tensor, DiseaseImageRecord]]):
    xs, records = zip(*batch)
    return torch.stack(list(xs), dim=0), list(records)


def build_dataloader(cfg: DatasetSpec, extraction: ExtractionSpec) -> DataLoader:
    ds = DiseaseImageDataset(cfg, load_manifest_records(cfg))
    return DataLoader(
        ds,
        batch_size=extraction.batch_size,
        shuffle=False,
        num_workers=extraction.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate,
    )
