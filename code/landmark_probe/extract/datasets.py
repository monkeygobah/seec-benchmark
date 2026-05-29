from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from landmark_probe.config import DatasetSpec, ExtractionSpec, TaskSplitSpec


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    dataset_name: str
    image_rel_path: str
    image_name: str
    anatomical_side: str


class ProbeImageDataset(Dataset):
    def __init__(self, root: Path, records: list[ImageRecord], image_size: int, normalize_imagenet: bool):
        from torchvision import transforms

        self.root = root
        self.records = records
        ops: list[object] = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if normalize_imagenet:
            ops.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(self.root / record.image_rel_path).convert("RGB")
        return self.transform(image), record


def _collate(batch: list[tuple[torch.Tensor, ImageRecord]]):
    xs, records = zip(*batch)
    return torch.stack(list(xs), dim=0), list(records)


def load_split_records(dataset_cfg: DatasetSpec, split_spec: TaskSplitSpec) -> list[ImageRecord]:
    manifest_df = pd.read_csv(dataset_cfg.metadata.manifest_csv)
    split_df = pd.read_csv(dataset_cfg.metadata.split_csv)
    split_ids = set(
        split_df.loc[
            (split_df["dataset_name"] == split_spec.dataset_name) & (split_df["split"] == split_spec.split),
            "sample_id",
        ]
    )
    subset = manifest_df.loc[
        (manifest_df["dataset_name"] == split_spec.dataset_name) & (manifest_df["sample_id"].isin(split_ids))
    ]
    if subset.empty:
        raise ValueError(f"No manifest rows found for {split_spec.dataset_name}:{split_spec.split}")
    return [
        ImageRecord(
            sample_id=str(row.sample_id),
            dataset_name=str(row.dataset_name),
            image_rel_path=str(row.image_rel_path),
            image_name=str(row.image_name),
            anatomical_side=str(row.anatomical_side),
        )
        for row in subset.itertuples(index=False)
    ]


def build_dataloader(dataset_cfg: DatasetSpec, split_spec: TaskSplitSpec, extraction: ExtractionSpec) -> DataLoader:
    records = load_split_records(dataset_cfg, split_spec)
    ds = ProbeImageDataset(
        root=dataset_cfg.root,
        records=records,
        image_size=dataset_cfg.image_size,
        normalize_imagenet=dataset_cfg.normalize_imagenet,
    )
    return DataLoader(
        ds,
        batch_size=extraction.batch_size,
        shuffle=False,
        num_workers=extraction.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        drop_last=False,
    )
