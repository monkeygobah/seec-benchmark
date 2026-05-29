from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from embedding_extract.pipeline_config import DatasetSpec, ExtractionSpec
from src.dataset_utils import ImageFolderDataset, ImageSample, ManifestImageDataset


def build_inference_transform(dataset: DatasetSpec):
    ops: list[Any] = [
        transforms.Resize((dataset.image_size, dataset.image_size)),
        transforms.ToTensor(),
    ]
    if dataset.normalize_imagenet:
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(ops)


def _collate(batch: list[tuple[torch.Tensor, ImageSample]]):
    imgs, metas = zip(*batch)
    return torch.stack(list(imgs), dim=0), list(metas)


def build_dataloader(dataset: DatasetSpec, extraction: ExtractionSpec) -> DataLoader:
    if dataset.manifest is not None:
        ds = ManifestImageDataset(
            root=dataset.root,
            manifest=dataset.manifest,
            transform=build_inference_transform(dataset),
        )
    else:
        ds = ImageFolderDataset(root=dataset.root, transform=build_inference_transform(dataset))
    if len(ds) == 0:
        raise ValueError(f"Dataset is empty: {dataset.root}")
    return DataLoader(
        ds,
        batch_size=extraction.batch_size,
        shuffle=False,
        num_workers=extraction.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate,
    )
