from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset, IterableDataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    stem: str
    filename: str
    path: Path
    rel_path: Path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._index()

    def _index(self) -> list[ImageSample]:
        out: list[ImageSample] = []
        for p in sorted(self.root.rglob("*")):
            if not is_image_file(p):
                continue

            rel_path = p.relative_to(self.root)
            sample = ImageSample(
                stem=p.stem,
                filename=p.name,
                path=p,
                rel_path=rel_path,
            )
            out.append(sample)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        manifest: Path,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.transform = transform
        self.samples = self._index()

    def _index(self) -> list[ImageSample]:
        out: list[ImageSample] = []
        with self.manifest.open("r") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue

                p = Path(raw)
                if not p.is_absolute():
                    p = self.root / p
                if p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                try:
                    rel_path = p.relative_to(self.root)
                except ValueError:
                    rel_path = Path(p.name)

                out.append(
                    ImageSample(
                        stem=p.stem,
                        filename=p.name,
                        path=p,
                        rel_path=rel_path,
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s


class WebDatasetImageDataset(IterableDataset):
    def __init__(
        self,
        shards,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        shuffle_buffer: int = 1000,
    ):
        try:
            import webdataset as wds
        except ImportError as exc:
            raise ImportError(
                "data.train_shards requires webdataset. Install it with "
                "`python -m pip install webdataset`."
            ) from exc

        self.shards = _resolve_train_shards(shards)
        self.transform = transform
        self.shuffle = bool(shuffle)
        self.shuffle_buffer = int(shuffle_buffer)
        self._wds = wds

    def __iter__(self):
        wds = self._wds
        ds = wds.WebDataset(
            self.shards,
            shardshuffle=self.shuffle,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        if self.shuffle:
            ds = ds.shuffle(self.shuffle_buffer)
        ds = ds.decode("pil").to_tuple("jpg;jpeg;png;bmp;tif;tiff;webp", "__key__")

        for img, key in ds:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

            key_path = Path(str(key))
            sample = ImageSample(
                stem=key_path.stem,
                filename=key_path.name,
                path=key_path,
                rel_path=key_path,
            )
            yield img, sample


def _read_shard_list(path: Path) -> list[str]:
    shards: list[str] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw and not raw.startswith("#"):
                shards.append(raw)
    return shards


def _resolve_train_shards(value) -> list[str] | str:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]

    raw = str(value)
    if raw.startswith("@"):
        return _read_shard_list(Path(raw[1:]))

    path = Path(raw)
    if path.suffix == ".txt" and path.is_file():
        return _read_shard_list(path)

    if any(ch in raw for ch in "*?[]"):
        matches = sorted(p for p in glob.glob(raw) if Path(p).is_file())
        if matches:
            return matches

    return raw


def is_iterable_dataset(ds) -> bool:
    return isinstance(ds, IterableDataset)


def build_dataset(cfg, transform=None, root_key: str = "train_root"):
    data_cfg = cfg.get("data", {})
    shards_key = root_key.replace("_root", "_shards")
    shards = data_cfg.get(shards_key)
    if shards:
        return WebDatasetImageDataset(
            shards=_resolve_train_shards(shards),
            transform=transform,
            shuffle=bool(data_cfg.get("shard_shuffle", True)),
            shuffle_buffer=int(data_cfg.get("shard_shuffle_buffer", 1000)),
        )

    root = data_cfg.get(root_key)
    if root is None:
        raise ValueError(f"Missing data.{root_key} in config.")

    manifest_key = root_key.replace("_root", "_manifest")
    manifest = data_cfg.get(manifest_key)
    if manifest:
        return ManifestImageDataset(
            root=root,
            manifest=manifest,
            transform=transform,
        )

    return ImageFolderDataset(
        root=root,
        transform=transform,
    )
