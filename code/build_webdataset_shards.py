from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from src.dataset_utils import IMAGE_SUFFIXES


def iter_manifest(path: Path):
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw and not raw.startswith("#"):
                yield Path(raw)


def write_shards(
    manifest: Path,
    out_dir: Path,
    *,
    root: Path | None,
    shard_size: int,
    prefix: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_list_path = out_dir / f"{prefix}_shards.txt"

    shard_paths: list[Path] = []
    tar: tarfile.TarFile | None = None
    shard_idx = -1
    count_in_shard = 0
    total = 0

    def open_next_shard() -> tarfile.TarFile:
        nonlocal shard_idx, count_in_shard
        shard_idx += 1
        count_in_shard = 0
        shard_path = out_dir / f"{prefix}-{shard_idx:06d}.tar"
        shard_paths.append(shard_path)
        return tarfile.open(shard_path, "w")

    try:
        for image_path in iter_manifest(manifest):
            if root is not None and not image_path.is_absolute():
                image_path = root / image_path
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            if tar is None or count_in_shard >= shard_size:
                if tar is not None:
                    tar.close()
                tar = open_next_shard()

            try:
                rel = image_path.relative_to(root) if root is not None else image_path.name
            except ValueError:
                rel = image_path.name

            stem = Path(rel).with_suffix("").as_posix()
            suffix = image_path.suffix.lower().lstrip(".")
            arcname = f"{stem}.{suffix}"
            tar.add(image_path, arcname=arcname, recursive=False)
            count_in_shard += 1
            total += 1
    finally:
        if tar is not None:
            tar.close()

    with shard_list_path.open("w") as f:
        for shard_path in shard_paths:
            f.write(str(shard_path) + "\n")

    print(f"wrote {total} images into {len(shard_paths)} shards")
    print(shard_list_path)
    return shard_list_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--prefix", default="train")
    args = ap.parse_args()

    write_shards(
        manifest=Path(args.manifest),
        out_dir=Path(args.out_dir),
        root=Path(args.root) if args.root else None,
        shard_size=args.shard_size,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
