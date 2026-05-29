from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path.relative_to(root).as_posix()


def write_manifest(rows: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset6-root", required=True)
    ap.add_argument("--out-dir", default="manifests")
    ap.add_argument("--limit-10k", type=int, default=10_000)
    args = ap.parse_args()

    subset6_root = Path(args.subset6_root)
    if not subset6_root.is_dir():
        raise FileNotFoundError(f"subset6 root does not exist: {subset6_root}")

    rows = list(iter_images(subset6_root))
    if len(rows) < args.limit_10k:
        raise ValueError(f"Need at least {args.limit_10k} images for Pretrain-10K, found {len(rows)}")

    out_dir = Path(args.out_dir)
    write_manifest(rows[: args.limit_10k], out_dir / "pretrain/pretrain_10k.txt")
    print(f"Wrote {args.limit_10k} rows to {out_dir / 'pretrain/pretrain_10k.txt'}")


if __name__ == "__main__":
    main()
