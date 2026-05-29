from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_DISEASE_COLUMNS = {"image_path", "label", "group_id"}


def _require_dir(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_dir():
        missing.append(f"{label}: {path}")


def _validate_disease_manifest(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    missing = REQUIRED_DISEASE_COLUMNS - columns
    if missing:
        raise ValueError(f"Disease manifest missing required columns: {sorted(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    missing: list[str] = []
    _require_dir(root / "subset6", "subset6 corpus", missing)
    _require_dir(root / "landmark_raw/celeb/images", "Celeb landmark images", missing)
    _require_dir(root / "landmark_raw/celeb/masks", "Celeb landmark masks", missing)
    _require_dir(root / "landmark_raw/cfd/images", "CFD landmark images", missing)
    _require_dir(root / "landmark_raw/cfd/masks", "CFD landmark masks", missing)
    _validate_disease_manifest(root / "disease_byod/manifest.csv")

    if missing:
        print("Missing optional/required release inputs:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)

    print(f"Release input layout looks valid: {root}")


if __name__ == "__main__":
    main()
