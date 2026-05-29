from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_manifests_are_nonempty_text_files() -> None:
    manifest_paths = [
        ROOT / "manifests/pretrain/pretrain_10k.txt",
        ROOT / "manifests/pretrain/pretrain_100k.txt",
        ROOT / "manifests/pretrain/pretrain_1m.txt",
        ROOT / "manifests/geometry/holdout.txt",
        ROOT / "manifests/geometry/open_hr.txt",
    ]
    for path in manifest_paths:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines, path
        assert all(not line.startswith("/") for line in lines[:100]), path


def test_disease_byod_example_schema() -> None:
    path = ROOT / "manifests/examples/disease_manifest_example.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    assert {"image_path", "label", "group_id"} <= columns
