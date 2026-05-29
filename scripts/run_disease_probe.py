from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from release_utils import add_release_code_to_path, load_expanded_yaml, write_yaml


def _column_or_default(df: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column in df.columns:
        return df[column].astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)


def adapt_byod_manifest(cfg: dict) -> None:
    dataset = cfg["dataset"]
    manifest_path = Path(dataset["manifest_csv"])
    df = pd.read_csv(manifest_path)
    if {"output_path", "folder_label", "source_image_path"}.issubset(df.columns):
        return

    required = {"image_path", "label", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BYOD disease manifest missing required columns: {sorted(missing)}")

    adapted = pd.DataFrame()
    adapted["output_path"] = df["image_path"].astype(str)
    adapted["source_image_path"] = df.get("source_image_path", df["group_id"]).astype(str)
    adapted["folder_label"] = df["label"].astype(str)
    adapted["filename"] = df["image_path"].map(lambda p: Path(str(p)).name)
    adapted["eye"] = _column_or_default(df, "eye", "")
    adapted["disease_status"] = _column_or_default(df, "disease_status", "disease")
    adapted["source_mode"] = "byod"
    adapted["group_id"] = df["group_id"].astype(str)
    if "patient_id" in df.columns:
        adapted["patient_id"] = df["patient_id"].astype(str)

    out_root = Path(cfg["study"]["output_root"])
    out_path = out_root / "_release_adapted_disease_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_csv(out_path, index=False)
    dataset["manifest_csv"] = str(out_path)
    cfg.setdefault("supervised", {})
    cfg["supervised"]["label_column"] = "folder_label"
    cfg["supervised"]["group_column"] = "group_id"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-linear", action="store_true")
    ap.add_argument("--skip-finetune", action="store_true")
    args = ap.parse_args()

    add_release_code_to_path()
    cfg_data, expanded_cfg_path = load_expanded_yaml(args.cfg)
    adapt_byod_manifest(cfg_data)
    adapted_cfg_path = expanded_cfg_path.with_name(expanded_cfg_path.stem + "_adapted.yaml")
    write_yaml(cfg_data, adapted_cfg_path)

    from disease_embeddings.config import load_study_config
    from disease_embeddings.extract import extract_study
    from disease_embeddings.summarize import summarize_finetune, summarize_linear_probe
    from disease_embeddings.supervised import run_finetune_study, run_linear_probe_study

    cfg = load_study_config(adapted_cfg_path)
    if not args.skip_extract:
        written = extract_study(cfg, overwrite=args.overwrite)
        print(f"Wrote or reused {len(written)} disease embedding artifact(s).")
    if not args.skip_linear:
        linear = run_linear_probe_study(cfg)
        print(f"Wrote {len(linear)} linear probe run(s).")
        summarize_linear_probe(cfg)
    if not args.skip_finetune:
        finetuned = run_finetune_study(cfg, method="pca")
        print(f"Wrote {len(finetuned)} fine-tune run(s).")
        summarize_finetune(cfg)


if __name__ == "__main__":
    main()
