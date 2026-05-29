from __future__ import annotations

import argparse

from release_utils import add_release_code_to_path, expand_env_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    add_release_code_to_path()
    cfg_path = expand_env_config(args.cfg)

    from embedding_extract.aggregate_pipeline import aggregate_study
    from embedding_extract.analyze_pipeline import analyze_study
    from embedding_extract.extract_pipeline import extract_study
    from embedding_extract.pipeline_config import load_study_config

    cfg = load_study_config(cfg_path)
    if not args.skip_extract:
        written = extract_study(cfg, overwrite=args.overwrite)
        print(f"Wrote or reused {len(written)} embedding artifact(s).")
    metric_paths = analyze_study(cfg, overwrite=args.overwrite)
    print(f"Wrote or reused {len(metric_paths)} metric artifact(s).")
    summary = aggregate_study(cfg)
    print(f"Wrote summary table: {summary}")


if __name__ == "__main__":
    main()
