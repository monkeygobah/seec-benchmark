from __future__ import annotations

import argparse
from dataclasses import replace

from release_utils import add_release_code_to_path, expand_env_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--overwrite-extract", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    add_release_code_to_path()
    cfg_path = expand_env_config(args.cfg)

    from landmark_probe.aggregate.pipeline import aggregate_study
    from landmark_probe.config import load_dataset_config, load_probe_config, load_study_config
    from landmark_probe.extract.pipeline import extract_study
    from landmark_probe.probe.pipeline import run_probe_study

    study_cfg = load_study_config(cfg_path)
    if args.overwrite_extract:
        study_cfg = replace(study_cfg, extraction=replace(study_cfg.extraction, overwrite=True))
    dataset_cfg = load_dataset_config(study_cfg.dataset_cfg_path)
    probe_cfg = load_probe_config(study_cfg.probe_cfg_path)

    if not args.skip_extract:
        written = extract_study(study_cfg, dataset_cfg)
        print(f"Wrote or reused {len(written)} landmark embedding artifact(s).")
    if not args.skip_probe:
        probe_dirs = run_probe_study(study_cfg, dataset_cfg, probe_cfg)
        print(f"Wrote or reused {len(probe_dirs)} probe run directory(s).")
    overall, per_landmark = aggregate_study(study_cfg)
    print(f"Wrote summaries:\n- {overall}\n- {per_landmark}")


if __name__ == "__main__":
    main()
