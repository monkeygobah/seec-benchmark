from __future__ import annotations

import argparse

from release_utils import add_release_code_to_path, expand_env_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-samples-per-dataset", type=int, default=None)
    args = ap.parse_args()

    add_release_code_to_path()
    cfg_path = expand_env_config(args.cfg)

    from landmark_probe.config import load_dataset_config
    from landmark_probe.prepare.pipeline import build_dataset

    cfg = load_dataset_config(cfg_path)
    manifest, landmarks, splits = build_dataset(
        cfg,
        overwrite=args.overwrite,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
    print(f"Wrote landmark manifest: {manifest}")
    print(f"Wrote landmarks: {landmarks}")
    print(f"Wrote splits: {splits}")


if __name__ == "__main__":
    main()
