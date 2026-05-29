from __future__ import annotations

import argparse
import runpy
import sys

from release_utils import CODE_ROOT, add_release_code_to_path, expand_env_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args, rest = ap.parse_known_args()

    add_release_code_to_path()
    cfg_path = expand_env_config(args.cfg)
    sys.argv = [str(CODE_ROOT / "train_ssl.py"), "--cfg", str(cfg_path), *rest]
    runpy.run_path(str(CODE_ROOT / "train_ssl.py"), run_name="__main__")


if __name__ == "__main__":
    main()
