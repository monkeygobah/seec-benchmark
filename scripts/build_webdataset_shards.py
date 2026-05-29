from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from release_utils import CODE_ROOT, add_release_code_to_path


def main() -> None:
    add_release_code_to_path()
    script_path = CODE_ROOT / "build_webdataset_shards.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "build_webdataset_shards.py was not copied into the release code directory"
        )
    sys.argv = [str(script_path), *sys.argv[1:]]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
