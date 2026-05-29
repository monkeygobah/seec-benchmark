from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import yaml


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"


def add_release_code_to_path() -> None:
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))


def expand_env_config(path: str | Path) -> Path:
    os.environ.setdefault("EEB_RELEASE_ROOT", str(RELEASE_ROOT))
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    expanded_text = os.path.expandvars(raw_text)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=f".{path.name}",
        prefix="eeb_cfg_",
        delete=False,
    )
    with tmp:
        tmp.write(expanded_text)
    return Path(tmp.name)


def load_expanded_yaml(path: str | Path) -> tuple[dict, Path]:
    expanded_path = expand_env_config(path)
    with expanded_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, expanded_path


def write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
