from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"


def test_release_code_imports() -> None:
    sys.path.insert(0, str(CODE_ROOT))
    import disease_embeddings.config  # noqa: F401
    import embedding_extract.pipeline_config  # noqa: F401
    import landmark_probe.config  # noqa: F401
    import src.dataset_utils  # noqa: F401
