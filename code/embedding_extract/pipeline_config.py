from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path("/workspace")


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class StudySpec:
    name: str
    output_root: Path


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    run_dir: Path
    checkpoint_step: int
    checkpoint_path: Path | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    root: Path
    split_label: str
    image_size: int
    normalize_imagenet: bool = True
    manifest: Path | None = None


@dataclass(frozen=True)
class ExtractionSpec:
    batch_size: int
    num_workers: int
    device: str
    precision: str
    overwrite: bool = False


@dataclass(frozen=True)
class ArtifactSpec:
    save_backbone: bool = True
    save_projected: bool = True


@dataclass(frozen=True)
class IsotropyAnalysisSpec:
    enabled: bool = True
    num_pairs: int = 200_000
    seed: int = 0
    embedding_keys: tuple[str, ...] = ("emb", "proj")


@dataclass(frozen=True)
class AggregationSpec:
    summary_csv: str = "isotropy_summary.csv"
    row_keys: tuple[str, ...] = (
        "run_name",
        "checkpoint_step",
        "dataset_name",
        "split_label",
        "embedding_key",
    )


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    study: StudySpec
    runs: tuple[RunSpec, ...]
    datasets: tuple[DatasetSpec, ...]
    extraction: ExtractionSpec
    artifact: ArtifactSpec
    isotropy: IsotropyAnalysisSpec
    aggregation: AggregationSpec

    @property
    def outputs_root(self) -> Path:
        return self.study.output_root

    @property
    def embeddings_dir(self) -> Path:
        return self.outputs_root / "embeddings" / self.study.name

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_root / "metrics" / self.study.name

    @property
    def tables_dir(self) -> Path:
        return self.outputs_root / "tables" / self.study.name


def _resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        try:
            workspace_relative = path.relative_to(WORKSPACE_ROOT)
        except ValueError:
            return path
        return (PROJECT_ROOT / workspace_relative).resolve()
    return (base_dir / path).resolve()


def _require(value: Any, key: str) -> Any:
    if value is None:
        raise ValueError(f"Missing required config value: {key}")
    return value


def _parse_run(base_dir: Path, raw: dict[str, Any]) -> RunSpec:
    run_dir = _resolve_path(base_dir, _require(raw.get("run_dir"), "runs[].run_dir"))
    checkpoint_step = int(_require(raw.get("checkpoint_step"), "runs[].checkpoint_step"))
    checkpoint_path = _resolve_path(base_dir, raw.get("checkpoint_path"))
    run_name = raw.get("run_name") or run_dir.name
    overrides = dict(raw.get("config_overrides", {}))
    return RunSpec(
        run_name=str(run_name),
        run_dir=run_dir,
        checkpoint_step=checkpoint_step,
        checkpoint_path=checkpoint_path,
        config_overrides=overrides,
    )


def _parse_dataset(base_dir: Path, raw: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        dataset_name=str(_require(raw.get("dataset_name"), "datasets[].dataset_name")),
        root=_resolve_path(base_dir, _require(raw.get("root"), "datasets[].root")),
        split_label=str(_require(raw.get("split_label"), "datasets[].split_label")),
        image_size=int(raw.get("image_size", 224)),
        normalize_imagenet=bool(raw.get("normalize_imagenet", True)),
        manifest=_resolve_path(base_dir, raw.get("manifest")),
    )


def load_study_config(path: str | Path) -> StudyConfig:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_dir = cfg_path.parent

    study_raw = raw.get("study", {})
    study_name = str(_require(study_raw.get("name"), "study.name"))
    output_root = _resolve_path(base_dir, study_raw.get("output_root")) or (
        base_dir.parent / "outputs"
    )

    runs_raw = raw.get("runs")
    if not runs_raw:
        raise ValueError("Study config must define at least one run.")
    runs = tuple(_parse_run(base_dir, item) for item in runs_raw)

    datasets_raw = raw.get("datasets")
    if not datasets_raw:
        raise ValueError("Study config must define at least one dataset.")
    datasets = tuple(_parse_dataset(base_dir, item) for item in datasets_raw)

    extraction_raw = raw.get("extraction", {})
    extraction = ExtractionSpec(
        batch_size=int(extraction_raw.get("batch_size", 256)),
        num_workers=int(extraction_raw.get("num_workers", 4)),
        device=str(extraction_raw.get("device", "auto")),
        precision=str(extraction_raw.get("precision", "fp32")).lower(),
        overwrite=bool(extraction_raw.get("overwrite", False)),
    )

    artifact_raw = raw.get("artifact", {})
    artifact = ArtifactSpec(
        save_backbone=bool(artifact_raw.get("save_backbone", True)),
        save_projected=bool(artifact_raw.get("save_projected", True)),
    )
    if not artifact.save_backbone and not artifact.save_projected:
        raise ValueError("artifact must enable at least one of save_backbone/save_projected")

    isotropy_raw = raw.get("analyses", {}).get("isotropy", {})
    isotropy = IsotropyAnalysisSpec(
        enabled=bool(isotropy_raw.get("enabled", True)),
        num_pairs=int(isotropy_raw.get("num_pairs", 200_000)),
        seed=int(isotropy_raw.get("seed", 0)),
        embedding_keys=tuple(isotropy_raw.get("embedding_keys", ["emb", "proj"])),
    )

    aggregation_raw = raw.get("aggregation", {})
    aggregation = AggregationSpec(
        summary_csv=str(aggregation_raw.get("summary_csv", "isotropy_summary.csv")),
        row_keys=tuple(
            aggregation_raw.get(
                "row_keys",
                ["run_name", "checkpoint_step", "dataset_name", "split_label", "embedding_key"],
            )
        ),
    )

    cfg = StudyConfig(
        path=cfg_path,
        study=StudySpec(name=study_name, output_root=output_root),
        runs=runs,
        datasets=datasets,
        extraction=extraction,
        artifact=artifact,
        isotropy=isotropy,
        aggregation=aggregation,
    )
    validate_study_config(cfg)
    return cfg


def merge_training_config(base_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return _deep_update(base_cfg, overrides)


def validate_study_config(cfg: StudyConfig) -> None:
    seen_targets: set[tuple[str, int, str, str]] = set()

    for run in cfg.runs:
        if not run.run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run.run_dir}")
        if not (run.run_dir / "config.yaml").exists():
            raise FileNotFoundError(f"Run config missing at: {run.run_dir / 'config.yaml'}")

    for dataset in cfg.datasets:
        if not dataset.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset.root}")
        if dataset.manifest is not None and not dataset.manifest.exists():
            raise FileNotFoundError(f"Dataset manifest does not exist: {dataset.manifest}")

    for run in cfg.runs:
        for dataset in cfg.datasets:
            key = (run.run_name, run.checkpoint_step, dataset.dataset_name, dataset.split_label)
            if key in seen_targets:
                raise ValueError(f"Duplicate run+dataset target in study config: {key}")
            seen_targets.add(key)
