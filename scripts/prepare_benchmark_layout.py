"""Prepare a separate input view for the shipped fixed benchmark manifests.

Default auto mode uses hard links, then copies on link failure (e.g. cross-volume).
Hard links normally require no administrator rights on Windows, Linux, or macOS.
Explicit symlink mode may require Windows Developer Mode/permissions. Linked views
must be treated as read-only: editing a hard link also edits its source. Use copy
mode for an independent view. This utility never writes source images or manifests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Iterable

RELEASE_ROOT = Path(__file__).resolve().parents[1]
FIXED_MANIFESTS = (
    ("s6", RELEASE_ROOT / "manifests/pretrain/pretrain_10k.txt"),
    ("s6", RELEASE_ROOT / "manifests/pretrain/pretrain_100k.txt"),
    ("s6", RELEASE_ROOT / "manifests/pretrain/pretrain_1m.txt"),
    ("s6", RELEASE_ROOT / "manifests/geometry/holdout.txt"),
    ("s7", RELEASE_ROOT / "manifests/geometry/open_hr.txt"),
)


@dataclass(frozen=True)
class LayoutSummary:
    entries: int
    unique_files: int
    linked: int
    copied: int


class LayoutValidationError(ValueError):
    pass


def map_entry(kind: str, entry: str) -> Path:
    path = PurePosixPath(entry)
    if not entry or "\\" in entry or ":" in entry or path.is_absolute() or any(
        part in {"", ".", ".."} for part in entry.split("/")
    ):
        raise LayoutValidationError(f"Unsafe manifest path: {entry!r}")
    if kind == "s6":
        if len(path.parts) != 1 or "__" not in entry or not entry.endswith("_224.jpg"):
            raise LayoutValidationError(f"Expected dataset__filename_224.jpg: {entry!r}")
        dataset, filename = entry.split("__", 1)
        if dataset in {"", ".", ".."} or not filename:
            raise LayoutValidationError(f"Invalid 224 entry: {entry!r}")
        return Path(dataset) / filename
    if kind == "s7":
        if len(path.parts) != 2 or not entry.endswith("_512.jpg"):
            raise LayoutValidationError(f"Expected dataset/filename_512.jpg: {entry!r}")
        return Path(*path.parts)
    raise LayoutValidationError(f"Unknown manifest kind: {kind}")


def prepare_layout(
    subset6_root: Path,
    subset7_root: Path,
    out_root: Path,
    mode: str = "auto",
    *,
    manifests: Iterable[tuple[str, Path]] = FIXED_MANIFESTS,
) -> LayoutSummary:
    """Preflight the complete plan, then create files exclusively without overwrites.

    manifests is injectable for tiny unit fixtures; the CLI always uses fixed splits.
    Validation failure makes no output changes. An I/O failure during creation may
    leave a partial view; it is reported, and existing destinations are never replaced.
    """
    if mode not in {"auto", "hardlink", "symlink", "copy"}:
        raise ValueError(f"Unknown mode: {mode}")
    roots = {"s6": Path(subset6_root).resolve(), "s7": Path(subset7_root).resolve()}
    out = Path(out_root).resolve()
    for source_root in roots.values():
        if not source_root.is_dir():
            raise LayoutValidationError(f"Missing source directory: {source_root}")
        if out.is_relative_to(source_root) or source_root.is_relative_to(out):
            raise LayoutValidationError("Output and source roots must not overlap")
    if out.exists() and not out.is_dir():
        raise LayoutValidationError(f"Output root is not a directory: {out}")

    plan: dict[str, Path] = {}
    destination_keys: dict[str, str] = {}
    entries = 0
    missing = collisions = 0
    examples: list[str] = []
    for kind, manifest in manifests:
        with Path(manifest).open(encoding="utf-8") as handle:
            for line in handle:
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                entries += 1
                relative_source = map_entry(kind, entry)
                source = roots[kind] / relative_source
                if entry in plan:
                    if plan[entry] != source:
                        raise LayoutValidationError(f"Conflicting manifest destination: {entry}")
                    continue  # Nested/overlapping pretraining sets share one view file.
                destination = out / entry
                destination_key = os.path.normcase(str(destination))
                if destination_key in destination_keys:
                    raise LayoutValidationError(
                        f"Conflicting manifest destinations on this platform: "
                        f"{destination_keys[destination_key]} and {entry}"
                    )
                destination_keys[destination_key] = entry
                plan[entry] = source
                if not destination.resolve().is_relative_to(out):
                    raise LayoutValidationError(f"Destination escapes output root: {entry}")
                if not source.is_file():
                    missing += 1
                    if len(examples) < 10:
                        examples.append(f"Missing source: {source}")
                if os.path.lexists(destination):
                    collisions += 1
                    if len(examples) < 10:
                        examples.append(f"Destination collision: {destination}")
                parent = destination.parent
                while parent != out:
                    if parent.exists() and not parent.is_dir():
                        collisions += 1
                        if len(examples) < 10:
                            examples.append(f"Destination parent is not a directory: {parent}")
                        break
                    parent = parent.parent
    if missing or collisions:
        raise LayoutValidationError(
            f"Preflight failed: {entries} manifest entries, {len(plan)} unique files, "
            f"{missing} missing sources, {collisions} collisions. No files created.\n"
            + "\n".join(examples)
        )

    linked = copied = 0
    for entry, source in plan.items():
        destination = out / entry
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Check parents again before creation, and use exclusive creation operations.
        if not destination.resolve().is_relative_to(out):
            raise LayoutValidationError(f"Destination escapes output root: {entry}")
        if mode in {"auto", "hardlink"}:
            try:
                os.link(source, destination)
                linked += 1
                continue
            except FileExistsError:
                raise  # Never treat a collision as a reason to overwrite/copy.
            except OSError:
                if mode == "hardlink":
                    raise
        elif mode == "symlink":
            destination.symlink_to(source)
            linked += 1
            continue
        # Opening in xb mode also protects against a destination created after preflight.
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        copied += 1
    return LayoutSummary(entries, len(plan), linked, copied)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset6-root", required=True, type=Path)
    parser.add_argument("--subset7-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Separate view root; use EEB_DATA_ROOT/subset6 for existing configs")
    parser.add_argument("--mode", choices=("auto", "hardlink", "symlink", "copy"), default="auto")
    args = parser.parse_args()
    try:
        result = prepare_layout(args.subset6_root, args.subset7_root, args.out_root, args.mode)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Layout preparation failed: {error}\n")
    print(f"Prepared {result.unique_files} unique files from {result.entries} manifest entries: "
          f"{result.linked} linked, {result.copied} copied; 0 missing, 0 collisions.")
    print(f"Benchmark input root: {args.out_root.resolve()}")
    print("Treat linked views as read-only; use --mode copy for independent files.")


if __name__ == "__main__":
    main()
