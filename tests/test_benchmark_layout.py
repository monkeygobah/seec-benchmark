from __future__ import annotations

import errno
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_benchmark_layout as bridge
import prepare_subset6_splits as sampler


@pytest.fixture
def fixture_layout(tmp_path):
    s6, s7, out = (tmp_path / name for name in ("SUBSET_6", "SUBSET_7", "view"))
    (s6 / "celeb").mkdir(parents=True)
    (s7 / "cfd").mkdir(parents=True)
    (s6 / "celeb/001049_rfc_OS_224.jpg").write_bytes(b"224 fixture")
    (s7 / "cfd/CFD-AF-200-228-N_rfc_OD_512.jpg").write_bytes(b"512 fixture")
    flat = "celeb__001049_rfc_OS_224.jpg"
    hr = "cfd/CFD-AF-200-228-N_rfc_OD_512.jpg"
    a, b = tmp_path / "pretrain.txt", tmp_path / "hr.txt"
    a.write_text(flat + "\n" + flat + "\n", encoding="utf-8")
    b.write_text(hr + "\n", encoding="utf-8")
    return s6, s7, out, (("s6", a), ("s7", b)), flat, hr


def test_copy_mapping_and_dedup_preserve_sources_and_manifests(fixture_layout):
    s6, s7, out, manifests, flat, hr = fixture_layout
    before = {p: p.read_bytes() for p in (s6 / "celeb/001049_rfc_OS_224.jpg",
              s7 / hr, manifests[0][1], manifests[1][1])}
    summary = bridge.prepare_layout(s6, s7, out, "copy", manifests=manifests)
    assert summary == bridge.LayoutSummary(entries=3, unique_files=2, linked=0, copied=2)
    assert (out / flat).read_bytes() == b"224 fixture"
    assert (out / hr).read_bytes() == b"512 fixture"
    assert all(p.read_bytes() == value for p, value in before.items())


def test_missing_source_preflight_creates_nothing(fixture_layout):
    s6, s7, out, manifests, flat, hr = fixture_layout
    (s7 / hr).unlink()
    with pytest.raises(bridge.LayoutValidationError, match="1 missing sources") as failure:
        bridge.prepare_layout(s6, s7, out, manifests=manifests)
    assert str(s7 / hr) in str(failure.value)
    assert not out.exists()


def test_collision_preserves_existing_destination(fixture_layout):
    s6, s7, out, manifests, flat, hr = fixture_layout
    out.mkdir()
    (out / flat).write_bytes(b"do not overwrite")
    with pytest.raises(bridge.LayoutValidationError, match="1 collisions"):
        bridge.prepare_layout(s6, s7, out, manifests=manifests)
    assert (out / flat).read_bytes() == b"do not overwrite"
    assert not (out / hr).exists()


def test_auto_links_without_admin(fixture_layout):
    s6, s7, out, manifests, flat, hr = fixture_layout
    summary = bridge.prepare_layout(s6, s7, out, "hardlink", manifests=manifests)
    assert summary.linked == 2 and summary.copied == 0
    assert bridge.os.path.samefile(out / flat, s6 / "celeb/001049_rfc_OS_224.jpg")
    assert (out / flat).read_bytes() == b"224 fixture"
    assert (out / hr).read_bytes() == b"512 fixture"


def test_auto_copy_fallback(fixture_layout, monkeypatch):
    s6, s7, out, manifests, flat, hr = fixture_layout
    def unavailable(*args):
        raise OSError(errno.EXDEV, "cross-volume link")
    monkeypatch.setattr(bridge.os, "link", unavailable)
    summary = bridge.prepare_layout(s6, s7, out, manifests=manifests)
    assert summary.linked == 0 and summary.copied == 2
    assert (out / flat).read_bytes() == b"224 fixture"


def test_link_collision_after_preflight_never_falls_back(fixture_layout, monkeypatch):
    s6, s7, out, manifests, flat, hr = fixture_layout
    def raced(source, destination):
        Path(destination).write_bytes(b"another process")
        raise FileExistsError("destination appeared")
    monkeypatch.setattr(bridge.os, "link", raced)
    with pytest.raises(FileExistsError):
        bridge.prepare_layout(s6, s7, out, manifests=manifests)
    assert (out / flat).read_bytes() == b"another process"


@pytest.mark.parametrize("kind, entry", [("s6", "../escape_224.jpg"),
    ("s6", "..__escape_224.jpg"), ("s7", "cfd/../escape_512.jpg"),
    ("s7", "C:/escape_512.jpg")])
def test_unsafe_mapping_is_rejected(kind, entry):
    with pytest.raises(bridge.LayoutValidationError):
        bridge.map_entry(kind, entry)


def test_source_output_overlap_is_rejected(fixture_layout):
    s6, s7, out, manifests, flat, hr = fixture_layout
    with pytest.raises(bridge.LayoutValidationError, match="must not overlap"):
        bridge.prepare_layout(s6, s7, s6 / "view", manifests=manifests)


def test_development_sampler_refuses_canonical_output():
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/prepare_subset6_splits.py"),
        "--subset6-root", "not-needed", "--out-dir", str(ROOT / "manifests")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to write inside canonical manifests" in result.stderr


def test_development_sampler_cannot_overwrite(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sampler.write_manifest(["replacement"], target)
    assert target.read_text(encoding="utf-8") == "original"


def test_platform_normalized_collision_is_detected(fixture_layout, monkeypatch):
    s6, s7, out, manifests, flat, hr = fixture_layout
    # Simulate case-insensitive destination semantics without requiring a specific OS.
    monkeypatch.setattr(bridge.os.path, "normcase", lambda value: str(value).lower())
    with manifests[0][1].open("a", encoding="utf-8") as handle:
        handle.write("CELEB__001049_rfc_OS_224.jpg\n")
    with pytest.raises(bridge.LayoutValidationError, match="Conflicting manifest destinations"):
        bridge.prepare_layout(s6, s7, out, manifests=manifests)
    assert not out.exists()
