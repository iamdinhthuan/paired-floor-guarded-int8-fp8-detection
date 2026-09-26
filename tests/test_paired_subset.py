"""Tests for the frozen paired-subset selection record."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPT = SRC / "build_paired_subset.py"


def _sha(document: dict) -> str:
    payload = {k: v for k, v in document.items() if k != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parent(tmp_path: Path, ids: list[int]) -> Path:
    records = [
        {
            "dataset": "coco", "image_id": i, "source_relpath": f"images/{i}.jpg",
            "source_sha256": "0" * 64, "corruption": "clean", "severity": 0,
            "seed": 1, "output_relpath": f"clean/{i}.jpg", "sha256": "1" * 64,
            "generator": "g", "generator_version": "1",
        }
        for i in ids
    ]
    doc = {"schema_version": 1, "dataset": "coco", "split": "val2017",
           "expected_image_ids": ids, "records": records}
    doc["manifest_sha256"] = _sha(doc)
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(doc))
    Path(str(path) + ".complete").write_text(doc["manifest_sha256"] + "\n")
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )


def test_subset_is_deterministic_and_bounded(tmp_path):
    parent = _parent(tmp_path, list(range(100)))
    out1, out2 = tmp_path / "s1.json", tmp_path / "s2.json"
    base = ["--parent-manifest", str(parent), "--count", "20", "--seed", "7", "--attempt", "t"]
    assert _run(*base, "--out", str(out1)).returncode == 0
    assert _run(*base, "--out", str(out2)).returncode == 0
    d1, d2 = json.loads(out1.read_text()), json.loads(out2.read_text())
    assert d1["image_ids"] == d2["image_ids"]
    assert len(d1["image_ids"]) == 20 and d1["image_ids"] == sorted(d1["image_ids"])
    assert set(d1["image_ids"]) <= set(range(100))
    assert d1["parent_manifest_sha256"] == _sha(json.loads(parent.read_text()))
    marker = Path(str(out1) + ".complete").read_text().strip()
    assert marker == d1["selection_sha256"]


def test_subset_seed_changes_selection(tmp_path):
    parent = _parent(tmp_path, list(range(100)))
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    base = ["--parent-manifest", str(parent), "--count", "20", "--attempt", "t"]
    assert _run(*base, "--seed", "1", "--out", str(out1)).returncode == 0
    assert _run(*base, "--seed", "2", "--out", str(out2)).returncode == 0
    assert json.loads(out1.read_text())["image_ids"] != json.loads(out2.read_text())["image_ids"]


def test_subset_refuses_overwrite_and_bad_parent(tmp_path):
    parent = _parent(tmp_path, list(range(50)))
    out = tmp_path / "s.json"
    base = ["--parent-manifest", str(parent), "--count", "10", "--seed", "1", "--attempt", "t"]
    assert _run(*base, "--out", str(out)).returncode == 0
    assert _run(*base, "--out", str(out)).returncode != 0
    Path(str(parent) + ".complete").write_text("bogus\n")
    out2 = tmp_path / "s2.json"
    result = _run(*base, "--out", str(out2))
    assert result.returncode != 0 and "marker" in result.stderr.lower()


def test_subset_count_bounds(tmp_path):
    parent = _parent(tmp_path, list(range(10)))
    result = _run("--parent-manifest", str(parent), "--count", "11", "--seed", "1",
                  "--attempt", "t", "--out", str(tmp_path / "x.json"))
    assert result.returncode != 0
