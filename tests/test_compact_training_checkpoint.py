from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "src" / "compact_training_checkpoint.py"
    assert path.is_file(), "training checkpoint compactor has not been implemented"
    spec = importlib.util.spec_from_file_location("compact_training_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_fixture(tmp_path: Path, *, valid_last_hash: bool = True) -> tuple[Path, Path, Path]:
    weights = tmp_path / "weights"
    weights.mkdir()
    best, last = weights / "best.pt", weights / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    registry = tmp_path / "training.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "best_weights": str(best),
                "best_weights_sha256": digest(best),
                "last_weights": str(last),
                "last_weights_sha256": digest(last) if valid_last_hash else "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry.with_suffix(".json.complete").write_text(digest(registry) + "\n", encoding="utf-8")
    return registry, best, last


def test_compactor_deletes_only_validated_last_and_retains_best(tmp_path: Path) -> None:
    module = load_module()
    registry, best, last = registry_fixture(tmp_path)
    output = tmp_path / "compaction.json"

    report = module.compact_completed_run(registry, output)

    assert best.is_file()
    assert not last.exists()
    assert report["deleted_last_weights_sha256"] == hashlib.sha256(b"last").hexdigest()
    assert report["retained_best_weights_sha256"] == hashlib.sha256(b"best").hexdigest()
    assert output.with_suffix(".json.complete").read_text().strip() == digest(output)


def test_compactor_refuses_hash_mismatch_without_deleting_anything(tmp_path: Path) -> None:
    module = load_module()
    registry, best, last = registry_fixture(tmp_path, valid_last_hash=False)

    with pytest.raises(SystemExit, match="last checkpoint hash mismatch"):
        module.compact_completed_run(registry, tmp_path / "compaction.json")

    assert best.is_file() and last.is_file()

