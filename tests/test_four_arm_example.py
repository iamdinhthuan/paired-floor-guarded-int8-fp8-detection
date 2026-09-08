import hashlib
import json
from pathlib import Path

import pytest

from reproduce_four_arm_example import validate_example_files


def test_reproduction_refuses_changed_input_before_evaluation(tmp_path):
    data = tmp_path / "prediction.json"
    data.write_bytes(b"[]")
    manifest = {"files": {"prediction.json": hashlib.sha256(b"[]").hexdigest()}}
    validate_example_files(tmp_path, manifest)
    data.write_bytes(b"[1]")
    with pytest.raises(ValueError, match="hash"):
        validate_example_files(tmp_path, manifest)


def test_reproduction_refuses_paths_outside_example(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        validate_example_files(tmp_path, {"files": {"../external.json": "0" * 64}})
