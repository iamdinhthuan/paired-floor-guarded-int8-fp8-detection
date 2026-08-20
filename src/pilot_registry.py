"""Canonical hashing helpers for frozen engine registries and pilot plans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(document: dict[str, Any], hash_key: str) -> str:
    payload = {key: value for key, value in document.items() if key != hash_key}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def calibration_sha256(path: str | Path) -> str:
    """Return a calibration list's validated provenance digest.

    Current calibration lists are self-hashed JSON documents.  Older pilot
    assets did not embed ``calibration_sha256`` and remain bound by their
    whole-file digest, so retain that behavior as a compatibility fallback.
    """
    calibration_path = Path(path)
    raw = calibration_path.read_bytes()
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(raw).hexdigest()
    if not isinstance(document, dict) or "calibration_sha256" not in document:
        return hashlib.sha256(raw).hexdigest()
    digest = document.get("calibration_sha256")
    if not isinstance(digest, str) or digest != canonical_hash(document, "calibration_sha256"):
        raise ValueError(f"calibration SHA-256 mismatch: {calibration_path}")
    return digest
