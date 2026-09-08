#!/usr/bin/env python3
"""Explicit entry point for the bounded-parallel shared-mask V3 attempt."""
from __future__ import annotations

import os


EXPECTED_ATTEMPT = "shared_mask_pilot_v3"
observed = os.environ.get("SHARED_MASK_PILOT_ATTEMPT")
if observed not in (None, EXPECTED_ATTEMPT):
    raise SystemExit(
        f"SHARED-MASK V3 REFUSED: conflicting SHARED_MASK_PILOT_ATTEMPT={observed}"
    )
os.environ["SHARED_MASK_PILOT_ATTEMPT"] = EXPECTED_ATTEMPT

from run_shared_mask_pilot import main  # noqa: E402  (attempt must be frozen first)


if __name__ == "__main__":
    main()
