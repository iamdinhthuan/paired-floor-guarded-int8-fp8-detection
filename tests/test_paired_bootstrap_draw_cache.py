from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import paired_bootstrap


def test_draw_cache_round_trip_binds_exact_arrays(tmp_path: Path) -> None:
    excess = np.arange(24, dtype=np.float64).reshape(6, 4) / 100
    psi = excess[:, 1] - excess[:, 3]
    path = tmp_path / "cell.draws.npz"

    record = paired_bootstrap.write_draw_cache(
        path, excess=excess, psi=psi, n_boot=6, seed=17
    )

    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["excess_shape"] == [6, 4]
    with np.load(path, allow_pickle=False) as cache:
        assert np.array_equal(cache["excess"], excess)
        assert np.array_equal(cache["psi"], psi)
        assert int(cache["n_boot"]) == 6
        assert int(cache["seed"]) == 17
