#!/usr/bin/env python3
"""Paired contrasts with plug-in points and common-image bootstrap intervals.

Point estimates are full-sample plug-in AP (``cell_points.json``, AP points);
intervals are 2.5/97.5 percentiles of the paired per-cell bootstrap draws
(``*__draws.npz``, AP fractions). Two-sided bootstrap p-values use
p = min(1, 2 * min(P(d <= 0), P(d >= 0))).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CORRUPTIONS = ("fog", "gaussian_noise", "jpeg", "motion_blur")
SEVERITIES = (1, 3, 5)
FAMILIES = {
    "corr12": [(c, s) for c in CORRUPTIONS for s in SEVERITIES],
    "in_family": [(c, s) for c in ("gaussian_noise", "jpeg") for s in SEVERITIES],
    "held_out": [(c, s) for c in ("fog", "motion_blur") for s in SEVERITIES],
    "held_out_ex_mb5": [(c, s) for c in ("fog", "motion_blur") for s in SEVERITIES
                        if (c, s) != ("motion_blur", 5)],
    "corr11_ex_mb5": [(c, s) for c in CORRUPTIONS for s in SEVERITIES
                      if (c, s) != ("motion_blur", 5)],
    "fam_fog": [("fog", s) for s in SEVERITIES],
    "fam_gaussian_noise": [("gaussian_noise", s) for s in SEVERITIES],
    "fam_jpeg": [("jpeg", s) for s in SEVERITIES],
    "fam_motion_blur": [("motion_blur", s) for s in SEVERITIES],
}


class _Merged:
    """Union of several draw caches built on one resample schedule."""

    def __init__(self, paths: list[Path]):
        self._d: dict[str, np.ndarray] = {}
        for path in paths:
            z = np.load(path)
            for key in z.files:
                if key in self._d and key not in ("n_boot", "samples_seed"):
                    if not np.array_equal(self._d[key], z[key], equal_nan=True):
                        raise SystemExit(f"unpaired draws for {key} in {path}")
                self._d.setdefault(key, z[key])
        self.files = list(self._d)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._d[key]


class Block:
    def __init__(self, draws, points: dict):
        self.z = _Merged([draws] if isinstance(draws, Path) else list(draws))
        self.p = points

    def _draw(self, arm: str, cond: str) -> np.ndarray:
        return self.z[f"{arm}__cell__{cond}"] * 100.0

    def level(self, arm: str, family: str) -> tuple[float, np.ndarray]:
        """Plug-in point and draws of an arm-level summary."""
        if family == "j95":
            conds = ["codec-control-s0"]
        elif family == "clean":
            conds = ["clean-s0"]
        else:
            conds = [f"{c}-s{s}" for c, s in FAMILIES[family]]
        point = float(np.mean([self.p[arm][c] for c in conds]))
        draws = np.mean([self._draw(arm, c) for c in conds], axis=0)
        return point, draws

    def has(self, arm: str, family: str) -> bool:
        cond = {"j95": "codec-control-s0", "clean": "clean-s0"}.get(family, "fog-s1")
        return arm in self.p and cond in self.p[arm] and f"{arm}__cell__{cond}" in self.z.files

    def diff(self, a: str, b: str, family: str) -> dict:
        pa, da = self.level(a, family)
        pb, db = self.level(b, family)
        return summarize(pa - pb, da - db)

    def delta_e(self, a: str, b: str, family: str = "corr12", base: str = "j95") -> dict:
        pa_c, da_c = self.level(a, family)
        pb_c, db_c = self.level(b, family)
        pa_0, da_0 = self.level(a, base)
        pb_0, db_0 = self.level(b, base)
        return summarize((pa_c - pb_c) - (pa_0 - pb_0), (da_c - db_c) - (da_0 - db_0))


def summarize(point: float, draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
    return {"point": float(point), "ci95": [float(lo), float(hi)],
            "se": float(np.std(draws, ddof=1)), "p": float(p)}


def load_block(phase_dir: Path, dataset: str, model: str,
               extra_dirs: tuple[Path, ...] = ()) -> Block:
    """Load one block; ``extra_dirs`` add arms from other caches on the same schedule."""
    points: dict = {}
    draws = []
    for d in (phase_dir, *extra_dirs):
        npz = d / "bootstrap" / f"{dataset}__{model}__draws.npz"
        if not npz.is_file():
            continue
        for arm, cells in json.loads((d / "cell_points.json").read_text()).get(
                f"{dataset}/{model}", {}).items():
            points.setdefault(arm, {}).update(cells)
        draws.append(npz)
    return Block(draws, points)
