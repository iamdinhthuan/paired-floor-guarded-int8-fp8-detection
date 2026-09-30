#!/usr/bin/env python3
"""Multiplicity audit for the paired image bootstrap.

Reviewer-facing check: ``paired_bootstrap.accumulate_ap`` resamples evalImgs
by *position* (``image_positions`` indexes the evaluated-image list, not the
COCO ``imgIds`` namespace), so an image drawn twice must contribute its
detections and ground truth twice. These tests build a stub evaluation object
with synthetic per-image entries and verify that behaviour directly, without
requiring pycocotools.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paired_bootstrap import accumulate_ap  # noqa: E402


def _entry(dt_scores, dt_matches, gt_ignore):
    n_iou = 1
    return {
        "dtScores": np.asarray(dt_scores, dtype=float),
        "dtMatches": np.asarray(dt_matches, dtype=float).reshape(n_iou, -1),
        "dtIgnore": np.zeros((n_iou, len(dt_scores))),
        "gtIgnore": np.asarray(gt_ignore, dtype=float),
    }


def _stub_eval(entries, n_images):
    params = SimpleNamespace(
        iouThrs=np.array([0.5]),
        recThrs=np.linspace(0, 1, 101),
        catIds=[0],
        areaRng=[[0.0, 1e10]],
        areaRngLbl=["all"],
    )
    return SimpleNamespace(
        params=params,
        _paramsEval=SimpleNamespace(imgIds=list(range(n_images))),
        evalImgs=entries,
    )


def test_duplicate_positions_contribute_repeatedly():
    # image 0: one unignored GT and one unmatched (false-positive) detection.
    # image 1: one unignored GT and one matched detection at a higher score.
    img0 = _entry([0.8], [[0.0]], [0.0])
    img1 = _entry([0.9], [[1.0]], [0.0])
    ev = _stub_eval([img0, img1], n_images=2)

    ap_distinct = accumulate_ap(ev, [0, 1])[0]
    ap_dup = accumulate_ap(ev, [0, 0, 1])[0]

    # With 2 positives but only 1 true positive, recall saturates at 0.5:
    # the 51 recall thresholds <= 0.5 sit at precision 1.0, the rest at 0.
    expected_distinct = 51.0 / 101.0
    assert abs(ap_distinct - expected_distinct) < 1e-9
    # If duplicates were deduplicated to imgIds, [0,0,1] would equal [0,1].
    # With multiplicity, [0,0,1] has 3 positives (img0 counted twice) and the
    # single covered GT caps recall at 1/3: the 34 thresholds <= 1/3 sit at
    # precision 1.0 and the rest score zero. The two paths differ, proving a
    # resampled image contributes its detections and GT per occurrence.
    expected_dup = 34.0 / 101.0
    assert abs(ap_dup - expected_dup) < 1e-9


def test_singleton_resample_uses_that_image_only():
    img0 = _entry([0.9], [[1.0]], [0.0])
    img1 = _entry([0.9], [[0.0]], [0.0])  # GT with no matching detection
    ev = _stub_eval([img0, img1], n_images=2)
    assert abs(accumulate_ap(ev, [0])[0] - 1.0) < 1e-12
    assert accumulate_ap(ev, [1])[0] == 0.0


if __name__ == "__main__":
    test_duplicate_positions_contribute_repeatedly()
    test_singleton_resample_uses_that_image_only()
    print("multiplicity checks passed")
