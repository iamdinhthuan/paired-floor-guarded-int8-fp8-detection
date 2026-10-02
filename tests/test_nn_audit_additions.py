"""Guards for the 2026-10-02 audit additions: retrained KITTI holdout (phase N),
drive-clustered bootstrap (phase M), and the scripted max-calibration patch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "submission_support_20260911"
STATS = SUPPORT / "nn_final_stats.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_ok(directory: Path) -> None:
    for line in (directory / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        assert _sha(directory / rel.strip()) == digest, rel


def test_phase_n_holdout_is_disjoint_and_complete() -> None:
    d = SUPPORT / "phase_n_kitti_holdout_v2"
    _manifest_ok(d)
    assert len(list((d / "metrics").glob("kitti_final__*.json"))) == 84
    assert len(list((d / "metrics").glob("kitti_val__*.json"))) == 84
    final = {Path(l).name for l in (ROOT / "manifests/splits/kitti_confirmatory_v1/lists/"
                                    "kitti_confirmatory_v1_final.txt").read_text().split()}
    train = {Path(l).name for l in (ROOT / "manifests/splits/kitti_confirmatory_v1/lists/"
                                    "kitti_confirmatory_v1_train.txt").read_text().split()}
    assert not final & train
    calib = json.loads((d / "manifests/calibration/"
                        "kitti_confirmatory_train_512_s20260818_v1.json").read_text())
    names = {Path(str(x)).name for x in json.dumps(calib).split('"') if x.endswith(".png")}
    assert names and not names & final


def test_holdout_v2_reproduces_head_activation_pattern() -> None:
    c = json.loads(STATS.read_text())["kitti_holdout_v2"]["final_image"]["contrasts"]
    assert c["fp8-matched512_minus_int8-matched512"]["j95"]["ci95"][0] > 5.0
    assert c["int8-selective512_minus_int8-matched512"]["j95"]["ci95"][0] > 5.0
    lo, hi = c["int8-aonly512_minus_int8-matched512"]["j95"]["ci95"]
    assert lo < 0 < hi


def test_phase_m_cluster_bootstrap_is_complete() -> None:
    d = SUPPORT / "phase_m_cluster_bootstrap"
    _manifest_ok(d)
    cl = json.loads(STATS.read_text())["kitti_drive_cluster"]
    assert cl["n_drives"] == 108 and len(cl["rows"]) >= 30
    for row in cl["rows"]:
        assert abs(row["image"]["point"] - row["drive"]["point"]) < 1e-9


def test_maxreg_patch_reproduces_archived_graphs() -> None:
    rec = json.loads((SUPPORT / "phase_k_maxreg_results/patch_reproduction.json").read_text())
    for ds in ("kitti", "voc"):
        r = rec["results"][ds]
        assert r["nodes_identical"] and r["initializers_identical"] and r["archived_matches_registry"]
    assert (ROOT / "src/patch_maxreg_scales.py").is_file()


def test_manuscript_never_calls_seen_images_untouched() -> None:
    for name in ("main_nn.tex", "supplement.tex"):
        text = (ROOT / "paper" / name).read_text()
        assert "untouched" not in text
