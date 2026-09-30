"""Integration guards for the deployment-side audits (B3/B4).

These guard the ledger files that feed nn_final_stats["latency"] and
["rebuild_variance"], plus the rendered macros the manuscript consumes.
When the ledgers are absent the tests skip; when present they assert the
invariants the prose claims (all-distinct rebuild hashes, idle-gated reps,
macro parity with the summary JSON).
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "submission_support_20260911"
STATS = SUPPORT / "nn_final_stats.json"
LAT_DIR = SUPPORT / "nn_latency_v1_20260930"
RB_DIR = SUPPORT / "nn_rebuild_variance_v1_20260930"
NUMBERS = ROOT / "paper" / "generated" / "nn_numbers.tex"


def _final():
    if not STATS.is_file():
        pytest.skip("nn_final_stats.json not generated")
    return json.loads(STATS.read_text())


def test_rebuild_variance_all_distinct():
    final = _final()
    if "rebuild_variance" not in final:
        pytest.skip("rebuild variance ledger not integrated")
    rb = final["rebuild_variance"]
    assert rb["all_rebuilds_distinct"] is True
    assert rb["total_rebuilds"] == 35
    for arm, a in rb["arms"].items():
        assert a["n_rebuilds"] == 5
        assert a["distinct_engine_sha256"] == 5
        assert a["distinct_inspector_sha256"] == 5


def test_rebuild_ap_spread_bounded():
    final = _final()
    if "rebuild_variance" not in final:
        pytest.skip("rebuild variance ledger not integrated")
    spreads = final["rebuild_variance"].get("ap_spread", {})
    if not spreads:
        pytest.skip("ap-eval metrics not synced")
    for key, v in spreads.items():
        # the audit claims rebuilds do not move measured AP beyond the
        # printed precision; anything above 0.05 AP would falsify the claim
        assert v["spread"] <= 0.05, f"{key} spread {v['spread']}"
    # full coverage: two conditions x five rebuilds per evaluated arm
    complete = [k for k, v in spreads.items() if v["n"] == 5]
    if len(complete) < len(spreads):
        pytest.skip("ap-eval incomplete: <5 rebuilds measured for some cells")


@pytest.mark.skipif(not LAT_DIR.is_dir(), reason="latency benchmark records absent")
def test_latency_records_idle_gated():
    files = list(LAT_DIR.glob("*__rep-*.json"))
    assert len(files) == 78  # 26 engines x 3 repetitions
    for f in files:
        d = json.loads(f.read_text())
        assert int(d["n_samples"]) == 500
        assert int(d["warmup_iters"]) == 200
        assert float(d["latency_median_ms"]) > 0
        # idle gate: the allow-listed compute pids must be the only entries
        allow_raw = d["gpu_idle_allow_pids"]
        allowed = set(json.loads(allow_raw) if isinstance(allow_raw, str)
                      else allow_raw)
        observed = {int(x) for x in str(d["gpu_idle_query_output"]).split()}
        assert observed <= allowed


def test_latency_integrated_and_macro_parity():
    final = _final()
    if "latency" not in final:
        pytest.skip("latency ledger not integrated")
    conds = final["latency"]["conditions"]
    assert len(conds) == 26
    assert all(e["n_reps"] == 3 for e in conds.values())
    text = NUMBERS.read_text()
    m = re.search(r"newcommand\{\\nnLatIntEightMin\}\{([0-9.]+)\}", text)
    assert m, "macro nnLatIntEightMin missing"
    int8 = min(e["median_ms"] for c, e in conds.items()
               if c.endswith("int8-matched512"))
    assert abs(float(m.group(1)) - int8) < 0.005
