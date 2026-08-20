from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from topic_c.manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "src" / "run_fixed_universe_sensitivity.py"
    assert path.is_file(), "fixed-universe sensitivity runner has not been implemented"
    spec = importlib.util.spec_from_file_location("run_fixed_universe_sensitivity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_contrast_uses_fp8_minus_int8_sign_and_height_difference() -> None:
    module = load_module()
    arms = [
        np.asarray([0.50, 0.40, 0.60]),  # INT8 clean
        np.asarray([0.55, 0.48, 0.62]),  # FP8 clean
        np.asarray([0.20, 0.10, 0.30]),  # INT8 corrupt
        np.asarray([0.28, 0.22, 0.31]),  # FP8 corrupt
    ]

    contrast = module.paired_contrast(arms)

    np.testing.assert_allclose(contrast["delta_e"], np.asarray([0.03, 0.04, -0.01]))
    assert contrast["delta_psi"] == pytest.approx(0.04)


def test_summary_reports_2k_vs_full_monte_carlo_endpoint_stability() -> None:
    module = load_module()
    values = np.linspace(-0.02, 0.03, 10_000)

    summary = module.summarize_draws(values, checkpoint=2_000)

    assert summary["n_boot"] == 10_000
    assert summary["checkpoint_n_boot"] == 2_000
    assert len(summary["percentile_interval"]) == 3
    expected = max(
        abs(left - right)
        for left, right in zip(
            np.percentile(values[:2_000], [2.5, 50, 97.5]),
            np.percentile(values, [2.5, 50, 97.5]),
        )
    )
    assert summary["max_checkpoint_percentile_shift_native_ap"] == pytest.approx(expected)


def test_materialized_weight_schedule_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    module = load_module()
    first = module.write_weight_schedule(tmp_path / "first.npz", n_images=4, n_boot=3, seed=17)
    second = module.write_weight_schedule(tmp_path / "second.npz", n_images=4, n_boot=3, seed=17)

    assert first["weights_sha256"] == second["weights_sha256"]
    assert first["schedule_identity_sha256"] == second["schedule_identity_sha256"]
    with np.load(tmp_path / "first.npz", allow_pickle=False) as schedule:
        assert schedule["weights"].shape == (3, 4)
        assert np.all(schedule["weights"] > 0)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        module.write_weight_schedule(tmp_path / "first.npz", n_images=4, n_boot=3, seed=17)


def test_uniform_point_uses_prepared_accumulators(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(
        module,
        "accumulate_prepared_ap",
        lambda prepared, weights: np.asarray(prepared["ap"], dtype=float),
    )
    standard = [
        {"ap": [0.50]}, {"ap": [0.55]}, {"ap": [0.20]}, {"ap": [0.28]},
    ]
    height = [
        {"ap": [0.40, 0.60]}, {"ap": [0.48, 0.62]},
        {"ap": [0.10, 0.30]}, {"ap": [0.22, 0.31]},
    ]

    overall, size = module.evaluate_uniform_point(standard, height, n_images=3)

    assert overall["delta_e"][0] == pytest.approx(0.03)
    assert size["delta_psi"] == pytest.approx(0.05)


def test_validate_artifact_recomputes_draw_summaries_and_rejects_mutation(tmp_path: Path) -> None:
    module = load_module()
    schedule_path = tmp_path / "weights.npz"
    schedule = module.write_weight_schedule(schedule_path, n_images=4, n_boot=3, seed=17)
    delta_e = np.asarray([-0.1, 0.0, 0.2])
    delta_psi = np.asarray([-0.3, -0.2, 0.1])
    draws_path = tmp_path / "draws.npz"
    np.savez_compressed(
        draws_path,
        schema_version=np.asarray(1),
        schedule_identity_sha256=np.asarray(schedule["schedule_identity_sha256"]),
        delta_e_all=delta_e,
        delta_psi_height=delta_psi,
    )
    annotation = tmp_path / "annotation.json"
    annotation.write_text("{}\n", encoding="utf-8")
    component = tmp_path / "component.json"
    component.write_text(
        json.dumps(
            {
                "point": {"delta_e": {"all": 0.01}, "delta_psi": -0.02},
                "n_boot": 2,
                "percentile_intervals": {
                    "delta_e": {"all": [-0.2, 0.0, 0.2]},
                    "delta_psi": [-0.4, -0.2, 0.2],
                },
            }
        ),
        encoding="utf-8",
    )
    bindings = {}
    image_hash = "a" * 64
    for arm in module.ARM_NAMES:
        prediction = tmp_path / f"{arm}.pred.json"
        inputs = tmp_path / f"{arm}.input.json"
        run = tmp_path / f"{arm}.run.json"
        prediction.write_text("[]\n", encoding="utf-8")
        inputs.write_text(
            json.dumps({"input_manifest_sha256": "b" * 64, "image_ids_sha256": image_hash}),
            encoding="utf-8",
        )
        run.write_text("{}\n", encoding="utf-8")
        bindings[arm] = {
            "prediction_sha256": sha256_file(prediction),
            "input_record_sha256": sha256_file(inputs),
            "input_manifest_sha256": "b" * 64,
            "image_ids_sha256": image_hash,
            "run_record_sha256": sha256_file(run),
            "prediction_path": prediction.name,
            "input_record_path": inputs.name,
            "run_record_path": run.name,
        }
    report = {
        "schema_version": 1,
        "method": "paired fixed-category-universe Bayesian image bootstrap with positive image weights",
        "scope": "TT100K sensitivity estimand; not a replacement for the ordinary multinomial image bootstrap",
        "component": component.name,
        "component_sha256": sha256_file(component),
        "annotation": {"path": annotation.name, "sha256": sha256_file(annotation)},
        "n_images": 4,
        "n_boot": 3,
        "seed": 17,
        "schedule": {**schedule, "path": schedule_path.name},
        "input_hashes": bindings,
        "point": {"delta_e_all": 0.01, "delta_psi_height": -0.02},
        "fixed_universe": {
            "delta_e_all": module.summarize_draws(delta_e, checkpoint=2),
            "delta_psi_height": module.summarize_draws(delta_psi, checkpoint=2),
        },
        "ordinary_reference": {
            "n_boot": 2,
            "delta_e_all": [-0.2, 0.0, 0.2],
            "delta_psi_height": [-0.4, -0.2, 0.2],
        },
        "draws": {"path": draws_path.name, "sha256": sha256_file(draws_path)},
        "runtime_seconds_draws_only": 1.0,
    }
    report["artifact_sha256"] = module.canonical_hash(report, "artifact_sha256")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    validated = module.validate_fixed_universe_artifact(
        tmp_path, report_path, expected_n_boot=3, expected_images=4
    )
    assert validated["artifact_sha256"] == report["artifact_sha256"]

    with np.load(draws_path, allow_pickle=False) as cache:
        changed = {name: cache[name] for name in cache.files}
    changed["delta_e_all"] = np.asarray([9.0, 9.0, 9.0])
    np.savez_compressed(draws_path, **changed)
    with pytest.raises(SystemExit, match="draw cache hash mismatch"):
        module.validate_fixed_universe_artifact(
            tmp_path, report_path, expected_n_boot=3, expected_images=4
        )
