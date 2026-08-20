from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

from format_contrast_scheduler import CellJob, CellSchedulerError, run_bounded_cells
import format_contrast_scheduler
import run_format_contrast_p0
from run_format_contrast_p0 import build_clean_cell_jobs, build_corrupt_cell_jobs, validate_runtime_config


def _job(tmp_path: Path, key: str, code: str, output_name: str | None = None) -> CellJob:
    output = tmp_path / (output_name or f"{key}.json")
    command = (
        "from pathlib import Path; "
        f"Path({str(output)!r}).write_text('immutable output', encoding='utf-8'); "
        + code
    )
    return CellJob(
        key=key,
        command=(sys.executable, "-c", command),
        output_paths=(output,),
        log_path=tmp_path / "logs" / f"{key}.log",
    )


def _execution_package(tmp_path: Path, sources: dict[str, Path] | None = None) -> dict[str, str]:
    manifest = tmp_path / "outputs" / "reports" / "execution-package.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    files = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in (sources or {}).items()
    }
    document = {
        "schema_version": 1, "attempt": "ivc_format_contrast_v1",
        "execution_root": str(tmp_path.resolve()), "files_sha256": files,
    }
    document["manifest_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return {
        "path": str(manifest.relative_to(tmp_path)),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "manifest_sha256": document["manifest_sha256"],
    }


def test_scheduler_enforces_cap_records_every_cell_and_uses_unique_logs(tmp_path: Path) -> None:
    """Catches a cross-cell scheduler that oversubscribes or loses immutable task records."""
    jobs = [
        _job(tmp_path, f"cell-{index}", "import time; time.sleep(0.04)")
        for index in range(3)
    ]
    ledger = tmp_path / "ledger.json"

    document = run_bounded_cells(
        jobs, cell_workers=2, ledger_path=ledger, verify=lambda job: {"verified": job.key},
        execution_package=_execution_package(tmp_path),
    )

    assert document["status"] == "complete"
    assert document["peak_active"] <= 2
    assert [record["key"] for record in document["records"]] == ["cell-0", "cell-1", "cell-2"]
    assert all(record["exit_status"] == 0 and record["pid"] > 0 for record in document["records"])
    assert len({record["log_path"] for record in document["records"]}) == 3
    assert all((tmp_path / record["log_path"]).is_file() for record in document["records"])
    for record in document["records"]:
        assert record["log_sha256"] == hashlib.sha256((tmp_path / record["log_path"]).read_bytes()).hexdigest()
        assert record["output_sha256"] == {
            record["output_paths"][0]: hashlib.sha256(b"immutable output").hexdigest(),
        }
    assert json.loads(ledger.read_text(encoding="utf-8"))["ledger_sha256"] == document["ledger_sha256"]


def test_scheduler_binds_a_precommitted_execution_package_and_each_task_log(tmp_path: Path) -> None:
    """Catches a long-run ledger that cannot identify its exact executing source package."""
    package = _execution_package(tmp_path)

    document = run_bounded_cells(
        [_job(tmp_path, "cell", "pass")], cell_workers=1, ledger_path=tmp_path / "ledger.json",
        verify=lambda job: {}, root=tmp_path, execution_package=package,
    )

    assert document["execution_package"] == package
    assert document["records"][0]["log_sha256"] == hashlib.sha256(
        (tmp_path / document["records"][0]["log_path"]).read_bytes()
    ).hexdigest()


def test_scheduler_rechecks_execution_package_source_bytes_before_launch(tmp_path: Path) -> None:
    """Catches a worker launch after a precommitted production source has changed."""
    source = tmp_path / "src" / "runner.py"
    source.parent.mkdir()
    source.write_text("reviewed source", encoding="utf-8")
    package = _execution_package(tmp_path, {"src/runner.py": source})
    source.write_text("mutated source", encoding="utf-8")

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        run_bounded_cells(
            [_job(tmp_path, "cell", "pass")], cell_workers=1, ledger_path=tmp_path / "ledger.json",
            verify=lambda job: {}, root=tmp_path, execution_package=package,
        )


def test_completion_ledger_binding_requires_the_completed_phase_and_execution_package(tmp_path: Path) -> None:
    """Catches a completion report binding a scheduler ledger from another package or incomplete phase."""
    from run_format_contrast_p0 import scheduler_ledger_binding

    package = _execution_package(tmp_path)
    ledger = tmp_path / "outputs" / "reports" / "ivc_format_contrast_v1_clean_scheduler.json"
    run_bounded_cells(
        [_job(tmp_path, "cell", "pass")], cell_workers=1, ledger_path=ledger,
        verify=lambda job: {}, root=tmp_path, execution_package=package,
    )

    binding = scheduler_ledger_binding(tmp_path, ledger, package, "clean", expected_records=1)
    assert binding["sha256"] == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert binding["ledger_sha256"] == json.loads(ledger.read_text(encoding="utf-8"))["ledger_sha256"]


def test_scheduler_refuses_overlapping_immutable_output_paths(tmp_path: Path) -> None:
    """Catches two independent processes targeting the same evidence file."""
    duplicate = tmp_path / "same.json"
    jobs = [
        _job(tmp_path, "first", "pass", output_name=duplicate.name),
        _job(tmp_path, "second", "pass", output_name=duplicate.name),
    ]

    with pytest.raises(ValueError, match="overlapping output path"):
        run_bounded_cells(
            jobs, cell_workers=2, ledger_path=tmp_path / "ledger.json", verify=lambda job: {},
            execution_package=_execution_package(tmp_path),
        )


def test_scheduler_stops_new_launches_after_failure_and_preserves_failure_ledger(tmp_path: Path) -> None:
    """Catches a scheduler that continues launching cells after one immutable cell fails."""
    jobs = [
        _job(tmp_path, "fail-first", "import sys; sys.exit(7)"),
        _job(tmp_path, "must-not-launch", "raise SystemExit(0)"),
    ]
    ledger = tmp_path / "failure-ledger.json"

    with pytest.raises(CellSchedulerError, match="cell scheduler failed"):
        run_bounded_cells(jobs, cell_workers=1, ledger_path=ledger, verify=lambda job: {}, execution_package=_execution_package(tmp_path))

    document = json.loads(ledger.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert [record["key"] for record in document["records"]] == ["fail-first"]
    assert document["records"][0]["exit_status"] == 7
    assert not (tmp_path / "logs" / "must-not-launch.log").exists()


def test_scheduler_binds_partial_outputs_missing_outputs_and_log_after_nonzero_exit(tmp_path: Path) -> None:
    """Catches a child failure ledger that loses evidence written before its nonzero exit."""
    output, missing = tmp_path / "partial.json", tmp_path / "missing-cache.npz"
    job = CellJob(
        key="partial", command=(sys.executable, "-c", f"from pathlib import Path; Path({str(output)!r}).write_text('partial'); raise SystemExit(9)"),
        output_paths=(output, missing), log_path=tmp_path / "logs" / "partial.log",
    )
    ledger = tmp_path / "partial-ledger.json"

    with pytest.raises(CellSchedulerError, match="process failure"):
        run_bounded_cells(
            [job], cell_workers=1, ledger_path=ledger, verify=lambda job: {}, root=tmp_path,
            execution_package=_execution_package(tmp_path),
        )

    record = json.loads(ledger.read_text(encoding="utf-8"))["records"][0]
    assert record["exit_status"] == 9
    assert record["output_sha256"] == {"partial.json": hashlib.sha256(b"partial").hexdigest()}
    assert record["missing_output_paths"] == ["missing-cache.npz"]
    assert record["log_sha256"] == hashlib.sha256((tmp_path / "logs" / "partial.log").read_bytes()).hexdigest()


def test_scheduler_refuses_a_ledger_path_that_escapes_its_root(tmp_path: Path) -> None:
    """Catches a ledger write outside the reviewed project root before any child launches."""
    outside = tmp_path.parent / f"{tmp_path.name}-outside-ledger.json"

    with pytest.raises(ValueError, match="ledger path escapes scheduler root"):
        run_bounded_cells(
            [_job(tmp_path, "cell", "pass")], cell_workers=1, ledger_path=outside, verify=lambda job: {}, root=tmp_path,
            execution_package=_execution_package(tmp_path),
        )

    assert not outside.exists()


def test_scheduler_ledger_creation_refuses_a_toctou_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a ledger writer that overwrites a file created after its precheck."""
    ledger = tmp_path / "race-ledger.json"
    original_exists = Path.exists

    def late_creation(path: Path) -> bool:
        if path == ledger:
            ledger.write_text("concurrent immutable ledger", encoding="utf-8")
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", late_creation)
    with pytest.raises(FileExistsError):
        run_bounded_cells(
            [_job(tmp_path, "cell", "pass")], cell_workers=1, ledger_path=ledger,
            verify=lambda job: {}, root=tmp_path, execution_package=_execution_package(tmp_path),
        )

    assert ledger.read_text(encoding="utf-8") == "concurrent immutable ledger"


def test_scheduler_drains_active_children_and_writes_failure_ledger_after_internal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches an unexpected scheduler exception that strands a peer or skips its durable ledger."""
    jobs = [
        _job(tmp_path, "first", "import time; time.sleep(0.08)"),
        _job(tmp_path, "second", "import time; time.sleep(0.08)"),
    ]
    ledger = tmp_path / "internal-error-ledger.json"
    original = getattr(format_contrast_scheduler, "_poll_process", None)
    calls = 0

    def injected(process):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected scheduler fault")
        return process.poll()

    monkeypatch.setattr(format_contrast_scheduler, "_poll_process", injected, raising=False)
    with pytest.raises(CellSchedulerError, match="unexpected scheduler error"):
        run_bounded_cells(
            jobs, cell_workers=2, ledger_path=ledger, verify=lambda job: {}, root=tmp_path,
            execution_package=_execution_package(tmp_path),
        )

    document = json.loads(ledger.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert len(document["records"]) == 2
    assert all(record["ended_at_utc"] and record["log_sha256"] for record in document["records"])
    assert all((tmp_path / path).is_file() for record in document["records"] for path in record["output_paths"])
    assert original is None or original is not injected


def test_scheduler_hashes_a_created_output_even_when_its_verifier_rejects_it(tmp_path: Path) -> None:
    """Catches a failure ledger that loses the bytes it deliberately preserves."""
    job = _job(tmp_path, "bad-cache", "pass")
    ledger = tmp_path / "failure-ledger.json"

    with pytest.raises(CellSchedulerError, match="verification failure"):
        run_bounded_cells(
            [job], cell_workers=1, ledger_path=ledger,
            verify=lambda ignored: (_ for _ in ()).throw(ValueError("tampered cache")),
            execution_package=_execution_package(tmp_path),
        )

    record = json.loads(ledger.read_text(encoding="utf-8"))["records"][0]
    assert record["exit_status"] == 0
    assert record["output_sha256"] == {"bad-cache.json": hashlib.sha256(b"immutable output").hexdigest()}
    assert "verification error: tampered cache" == record["error"]


def test_runtime_config_requires_cell_workers_and_prohibits_nested_bootstrap_workers() -> None:
    """Catches a nested cell-times-bootstrap process topology."""
    valid = {
        "schema_version": 1, "attempt": "ivc_format_contrast_v1", "n_boot": 2000, "seed_namespace": "scheduler-test",
        "bootstrap_workers": 1, "cell_workers": 4,
    }

    validate_runtime_config(valid)
    with pytest.raises(ValueError, match="cell_workers"):
        validate_runtime_config({key: value for key, value in valid.items() if key != "cell_workers"})
    with pytest.raises(ValueError, match="nested"):
        validate_runtime_config({**valid, "bootstrap_workers": 2})
    with pytest.raises(ValueError, match="cell worker cap"):
        validate_runtime_config({**valid, "cell_workers": 5})
    for name, value in (("n_boot", "2000"), ("bootstrap_workers", True), ("cell_workers", 1.0)):
        with pytest.raises(ValueError, match="invalid config"):
            validate_runtime_config({**valid, name: value})
    for attempt in ("../escape", "/tmp/escape", "other_attempt"):
        with pytest.raises(ValueError, match="attempt"):
            validate_runtime_config({**valid, "attempt": attempt})


def test_resume_mode_refuses_to_touch_a_stale_partial_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a claimed resume path that repairs, deletes, or trusts stale bytes."""
    config = tmp_path / "configs" / "ivc_format_contrast_v1.json"
    config.parent.mkdir()
    config.write_text(json.dumps({
        "schema_version": 1, "attempt": "ivc_format_contrast_v1", "n_boot": 2000, "seed_namespace": "resume-test",
        "bootstrap_workers": 1, "cell_workers": 1,
    }), encoding="utf-8")
    stale = tmp_path / "outputs" / "bootstrap" / "ivc_format_contrast_v1" / "altered-cache.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"altered cache bytes")
    before = stale.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "run_format_contrast_p0.py", "--project-root", str(tmp_path), "--config", str(config), "--resume-verified",
    ])

    with pytest.raises(SystemExit, match="resume-verified is not implemented"):
        run_format_contrast_p0.main()

    assert stale.read_bytes() == before


def test_runner_builds_deterministic_nonoverlapping_clean_and_corrupt_cell_jobs(tmp_path: Path) -> None:
    """Catches orchestration that reuses a cell log/output or reintroduces nested workers."""
    root, config_path = tmp_path, tmp_path / "configs" / "contrast.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    config = {"attempt": "attempt", "bootstrap_workers": 1, "n_boot": 2000, "seed_namespace": "seed"}
    clean = {
        "dataset": "coco", "model": "yolo11n", "output": root / "outputs" / "bootstrap" / "attempt" / "coco__yolo11n__clean-s0.json",
    }
    corrupt = {
        **clean, "endpoint": "area", "annotations": tmp_path / "annotations.json", "expected_images": 2,
        "annotation_sha256": "c" * 64, "corruption": "fog", "severity": 1,
        "output": root / "outputs" / "bootstrap" / "attempt" / "coco__yolo11n__fog-s1.json",
    }
    sources = ([tmp_path / "prediction.json"] * 4, [tmp_path / "input.json"] * 4, [tmp_path / "run.json"] * 4)
    schedule = {"path": str(tmp_path / "schedule.npz")}
    clean_reference = {"path": tmp_path / "clean.npz", "sha256": "a" * 64, "identity_sha256": "b" * 64}

    clean_jobs = build_clean_cell_jobs(root, config_path, config, [clean])
    corrupt_jobs = build_corrupt_cell_jobs(root, config, [corrupt], {"coco": schedule}, {("coco", "yolo11n"): clean_reference}, lambda task: sources)

    jobs = clean_jobs + corrupt_jobs
    assert [job.key for job in jobs] == ["clean:coco__yolo11n__clean-s0", "corrupt:coco__yolo11n__fog-s1"]
    assert len({path for job in jobs for path in job.output_paths}) == 4
    assert len({job.log_path for job in jobs}) == 2
    assert corrupt_jobs[0].command[corrupt_jobs[0].command.index("--workers") + 1] == "1"
