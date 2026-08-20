#!/usr/bin/env python3
"""Fail-closed bounded subprocess scheduler for immutable contrast cells."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class CellJob:
    """One independently runnable immutable cell command."""

    key: str
    command: tuple[str, ...]
    output_paths: tuple[Path, ...]
    log_path: Path


class CellSchedulerError(RuntimeError):
    """Raised only after a durable failure ledger has been written."""


def _canonical_hash(document: dict[str, Any], hash_field: str = "ledger_sha256") -> str:
    payload = {key: value for key, value in document.items() if key != hash_field}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(root: Path, path: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"{label} path escapes scheduler root: {path}") from exc


def _relative_value(root: Path, value: Any, label: str) -> Path:
    """Resolve a manifest path only when it is a safe project-relative value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be safe and relative: {value}")
    path = (root / relative).resolve()
    _relative(root, path, label)
    return path


def _validate_jobs(jobs: Sequence[CellJob], *, root: Path) -> list[CellJob]:
    if not jobs:
        raise ValueError("cell scheduler requires at least one job")
    ordered = sorted(jobs, key=lambda job: job.key)
    if any(not isinstance(job.key, str) or not job.key or not job.command or not job.output_paths for job in ordered):
        raise ValueError("cell scheduler job is incomplete")
    if len({job.key for job in ordered}) != len(ordered):
        raise ValueError("cell scheduler job keys must be unique")
    outputs = [path.resolve() for job in ordered for path in job.output_paths]
    if len(set(outputs)) != len(outputs):
        raise ValueError("cell scheduler has overlapping output path(s)")
    logs = [job.log_path.resolve() for job in ordered]
    if len(set(logs)) != len(logs):
        raise ValueError("cell scheduler has overlapping task log path(s)")
    for job in ordered:
        _relative(root, job.log_path, "task log")
        for output in job.output_paths:
            _relative(root, output, "task output")
            if output.exists():
                raise ValueError(f"cell scheduler refuses existing immutable output: {output}")
        if job.log_path.exists():
            raise ValueError(f"cell scheduler refuses existing task log: {job.log_path}")
    return ordered


def _validate_execution_package(root: Path, execution_package: dict[str, Any] | None) -> dict[str, str]:
    """Verify the immutable source/config manifest before a child can launch."""
    if (not isinstance(execution_package, dict)
            or set(execution_package) != {"path", "sha256", "manifest_sha256"}
            or any(not isinstance(execution_package[name], str) or len(execution_package[name]) != 64
                   for name in ("sha256", "manifest_sha256"))):
        raise ValueError("scheduler execution package binding is invalid")
    path = _relative_value(root, execution_package["path"], "execution package")
    if not path.is_file() or _sha256_file(path) != execution_package["sha256"]:
        raise ValueError("scheduler execution package file SHA-256 mismatch")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("scheduler execution package is invalid JSON") from exc
    if (not isinstance(document, dict)
            or document.get("execution_root") != str(root)
            or document.get("manifest_sha256") != _canonical_hash(document, "manifest_sha256")
            or document.get("manifest_sha256") != execution_package["manifest_sha256"]):
        raise ValueError("scheduler execution package manifest hash mismatch")
    files = document.get("files_sha256")
    if not isinstance(files, dict):
        raise ValueError("scheduler execution package source bindings are invalid")
    for relative, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("scheduler execution package source bindings are invalid")
        source = _relative_value(root, relative, "execution package source")
        if not source.is_file() or _sha256_file(source) != expected:
            raise ValueError(f"scheduler execution package source SHA-256 mismatch: {relative}")
    return dict(execution_package)


def _write_ledger(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise ValueError(f"cell scheduler refuses existing ledger: {path}")
    document["ledger_sha256"] = _canonical_hash(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preserved_bytes(root: Path, job: CellJob, record: dict[str, Any]) -> None:
    """Bind every preserved output and log, including nonzero child results."""
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for output in job.output_paths:
        relative = _relative(root, output, "task output")
        if output.is_file():
            hashes[relative] = _sha256_file(output)
        else:
            missing.append(relative)
    record["output_sha256"] = hashes
    record["missing_output_paths"] = missing
    if job.log_path.is_file():
        record["log_sha256"] = _sha256_file(job.log_path)
    else:
        record["log_sha256"] = None
        record["log_missing"] = True


def _poll_process(process: subprocess.Popen[bytes]) -> int | None:
    """Small seam for deterministic fault-injection coverage of finalization."""
    return process.poll()


def _finish_record(
    *, root: Path, key: str, job: CellJob, process: subprocess.Popen[bytes], handle: Any,
    record: dict[str, Any], records: dict[str, dict[str, Any]], verify: Callable[[CellJob], dict[str, Any]],
    returncode: int | None = None,
) -> str | None:
    """Close one child, bind its preserved bytes, and return an eventual failure description."""
    try:
        handle.close()
    except OSError:
        pass
    if returncode is None:
        returncode = process.wait()
    record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    record["exit_status"] = returncode
    failure: str | None = None
    if returncode == 0:
        try:
            record["verification"] = verify(job)
        except Exception as exc:  # verifier detail belongs in the durable ledger
            record["error"] = f"verification error: {exc}"
            failure = f"verification failure: {key}"
    else:
        record["error"] = f"process exited with status {returncode}"
        failure = f"process failure: {key}"
    try:
        _preserved_bytes(root, job, record)
    except Exception as exc:
        record["output_hash_error"] = str(exc)
        failure = failure or f"output-hash failure: {key}"
    records[key] = record
    if returncode == 0 and not record.get("missing_output_paths"):
        return failure
    if returncode == 0 and record.get("missing_output_paths"):
        record["error"] = record.get("error") or "declared output(s) are missing"
        return failure or f"output-hash failure: {key}"
    return failure


def run_bounded_cells(
    jobs: Sequence[CellJob], *, cell_workers: int, ledger_path: Path,
    verify: Callable[[CellJob], dict[str, Any]], root: Path | None = None,
    execution_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run independent immutable cells with a cap and durable failure ledger.

    New launches stop after the first nonzero exit or verification error.  Any
    already launched cell is allowed to exit; no output, cache, or log is ever
    deleted or overwritten.
    """
    if not isinstance(cell_workers, int) or isinstance(cell_workers, bool) or cell_workers <= 0:
        raise ValueError("cell scheduler worker cap must be positive")
    root = (root or ledger_path.parent).resolve()
    _relative(root, ledger_path, "ledger")
    if ledger_path.exists():
        raise ValueError(f"cell scheduler refuses existing ledger: {ledger_path}")
    ordered = _validate_jobs(jobs, root=root)
    package = _validate_execution_package(root, execution_package)
    records: dict[str, dict[str, Any]] = {}
    active: dict[str, tuple[CellJob, subprocess.Popen[bytes], Any, dict[str, Any]]] = {}
    next_index, peak_active = 0, 0
    failure: str | None = None
    unexpected: BaseException | None = None
    try:
        while next_index < len(ordered) or active:
            while failure is None and len(active) < cell_workers and next_index < len(ordered):
                job = ordered[next_index]
                next_index += 1
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                started = datetime.now(timezone.utc).isoformat()
                record: dict[str, Any] = {
                    "key": job.key,
                    "command": list(job.command),
                    "pid": None,
                    "log_path": _relative(root, job.log_path, "task log"),
                    "output_paths": [_relative(root, path, "task output") for path in job.output_paths],
                    "started_at_utc": started,
                    "ended_at_utc": None,
                    "exit_status": None,
                    "verification": None,
                }
                handle = None
                try:
                    handle = job.log_path.open("xb")
                    process = subprocess.Popen(job.command, stdout=handle, stderr=subprocess.STDOUT)
                except OSError as exc:
                    if handle is not None:
                        handle.close()
                    record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                    record["exit_status"] = None
                    record["error"] = f"launch error: {exc}"
                    _preserved_bytes(root, job, record)
                    records[job.key] = record
                    failure = f"launch failure: {job.key}"
                    break
                record["pid"] = process.pid
                active[job.key] = (job, process, handle, record)
                peak_active = max(peak_active, len(active))

            completed = []
            for key, (job, process, handle, record) in list(active.items()):
                returncode = _poll_process(process)
                if returncode is None:
                    continue
                record_failure = _finish_record(
                    root=root, key=key, job=job, process=process, handle=handle, record=record,
                    records=records, verify=verify, returncode=returncode,
                )
                failure = failure or record_failure
                completed.append(key)
            for key in completed:
                del active[key]
            if failure is not None and not active:
                break
            if active and not completed:
                time.sleep(0.01)
    except BaseException as exc:  # finalization below preserves every launched cell before returning control
        unexpected = exc
        failure = failure or f"unexpected scheduler error: {type(exc).__name__}: {exc}"
    finally:
        for key, (job, process, handle, record) in list(active.items()):
            try:
                returncode = process.wait()
                record_failure = _finish_record(
                    root=root, key=key, job=job, process=process, handle=handle, record=record,
                    records=records, verify=verify, returncode=returncode,
                )
                failure = failure or record_failure
            except BaseException as exc:
                try:
                    handle.close()
                except OSError:
                    pass
                record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                record["error"] = f"finalization error: {type(exc).__name__}: {exc}"
                try:
                    _preserved_bytes(root, job, record)
                except Exception as hash_exc:
                    record["output_hash_error"] = str(hash_exc)
                records[key] = record
                failure = failure or f"unexpected scheduler error: {type(exc).__name__}: {exc}"
        active.clear()

    document: dict[str, Any] = {
        "schema_version": 1,
        "scheduler": "bounded independent immutable format-contrast cells",
        "execution_package": package,
        "cell_workers": cell_workers,
        "peak_active": peak_active,
        "status": "failed" if failure is not None else "complete",
        "failure": failure,
        "records": [records[job.key] for job in ordered if job.key in records],
    }
    document = _write_ledger(ledger_path, document)
    if unexpected is not None:
        raise CellSchedulerError(f"cell scheduler failed: {failure}; ledger={ledger_path}") from unexpected
    if failure is not None:
        raise CellSchedulerError(f"cell scheduler failed: {failure}; ledger={ledger_path}")
    return document


if __name__ == "__main__":  # pragma: no cover - module is driven by the orchestrator
    sys.exit("format_contrast_scheduler is a library module")
