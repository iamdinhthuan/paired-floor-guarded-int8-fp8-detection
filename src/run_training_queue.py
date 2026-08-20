#!/usr/bin/env python3
"""Run the frozen VOC/KITTI YOLO11 training queue serially on one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import sha256_file


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def complete_registry(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    return path.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha256_file(path)


def running(run_id: str) -> bool:
    result = subprocess.run(["pgrep", "-f", f"train_yolo_dataset.py.*--run-id {run_id}"], stdout=subprocess.DEVNULL)
    return result.returncode == 0


def resolve_job_inputs(root: Path, queue: dict, dataset: str) -> dict[str, Path | None]:
    configured = queue.get("datasets")
    if configured is None:
        return {
            "data_yaml": (root / "configs" / "datasets" / f"{dataset}_ultralytics_v1.yaml").resolve(),
            "acquisition_registry": (root / "manifests" / "datasets" / f"{dataset}_acquisition_v1.json").resolve(),
            "split_registry": None,
        }
    specification = configured.get(dataset) if isinstance(configured, dict) else None
    if not isinstance(specification, dict) or set(specification) != {
        "data_yaml", "acquisition_registry", "split_registry"
    }:
        raise SystemExit(f"TRAINING QUEUE REFUSED: invalid dataset inputs for {dataset}")
    resolved = {key: (root / value).resolve() for key, value in specification.items() if isinstance(value, str)}
    if set(resolved) != set(specification) or not all(path.is_file() for path in resolved.values()):
        raise SystemExit(f"TRAINING QUEUE REFUSED: dataset input missing for {dataset}")
    return resolved


def compact_if_requested(root: Path, queue: dict, registry: Path, run_id: str) -> str | None:
    if not queue.get("retain_only_best", False):
        return None
    report = root / "manifests" / "training" / f"{run_id}_best_only_v1.json"
    marker = report.with_suffix(report.suffix + ".complete")
    if report.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha256_file(report):
        document = json.loads(report.read_text(encoding="utf-8"))
        best = Path(document.get("retained_best_weights", ""))
        last = Path(document.get("deleted_last_weights", ""))
        if (
            document.get("training_registry_sha256") == sha256_file(registry)
            and best.is_file()
            and sha256_file(best) == document.get("retained_best_weights_sha256")
            and not last.exists()
        ):
            return sha256_file(report)
        raise SystemExit(f"TRAINING QUEUE REFUSED: invalid checkpoint compaction report: {report}")
    if report.exists() or marker.exists():
        raise SystemExit(f"TRAINING QUEUE REFUSED: incomplete checkpoint compaction report: {report}")
    subprocess.run(
        [sys.executable, str(root / "src" / "compact_training_checkpoint.py"),
         "--training-registry", str(registry), "--out", str(report)],
        check=True,
    )
    if not report.is_file() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != sha256_file(report):
        raise SystemExit(f"TRAINING QUEUE REFUSED: checkpoint compaction did not complete: {run_id}")
    return sha256_file(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--resume-partial", action="store_true", help="resume only a hash-guarded partial run from weights/last.pt")
    args = parser.parse_args()
    root, queue_path, report = Path(args.project_root).resolve(), Path(args.queue).resolve(), Path(args.report_out).resolve()
    if report.exists():
        raise SystemExit(f"TRAINING QUEUE REFUSED: report already exists: {report}")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or not jobs or len({job.get("run_id") for job in jobs}) != len(jobs):
        raise SystemExit("TRAINING QUEUE REFUSED: frozen queue is invalid")
    profile = root / queue.get("profile", "")
    if not profile.is_file():
        raise SystemExit("TRAINING QUEUE REFUSED: frozen training profile is absent")
    records: list[dict] = []
    for ordinal, job in enumerate(jobs, start=1):
        dataset, model, run_id = job.get("dataset"), job.get("model"), job.get("run_id")
        if dataset not in {"voc", "kitti", "tt100k"} or model not in {"yolo11n", "yolo11m", "yolo11x"} or not isinstance(run_id, str):
            raise SystemExit(f"TRAINING QUEUE REFUSED: invalid job {ordinal}")
        registry = root / "manifests" / "training" / f"{run_id}.json"
        if complete_registry(registry):
            compaction_sha = compact_if_requested(root, queue, registry, run_id)
            records.append({"run_id": run_id, "registry_sha256": sha256_file(registry),
                            "compaction_report_sha256": compaction_sha, "state": "already_complete"})
            print(f"TRAIN QUEUE {ordinal}/{len(jobs)} already complete {run_id}", flush=True)
            continue
        while running(run_id):
            print(f"TRAIN QUEUE {ordinal}/{len(jobs)} waiting for active run {run_id}", flush=True)
            time.sleep(args.poll_seconds)
            if complete_registry(registry):
                break
        if complete_registry(registry):
            compaction_sha = compact_if_requested(root, queue, registry, run_id)
            records.append({"run_id": run_id, "registry_sha256": sha256_file(registry),
                            "compaction_report_sha256": compaction_sha, "state": "completed_while_waiting"})
            print(f"TRAIN QUEUE {ordinal}/{len(jobs)} complete {run_id}", flush=True)
            continue
        output = root / "outputs" / "training" / dataset / run_id
        resume_checkpoint = None
        if output.exists():
            if not args.resume_partial:
                raise SystemExit(f"TRAINING QUEUE REFUSED: partial training output without complete registry: {output}")
            resume_checkpoint = output / "weights" / "last.pt"
            if not resume_checkpoint.is_file():
                raise SystemExit(f"TRAINING QUEUE REFUSED: partial training output lacks weights/last.pt: {output}")
        inputs = resolve_job_inputs(root, queue, dataset)
        data_yaml = inputs["data_yaml"]
        acquisition = inputs["acquisition_registry"]
        command = [sys.executable, str(root / "src" / "train_yolo_dataset.py"), "--project-root", str(root),
                   "--profile", str(profile), "--dataset", dataset, "--model", model, "--data-yaml", str(data_yaml),
                   "--acquisition-registry", str(acquisition), "--run-id", run_id, "--registry-out", str(registry)]
        if inputs["split_registry"] is not None:
            command.extend(["--split-registry", str(inputs["split_registry"])])
        if resume_checkpoint is not None:
            command.extend(["--resume-from", str(resume_checkpoint)])
            print(f"TRAIN QUEUE {ordinal}/{len(jobs)} resuming {run_id}", flush=True)
        else:
            print(f"TRAIN QUEUE {ordinal}/{len(jobs)} starting {run_id}", flush=True)
        subprocess.run(command, check=True)
        if not complete_registry(registry):
            raise SystemExit(f"TRAINING QUEUE REFUSED: completed process lacks valid registry: {run_id}")
        compaction_sha = compact_if_requested(root, queue, registry, run_id)
        records.append({"run_id": run_id, "registry_sha256": sha256_file(registry),
                        "compaction_report_sha256": compaction_sha, "state": "completed_by_queue"})
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "queue": str(queue_path), "queue_sha256": sha256_file(queue_path), "jobs": records}
    document["queue_report_sha256"] = canonical_hash(document, "queue_report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"TRAINING QUEUE COMPLETE jobs={len(records)} report={report}")


if __name__ == "__main__":
    main()
