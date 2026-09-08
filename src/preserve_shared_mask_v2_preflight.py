#!/usr/bin/env python3
"""Preserve the exact V2 preflight source package before V3 source sync.

This transition tool is intentionally standalone: it can be copied to the
remote project without replacing any file named in V2's execution package.
It refuses to archive if V2 produced a progress/scientific artifact or if a
V2 driver/supervisor is still alive.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


PACKAGE_FILE_SHA256 = "cceeb271bc4755a5ce15008a3e2ae08187ab4ecad506eed908a8d32d86398b62"
PACKAGE_CANONICAL_SHA256 = "db8d1312b1d80fea9610c1ce1a1511b3c103c366859cea7f4690bdce45ff556f"
CONFIG_FILE_SHA256 = "7fc84968392615292932a59bb246bfc256f1ef74465047f1106c3902c080ade9"
CONFIG_CANONICAL_SHA256 = "b6c9e114a2fffbb2e220b0798c264d3aa834f105accdf0e042ad932450ac6427"

ABSENT_PATHS = (
    "outputs/reports/shared_mask_pilot_v2/progress.json",
    "outputs/reports/shared_mask_pilot_v2/complete.json",
    "manifests/quantization_masks/shared_mask_pilot_v2",
    "outputs/onnx/shared_mask_pilot_v2",
    "manifests/onnx/shared_mask_pilot_v2",
    "outputs/engines/shared_mask_pilot_v2",
    "manifests/engines/shared_mask_pilot_v2",
    "outputs/predictions/shared_mask_pilot_v2",
    "outputs/inputs/shared_mask_pilot_v2",
    "manifests/runs/shared_mask_pilot_v2",
    "outputs/metrics/shared_mask_pilot_v2",
    "outputs/bootstrap/shared_mask_pilot_v2",
    "outputs/analysis/shared_mask_pilot_v2",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )


def read_complete(path: Path, field: str) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    document = json.loads(path.read_text(encoding="utf-8"))
    declared = document.get(field)
    if (
        not marker.is_file()
        or not isinstance(declared, str)
        or declared != canonical_hash(document, field)
        or marker.read_text(encoding="utf-8").strip() != declared
    ):
        raise RuntimeError(f"invalid complete artifact: {path}")
    return document


def pid_identity_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except (OSError, ValueError):
        return False
    return b"shared_mask_pilot" in command


def archive_bytes(root: Path, members: dict[str, str]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative, expected_sha in sorted(members.items()):
                source = (root / relative).resolve()
                if root != source and root not in source.parents:
                    raise RuntimeError(f"V2 package path escapes project root: {relative}")
                if not source.is_file() or sha256_file(source) != expected_sha:
                    raise RuntimeError(f"V2 execution source changed: {relative}")
                data = source.read_bytes()
                info = tarfile.TarInfo(name=relative)
                info.size = len(data)
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def write_complete(path: Path, document: dict, field: str) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if path.exists() or marker.exists():
        raise RuntimeError(f"refusing to overwrite preservation evidence: {path}")
    value = dict(document)
    value[field] = canonical_hash(value, field)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    marker.write_text(value[field] + "\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    destination = root / "outputs/quarantine/shared_mask_pilot_v2_preflight_abandoned"
    destination.mkdir(parents=True, exist_ok=True)
    lock_handle = (destination / "preservation.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    for name in ("driver", "supervisor"):
        pid_path = root / "outputs/logs/shared_mask_pilot_v2" / f"{name}.pid"
        if pid_identity_alive(pid_path):
            raise SystemExit(f"V2 PREFLIGHT PRESERVATION REFUSED: active {name}")
    present = [relative for relative in ABSENT_PATHS if (root / relative).exists()]
    if present:
        raise SystemExit(
            "V2 PREFLIGHT PRESERVATION REFUSED: V2 is not preflight-only: " + ", ".join(present)
        )

    package_path = root / "outputs/reports/shared_mask_pilot_v2/execution_package.json"
    config_path = root / "configs/shared_mask_pilot_v2.json"
    if sha256_file(package_path) != PACKAGE_FILE_SHA256:
        raise SystemExit("V2 PREFLIGHT PRESERVATION REFUSED: package file hash changed")
    package = read_complete(package_path, "package_sha256")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        package.get("package_sha256") != PACKAGE_CANONICAL_SHA256
        or package.get("attempt") != "shared_mask_pilot_v2"
        or sha256_file(config_path) != CONFIG_FILE_SHA256
        or config.get("config_sha256") != CONFIG_CANONICAL_SHA256
        or canonical_hash(config, "config_sha256") != CONFIG_CANONICAL_SHA256
    ):
        raise SystemExit("V2 PREFLIGHT PRESERVATION REFUSED: package/config identity changed")
    members = package.get("files_sha256")
    if not isinstance(members, dict) or not members:
        raise SystemExit("V2 PREFLIGHT PRESERVATION REFUSED: package has no source ledger")

    archive_path = destination / "execution_sources.tar.gz"
    report_path = destination / "preservation_report.json"
    if archive_path.exists() or report_path.exists():
        raise SystemExit("V2 PREFLIGHT PRESERVATION REFUSED: destination already exists")
    payload = archive_bytes(root, members)
    temporary = destination / f".execution_sources.{os.getpid()}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, archive_path)
    report = write_complete(
        report_path,
        {
            "schema_version": 1,
            "attempt": "shared_mask_pilot_v2",
            "status": "preflight_sources_preserved",
            "admissible_as_scientific_result": False,
            "execution_package_file_sha256": PACKAGE_FILE_SHA256,
            "execution_package_canonical_sha256": PACKAGE_CANONICAL_SHA256,
            "config_file_sha256": CONFIG_FILE_SHA256,
            "config_canonical_sha256": CONFIG_CANONICAL_SHA256,
            "archive": str(archive_path.relative_to(root)),
            "archive_sha256": sha256_file(archive_path),
            "member_count": len(members),
            "members_sha256": dict(sorted(members.items())),
            "progress_artifact_absent": True,
            "scientific_artifacts_absent": True,
            "checked_absent_paths": list(ABSENT_PATHS),
        },
        "report_sha256",
    )
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "archive_sha256": report["archive_sha256"],
                "member_count": report["member_count"],
                "report": str(report_path),
                "report_file_sha256": sha256_file(report_path),
                "report_sha256": report["report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
