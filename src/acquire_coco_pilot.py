from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import read_manifest, sha256_file


BASE = "https://s3.amazonaws.com/images.cocodataset.org"
ANNOTATION_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"


def canonical(document):
    payload = {k: v for k, v in document.items() if k != "calibration_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


def validate_relative(relative):
    match = re.fullmatch(r"images/train2017/(\d{12}\.jpg)", relative)
    if not match:
        raise ValueError(f"noncanonical calibration image path: {relative}")
    return match.group(1)


def checked_publish(partial, destination, expected):
    if expected is not None and sha256_file(partial) != expected:
        raise ValueError(f"downloaded/extracted hash mismatch: {destination}")
    os.link(partial, destination)
    partial.unlink()


def download(url, destination, expected=None, attempts=4):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected is not None and sha256_file(destination) != expected:
            raise ValueError(f"existing download hash mismatch: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(attempts):
        partial.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("xb") as stream:
                total = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    total += len(block)
                if response.headers.get("Content-Length") and total != int(response.headers["Content-Length"]):
                    raise ValueError(f"truncated download: {url}")
            checked_publish(partial, destination, expected)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(30, 2 ** attempt))


def extract_verified(bundle, member, destination, expected):
    destination = Path(destination)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise ValueError(f"existing extracted hash mismatch: {destination}")
        return
    if bundle.getinfo(member).file_size > 256 * 1024**2:
        raise ValueError("unexpectedly large archive member")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with bundle.open(member) as source, partial.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    checked_publish(partial, destination, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.parent.is_dir() or shutil.disk_usage(root.parent).free < 15 * 1024**3:
        raise ValueError("existing parent with at least 15 GiB free required")
    calibration = json.loads(args.calibration.read_text())
    validation = read_manifest(args.validation_manifest)
    if calibration.get("calibration_sha256") != canonical(calibration):
        raise ValueError("calibration manifest hash mismatch")
    if (calibration["dataset"], calibration["split"], calibration["n_images"], len(calibration["records"])) != ("coco", "train", 512, 512):
        raise ValueError("expected exactly 512 frozen COCO train calibration images")
    if (validation["dataset"], validation["split"], len(validation["records"])) != ("coco", "val2017", 5000):
        raise ValueError("expected complete 5000-image COCO validation manifest")
    for item in calibration["records"]:
        validate_relative(item["source_relpath"])
    root.mkdir(exist_ok=args.resume)
    archive_root = root / "archives"
    archive_root.mkdir(exist_ok=args.resume)
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": BASE,
        "calibration_manifest_sha256": sha256_file(args.calibration),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "annotation_sha256": ANNOTATION_SHA256, "training_image_count": 512,
        "validation_image_count": 5000, "full_training_dataset": False,
        "image_terms": "Original COCO and image-source licenses remain applicable; no relicensing.",
        "software_sha256": sha256_file(__file__),
    }
    plan_path = root / "acquisition_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        for key in ("source", "calibration_manifest_sha256", "validation_manifest_sha256",
                    "annotation_sha256", "training_image_count", "validation_image_count"):
            if existing.get(key) != plan[key]:
                raise ValueError(f"resume plan mismatch: {key}")
    else:
        write_json(plan_path, plan)
    archives = {"validation": ("zips/val2017.zip", archive_root / "val2017.zip"),
                "annotations": ("annotations/annotations_trainval2017.zip", archive_root / "annotations_trainval2017.zip")}
    for label, (relative, destination) in archives.items():
        print(f"DOWNLOAD {label}: {BASE}/{relative}", flush=True)
        download(BASE + "/" + relative, destination)
        print(f"DOWNLOADED {label}: {destination.stat().st_size} bytes", flush=True)
    annotation = root / "annotations" / "instances_val2017.json"
    with zipfile.ZipFile(archives["annotations"][1]) as bundle:
        extract_verified(bundle, "annotations/instances_val2017.json", annotation, ANNOTATION_SHA256)
    images = json.loads(annotation.read_text())["images"]
    expected_ids = sorted(item["id"] for item in images)
    if expected_ids != validation["expected_image_ids"]:
        raise ValueError("annotation and image manifest identities disagree")
    digests = {"annotations/instances_val2017.json": ANNOTATION_SHA256}
    with zipfile.ZipFile(archives["validation"][1]) as bundle:
        for index, item in enumerate(validation["records"], 1):
            name = item["source_relpath"]
            if re.fullmatch(r"\d{12}\.jpg", name) is None or int(Path(name).stem) != item["image_id"]:
                raise ValueError("noncanonical validation image identity")
            relative = "images/val2017/" + name
            extract_verified(bundle, "val2017/" + name, root / relative, item["source_sha256"])
            digests[relative] = item["source_sha256"]
            if index % 1000 == 0:
                print(f"VERIFIED validation {index}/5000", flush=True)

    def acquire_train(item):
        name = validate_relative(item["source_relpath"])
        download(BASE + "/train2017/" + name, root / item["source_relpath"], item["sha256"])
        return item["source_relpath"], item["sha256"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (relative, digest) in enumerate(pool.map(acquire_train, calibration["records"]), 1):
            digests[relative] = digest
            if index % 64 == 0:
                print(f"VERIFIED calibration {index}/512", flush=True)
    rebound = copy.deepcopy(calibration)
    rebound.update(dataset_root=str(root), created_at_utc=datetime.now(timezone.utc).isoformat(),
                   parent_manifest_sha256=sha256_file(args.calibration),
                   rebind_reason="original dataset directory unavailable; identical source bytes reacquired")
    rebound["calibration_sha256"] = canonical(rebound)
    write_json(root / "calibration_rebound.json", rebound)
    with (root / "calibration_rebound.json.complete").open("x") as stream:
        stream.write(rebound["calibration_sha256"] + "\n")
    digests["calibration_rebound.json"] = sha256_file(root / "calibration_rebound.json")
    digests["acquisition_plan.json"] = sha256_file(root / "acquisition_plan.json")
    write_json(root / "complete.json", {"dataset_root": str(root), "training_image_count": 512,
               "validation_image_count": 5000, "resumed": args.resume,
               "acquisition_software_sha256": sha256_file(__file__), "files_sha256": digests,
               "archives_sha256": {label: sha256_file(value[1]) for label, value in archives.items()}})
    print(f"COCO PILOT DATA COMPLETE: {root}", flush=True)


if __name__ == "__main__":
    main()
