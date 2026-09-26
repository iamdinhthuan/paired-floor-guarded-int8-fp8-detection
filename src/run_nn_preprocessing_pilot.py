from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from topic_c.manifest import read_manifest, sha256_file, under_root


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def canonical(document, field):
    payload = {k: v for k, v in document.items() if k != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibration_subset(source, count, seed, parent_hash):
    records = source["records"]
    if source.get("split") != "train":
        raise ValueError("calibration must be train-only")
    if not 0 < count <= len(records) or source["n_images"] != len(records):
        raise ValueError("invalid calibration subset size")
    if len({r["source_relpath"] for r in records}) != len(records):
        raise ValueError("duplicate calibration images")
    indices = sorted(np.random.default_rng(seed).choice(len(records), count, replace=False).tolist())
    result = copy.deepcopy(source)
    result.update(n_images=count, seed=seed, records=[records[i] for i in indices],
                  selection="uniform_without_replacement_from_frozen_training_calibration_pool",
                  parent_manifest_sha256=parent_hash, parent_indices=indices)
    result["calibration_sha256"] = canonical(result, "calibration_sha256")
    return result


def treatment_quantize_options(treatment, root):
    args = []
    if treatment.get("quantize_op_types"):
        op_types = treatment["quantize_op_types"]
        args += ["--quantize-op-types", ",".join(op_types) if isinstance(op_types, list) else str(op_types)]
    if treatment.get("quantize_node_regex"):
        regexes = treatment["quantize_node_regex"]
        args += ["--quantize-node-regex", ",".join(regexes) if isinstance(regexes, list) else str(regexes)]
    if treatment.get("exclude_node_regex"):
        regexes = treatment["exclude_node_regex"]
        args += ["--exclude-node-regex", ",".join(regexes) if isinstance(regexes, list) else str(regexes)]
    if treatment.get("node_mask"):
        mask_path = root / treatment["node_mask"]
        check_hash(mask_path, treatment["node_mask_sha256"])
        args += ["--node-mask", str(mask_path)]
    if treatment["preprocessing"] == "yolo_letterbox":
        args.append("--allow-preprocessing-mismatch")
    return args


def clean_gate(values, policy):
    required = {"fp32", "int8_legacy_calibration", "int8_matched_calibration", "fp8_matched_calibration"}
    if set(values) != required or not all(np.isfinite(v) and 0 <= v <= 100 for v in values.values()):
        raise ValueError("four finite AP-point values required")
    losses = {k: values["fp32"] - values[k] for k in ("int8_matched_calibration", "fp8_matched_calibration")}
    return {"passed": bool(values["fp32"] >= policy["minimum_reference_ap_points"]
                           and max(losses.values()) <= policy["maximum_clean_reference_loss_ap_points"]),
            "reference_losses_ap_points": losses,
            "int8_matched_minus_legacy_ap_points": values["int8_matched_calibration"] - values["int8_legacy_calibration"],
            "interpretation": "Engineering continuation gate only; no statistical significance or new-method claim."}


def check_hash(path, expected):
    if not Path(path).is_file() or sha256_file(path) != expected:
        raise ValueError(f"missing or changed input: {path}")


def check_marker(path, expected):
    if Path(str(path) + ".complete").read_text().strip() != expected:
        raise ValueError(f"completion marker mismatch: {path}")


def code_hashes():
    source = Path(__file__).resolve().parent
    return {str(p): sha256_file(p) for p in sorted(source.rglob("*.py"))}


def prepare(root, config_path, out):
    config = read_json(config_path)
    if (config["dataset"], config["model"], config["imgsz"]) != ("kitti", "retinanet-r50-fpn-v2", 640):
        raise ValueError("this diagnostic is frozen to KITTI RetinaNet at 640")
    if config["execution"]["gpu_workers"] != 1 or config["build"]["tf32"]:
        raise ValueError("serial, TF32-off execution required")
    files = {str(config_path): sha256_file(config_path)}
    for field in ("source_registry", "calibration_manifest", "annotations", "clean_manifest"):
        path = root / config[field]
        check_hash(path, config[field + "_sha256"])
        files[str(path)] = sha256_file(path)
    source_path = root / config["source_registry"]
    source = read_json(source_path)
    check_marker(source_path, sha256_file(source_path))
    if source.get("decoder") != "torchvision_retinanet_raw_v1":
        raise ValueError("wrong source decoder")
    check_hash(source["onnx"], source["onnx_sha256"])
    files[source["onnx"]] = source["onnx_sha256"]
    parent_path = root / config["calibration_manifest"]
    parent = read_json(parent_path)
    if parent["calibration_sha256"] != canonical(parent, "calibration_sha256"):
        raise ValueError("invalid parent calibration digest")
    check_marker(parent_path, parent["calibration_sha256"])
    selected = calibration_subset(parent, config["calibration_images"], config["calibration_selection_seed"], sha256_file(parent_path))
    selected["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    selected["calibration_sha256"] = canonical(selected, "calibration_sha256")
    for item in selected["records"]:
        path = under_root(selected["dataset_root"], item["source_relpath"])
        check_hash(path, item["sha256"])
        files[str(path)] = item["sha256"]
    manifest_path = root / config["clean_manifest"]
    manifest = read_manifest(manifest_path)
    check_marker(manifest_path, manifest["manifest_sha256"])
    annotations = read_json(root / config["annotations"])
    ids = sorted(i["id"] for i in annotations["images"])
    if len(ids) != config["expected_images"] or ids != manifest["expected_image_ids"]:
        raise ValueError("clean image universe mismatch")
    if [r["image_id"] for r in manifest["records"]] != ids:
        raise ValueError("clean manifest ordering mismatch")
    for item in manifest["records"]:
        path = under_root(root / config["clean_cache_root"], item["output_relpath"])
        check_hash(path, item["sha256"])
        files[str(path)] = item["sha256"]
    from topic_c.coco_data import preprocess
    from topic_c.cross_family import preprocess_retinanet
    sample = under_root(selected["dataset_root"], selected["records"][0]["source_relpath"])
    legacy, matched = preprocess(str(sample), config["imgsz"])[0], preprocess_retinanet(sample, config["imgsz"])[0]
    diagnostic = {"sample_sha256": sha256_file(sample), "legacy_range": [float(legacy.min()), float(legacy.max())],
                  "matched_range": [float(matched.min()), float(matched.max())],
                  "max_absolute_tensor_difference": float(np.abs(legacy - matched).max()),
                  "identical": bool(np.array_equal(legacy, matched))}
    import tensorrt as trt
    if trt.__version__ != config["build"]["required_version"]:
        raise ValueError("TensorRT version differs from frozen plan")
    packages = {p: importlib.metadata.version(p) for p in ("numpy", "onnx", "torch", "torchvision", "nvidia-modelopt", "pycocotools")}
    calibration_path = out / "calibration.json"
    write_once(calibration_path, selected)
    with Path(str(calibration_path) + ".complete").open("x") as stream:
        stream.write(selected["calibration_sha256"] + "\n")
    files[str(calibration_path)] = sha256_file(calibration_path)
    registry = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "config_path": str(config_path), "project_root": str(root), "output_root": str(out),
                "source_registry": str(source_path), "calibration": str(calibration_path),
                "source_files": code_hashes(), "input_files": files, "packages": packages,
                "python": platform.python_version(), "tensorrt": trt.__version__,
                "preprocessing_diagnostic": diagnostic, "status": "prepared_not_executed"}
    write_once(out / "registry.json", registry)
    write_once(out / "registry.complete.json", {"registry_sha256": sha256_file(out / "registry.json")})
    print(json.dumps({"prepared": str(out), "diagnostic": diagnostic, "packages": packages}, indent=2), flush=True)


def resource_gate(config, root):
    policy = config["execution"]
    if shutil.disk_usage(root).free < policy["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("insufficient free disk; no existing files will be deleted")
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True)
    if int(result.stdout.strip().splitlines()[0]) < policy["minimum_free_gpu_mib"]:
        raise RuntimeError("insufficient free GPU memory; no unrelated process will be stopped")


def build_engine(registry_path, precision, destination, policy):
    import tensorrt as trt

    record = read_json(registry_path)
    check_marker(registry_path, sha256_file(registry_path))
    source = Path(record.get("output_onnx", record.get("onnx", "")))
    expected = record.get("output_onnx_sha256", record.get("onnx_sha256"))
    check_hash(source, expected)
    engine_path = destination / "model.engine"
    if engine_path.exists() or (destination / "engine.json").exists():
        raise FileExistsError("engine output already exists")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(source)):
        raise RuntimeError("ONNX parse failed: " + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    settings = builder.create_builder_config()
    settings.clear_flag(trt.BuilderFlag.TF32)
    settings.builder_optimization_level = policy["optimization_level"]
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, policy["workspace_mib"] * 1024**2)
    settings.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    payload = builder.build_serialized_network(network, settings)
    if payload is None:
        raise RuntimeError("TensorRT build failed")
    with engine_path.open("xb") as stream:
        stream.write(bytes(payload))
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(bytes(payload))
        if engine is None or engine.num_io_tensors != 1 + len(record["output_names"]):
            raise RuntimeError("built engine IO mismatch")
        inspector = engine.create_engine_inspector()
        with (destination / "engine_inspector.json").open("x") as stream:
            stream.write(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    result = {k: record[k] for k in ("dataset", "model", "imgsz", "input_names", "output_names", "decoder")}
    result.update(precision=precision, engine=str(engine_path), engine_sha256=sha256_file(engine_path),
                  source_onnx_registry_sha256=sha256_file(registry_path), source_onnx_sha256=expected,
                  build_policy=policy, builder_source_sha256=sha256_file(__file__), tensorrt_version=trt.__version__,
                  calibration_sha256=record.get("calibration_sha256"), latency_evidence="inadmissible_shared_gpu")
    path = destination / "engine.json"
    write_once(path, result)
    with Path(str(path) + ".complete").open("x") as stream:
        stream.write(sha256_file(path) + "\n")
    return path


def run_child(command, log_path, environment):
    with log_path.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)


def execute(out):
    registry_path = out / "registry.json"
    check_hash(registry_path, read_json(out / "registry.complete.json")["registry_sha256"])
    registry = read_json(registry_path)
    for path, digest in {**registry["source_files"], **registry["input_files"]}.items():
        check_hash(path, digest)
    config, root = registry["config"], Path(registry["project_root"])
    source_dir = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source_dir))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(config["execution"]["cpu_threads"])
    rows = []
    for treatment in config["treatments"]:
        name = treatment["id"]
        destination = out / name
        destination.mkdir(exist_ok=False)
        print(f"START {name}", flush=True)
        resource_gate(config, root)
        onnx_registry = Path(registry["source_registry"])
        if treatment["precision"] != "fp32":
            onnx_registry = destination / "onnx.json"
            calibration_list = registry["calibration"]
            if treatment.get("calibration_list"):
                override = root / treatment["calibration_list"]
                check_hash(override, treatment["calibration_list_sha256"])
                calibration_list = str(override)
            command = [sys.executable, str(source_dir / "quantize_yolo_onnx.py"),
                       "--onnx-registry", registry["source_registry"], "--mode", treatment["precision"],
                       "--imgsz", str(config["imgsz"]), "--calibration-list", calibration_list,
                       "--calibration-preprocessing", treatment["preprocessing"],
                       "--out", str(destination / "model.onnx"), "--registry-out", str(onnx_registry)]
            command += treatment_quantize_options(treatment, root)
            run_child(command, destination / "quantize.log", env)
        resource_gate(config, root)
        command = [sys.executable, str(Path(__file__).resolve()), "--config", registry["config_path"],
                   "--project-root", str(root), "--build-one", str(onnx_registry),
                   "--precision", treatment["precision"], "--destination", str(destination)]
        run_child(command, destination / "build.log", env)
        engine_registry = destination / "engine.json"
        resource_gate(config, root)
        prediction, inputs, run, metric = [destination / filename for filename in
                                          ("predictions.json", "inputs.json", "run.json", "metric.json")]
        command = [sys.executable, str(source_dir / "cross_family_infer_trt.py"),
                   "--engine-registry", str(engine_registry), "--annotations", str(root / config["annotations"]),
                   "--image-manifest", str(root / config["clean_manifest"]),
                   "--manifest-cache-root", str(root / config["clean_cache_root"]),
                   "--out", str(prediction), "--input-record", str(inputs), "--run-record", str(run),
                   "--condition-id", config["attempt"] + "__" + name, "--dataset", config["dataset"],
                   "--split", config["split"], "--corruption", "codec-control", "--severity", "0", "--confidence", "0.05"]
        run_child(command, destination / "inference.log", env)
        command = [sys.executable, str(source_dir / "coco_eval.py"),
                   "--annotations", str(root / config["annotations"]), "--predictions", str(prediction),
                   "--input-record", str(inputs), "--run-record", str(run), "--out", str(metric)]
        run_child(command, destination / "evaluation.log", env)
        result = read_json(metric)
        if result["n_images"] != config["expected_images"]:
            raise ValueError("evaluated image count differs from frozen plan")
        row = {"treatment": name, "ap_points": result["stats"]["AP"] * 100,
               "metric_sha256": sha256_file(metric), "input_image_ids_sha256": result["input_image_ids_sha256"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if len({row["input_image_ids_sha256"] for row in rows}) != 1:
        raise ValueError("treatment image pairing failed")
    values = {r["treatment"]: r["ap_points"] for r in rows}
    scope = config.get("summary_scope") or (
        "four-treatment clean-only development diagnostic" if len(rows) == 4
        else f"{len(rows)}-treatment clean-only development diagnostic"
    )
    summary = {"scope": scope, "rows": rows,
               "gate": clean_gate(values, config["continuation_gate"]) if config.get("continuation_gate") else None,
               "limitations": config["limitations"],
               "latency_evidence": "inadmissible", "registry_sha256": sha256_file(registry_path)}
    write_once(out / "summary.json", summary)
    write_once(out / "complete.json", {"files_sha256": {str(p.relative_to(out)): sha256_file(p)
               for p in sorted(out.rglob("*")) if p.is_file() and p.suffix in {".json", ".onnx", ".engine"}}})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--build-one", type=Path)
    parser.add_argument("--precision")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    config_path, root = args.config.resolve(), args.project_root.resolve()
    config = read_json(config_path)
    attempt = config["attempt"]
    if not attempt.startswith("nn_preprocessing_pilot_") or Path(attempt).name != attempt:
        raise ValueError("invalid isolated attempt name")
    out = root / "outputs" / attempt
    if args.build_one:
        if args.destination is None or args.destination.resolve().parent != out:
            raise ValueError("build output must be inside the isolated attempt")
        build_engine(args.build_one, args.precision, args.destination, config["build"])
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out / "attempt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare(root, config_path, out)
        else:
            execute(out)


if __name__ == "__main__":
    main()
