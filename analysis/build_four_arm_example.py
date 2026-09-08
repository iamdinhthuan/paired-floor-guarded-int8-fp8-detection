#!/usr/bin/env python3
"""Package one full fixed-treatment primary cell; no training or image files."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from topic_c.manifest import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite example: {out}")
    reference = root / "outputs/bootstrap/ivc_format_contrast_v1/kitti__yolo11m__fog-s1.json"
    expected = json.loads(reference.read_text())
    out.mkdir(parents=True)
    mappings = {}
    for arm in ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt"):
        precision, condition = arm.split("_")
        attempt = "codec_control_p0_v1" if condition == "clean" else "kitti_pilot_117_v1"
        token = "codec-control-s0" if condition == "clean" else "fog-s1"
        candidates = sorted((root / "manifests/runs" / attempt).glob(f"kitti_val__yolo11m__{precision}*__{token}__*.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"ambiguous run for {arm}")
        run_path = candidates[0]
        run = json.loads(run_path.read_text())
        if sha256_file(run_path) != expected["input_hashes"][arm]["run_record_sha256"]:
            raise RuntimeError(f"run does not bind to primary cell: {arm}")
        paths = dict(run=run_path, input=root / "outputs/inputs" / attempt / run_path.name,
                     prediction=root / "outputs/predictions" / attempt / run_path.name)
        if sha256_file(paths["prediction"]) != expected["input_hashes"][arm]["prediction_sha256"]:
            raise RuntimeError(f"prediction does not bind to primary cell: {arm}")
        mappings[arm] = {}
        for kind, source in paths.items():
            dest = out / "data" / f"{arm}_{kind}.json"
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(source, dest)
            mappings[arm][kind] = str(dest.relative_to(out))
    for precision in ("int8", "fp8"):
        runs = [json.loads((out / mappings[f"{precision}_{state}"]["run"]).read_text()) for state in ("clean", "corrupt")]
        if any(runs[0][key] != runs[1][key] for key in ("engine_sha256", "preprocess_sha256", "decoder_sha256", "class_map_sha256")):
            raise RuntimeError("example clean/corrupt treatments differ")
    annotation = root / "manifests/annotations/kitti_val_ultralytics_v1_coco.json"
    if sha256_file(annotation) != expected["annotation"]["sha256"]:
        raise RuntimeError("example annotation differs from primary cell")
    shutil.copy2(annotation, out / "data/annotations.json")
    shutil.copy2(reference, out / "expected.json")
    for name in ("bootstrap_format_contrast.py", "paired_bootstrap.py", "topic_c/__init__.py", "topic_c/manifest.py", "topic_c/tt100k_height.py"):
        target = out / "src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "src" / name, target)
    shutil.copy2(root / "analysis/reproduce_four_arm_example.py", out / "reproduce.py")
    (out / "requirements.txt").write_text("numpy==2.4.4\npycocotools==2.0.11\n")
    (out / "README.md").write_text(
        "# Full four-arm reproduction example\n\n"
        "KITTI historical evaluation split, YOLO11m, fog severity 1; 1,496 images.\n"
        "This convenience case is not a newly selected scientific endpoint.\n"
        "Four full prediction payloads, converted annotations and immutable run/input\n"
        "records reproduce primary AP and 2,000-draw intervals without engines or GPUs.\n"
        "This tests evaluation, not detector-inference regeneration or all 144 cells.\n\n"
        "Python 3.11 on Linux:\n\n```bash\npython -m pip install -r requirements.txt\n"
        "python reproduce.py --out /tmp/kitti_example_result.json --workers 8\n```\n\n"
        "Use a fresh output filename on reruns. All input hashes are checked.\n"
        "Runtime depends on CPU/resources. Results are compared at absolute tolerance\n"
        "1e-10 on the 0–1 AP scale. No cached AP draws are used in this recomputation.\n\n"
        "Original project code retains its MIT license. KITTI-derived annotations\n"
        "remain subject to the original dataset terms, not the project's MIT license.\n"
        "This local package is not a public release; verify redistribution terms\n"
        "and author approval before depositing its dataset-derived payloads.\n"
    )
    manifest = dict(schema_version=1, seed=expected["seed"], n_images=1496,
                    annotations="data/annotations.json", arms=mappings,
                    versions={"numpy": "2.4.4", "pycocotools": "2.0.11"},
                    files={str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()})
    (out / "example.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
