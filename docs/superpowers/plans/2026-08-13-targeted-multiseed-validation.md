# Targeted Multi-Seed Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an append-only 3x3 training/calibration seed validation for high-severity INT8--FP8 corruption interaction on YOLO11m transfer models.

**Architecture:** A self-hashed configuration drives an idempotent stage runner. Existing single-job training/export/quantization/engine/inference/evaluation programs remain the execution leaves; the new runner derives exact commands, validates completed leaves before resume, writes an immutable ledger, and produces a separately validated sensitivity report.

**Tech Stack:** Python 3.11, Ultralytics, PyTorch, ONNX, NVIDIA ModelOpt, TensorRT, COCOeval, JSON/SHA-256 ledgers, pytest, Bash/SSH for launch only.

## Global Constraints

- Attempt ID is `ivc_multiseed_yolo11m_s5_v1` and is never overwritten.
- Training seeds and calibration seeds are exactly `[20260807, 20260813, 20260814]`.
- Datasets are exactly VOC, KITTI, and TT100K; model is exactly YOLO11m.
- Fixed batch is 16 for VOC/KITTI and 4 for TT100K.
- Conditions are JPEG-95 matched-clean plus four severity-5 corruptions.
- Expected counts are 6 new trainings, 9 FP32 ONNX, 54 quantized ONNX/engines, 270 inference/metric records, and 108 direct cells.
- No manuscript result is generated before the completion report validates.
- Remote launch requires at least 20 GiB free and no unapproved GPU compute process.

---

### Task 1: Frozen configuration and command derivation

**Files:**
- Create: `configs/ivc_multiseed_yolo11m_s5_v1.json`
- Create: `src/run_multiseed_validation.py`
- Create: `tests/test_multiseed_validation.py`

**Interfaces:**
- Produces: `load_config(Path, Path) -> dict`, `derive_jobs(dict, Path) -> dict[str, list[dict]]`, and `preflight(dict, Path) -> dict`.

- [ ] Write a failing test with a hand-authored configuration fixture requiring the exact 6/9/54/270/108 counts and fixed batch mapping.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q tests/test_multiseed_validation.py::test_exact_frozen_factorial_grid` and confirm failure because the runner is absent.
- [ ] Implement strict schema/type/value checks, canonical self-hash validation, path containment, and deterministic condition IDs.
- [ ] Add failing mutation tests for a fourth seed, auto batch, non-severity-5 condition, duplicate ID, path escape, and altered canonical hash; implement the minimum validation to reject each.
- [ ] Run the focused config/derivation tests and confirm they pass.

### Task 2: Resume-verified training and calibration stages

**Files:**
- Modify: `src/run_multiseed_validation.py`
- Modify: `src/train_yolo_dataset.py`
- Test: `tests/test_multiseed_validation.py`

**Interfaces:**
- Consumes: exact training and calibration jobs from Task 1.
- Produces: six immutable training registries and six new plus three existing calibration-manifest bindings.

- [ ] Write failing tests that require dataset-specific fixed batch, reject a mismatched completed registry, and preserve a valid completed job.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Allow a profile's batch mapping to resolve by dataset while preserving scalar legacy profiles; record the resolved batch in the training registry.
- [ ] Implement serial subprocess stages with append-only per-job logs and exact complete-registry checks.
- [ ] Run the focused tests and the existing training/calibration tests.

### Task 3: Export, quantization, and engine stages

**Files:**
- Modify: `src/run_multiseed_validation.py`
- Test: `tests/test_multiseed_validation.py`

**Interfaces:**
- Consumes: nine complete training registries and nine calibration manifests.
- Produces: nine FP32 ONNX registries and 54 INT8/FP8 ONNX plus engine registries.

- [ ] Write failing command tests with literal expected paths and arguments for one baseline and one new-seed combination.
- [ ] Write failing resume tests that mutate source-checkpoint, calibration, ONNX, or engine hashes.
- [ ] Implement serial leaf commands using `export_yolo_onnx.py`, `quantize_yolo_onnx.py`, and `build_yolo_trt_engine.py`.
- [ ] Validate graph/registry treatment fields and exact 9/54 counts before advancing.
- [ ] Run focused and existing export/calibration/engine tests.

### Task 4: Inference and metrics stages

**Files:**
- Modify: `src/run_multiseed_validation.py`
- Test: `tests/test_multiseed_validation.py`

**Interfaces:**
- Consumes: 54 engines and 15 existing clean/corruption input manifests.
- Produces: 270 prediction/input/run-record triples and 270 metric records.

- [ ] Write failing tests for exact 270 conditions, clean-arm reuse, dataset-specific evaluator selection, and an input-manifest hash mismatch.
- [ ] Implement commands around `coco_infer_trt.py`, `validate_predictions.py`, and the frozen COCO/TT100K evaluators.
- [ ] Implement strict resume validation for prediction SHA, ordered IDs, manifest SHA, engine SHA, calibration SHA, run-record SHA, and metric semantics.
- [ ] Add a failure-ledger test proving one failed child stops new launches without removing valid completed outputs.
- [ ] Run focused and existing inference/evaluation tests.

### Task 5: Sensitivity analysis and completion chain

**Files:**
- Create: `src/analyze_multiseed_validation.py`
- Modify: `src/run_multiseed_validation.py`
- Test: `tests/test_multiseed_validation.py`

**Interfaces:**
- Consumes: the exact 270 validated metric records.
- Produces: 108-cell CSV, marginal CSV, variance-decomposition CSV, JSON report, and self-hashed completion report.

- [ ] Write a failing synthetic 3x3 test with hand-derived \(D\), \(\Delta E\), marginal means, and two-way sums of squares.
- [ ] Implement metric binding and balanced descriptive analysis without model fitting or p-values.
- [ ] Add mutation tests for a missing cell, duplicate cell, changed AP, wrong metric hash, and false completion hash.
- [ ] Run the analysis tests and full local suite.

### Task 6: Remote preflight and persistent launch

**Files:**
- Create remotely only after validation: `outputs/logs/ivc_multiseed_yolo11m_s5_v1.launch.log`
- Create remotely only after validation: `outputs/work/ivc_multiseed_yolo11m_s5_v1/`

**Interfaces:**
- Consumes: reviewed local source/config hashes synchronized to `/home/thuan/topic_c_ivc`.
- Produces: one persistent serial controller PID and a user-readable log.

- [ ] Run all local tests, `py_compile`, and `git diff --check`.
- [ ] Compare local/remote SHA-256 for every synchronized source and config file.
- [ ] Run remote `--preflight` under `conda activate qtsd`; require RTX 5090, >=20 GiB free disk, exact package versions, no conflicting attempt bytes, and no unapproved GPU PID.
- [ ] Launch with `setsid nohup` only when preflight passes, record PID/command/log, and confirm the controller survives SSH disconnect.
- [ ] Give the user `tail -f /home/thuan/topic_c_ivc/outputs/logs/ivc_multiseed_yolo11m_s5_v1.launch.log` and a structured progress command.

## Plan self-review

Every frozen design requirement maps to an implementation/test task. Counts,
seeds, condition names, paths, and stage interfaces are consistent. The plan has
no unresolved placeholder, and the remote launch is explicitly gated from the
local implementation and scientific completion stages.
