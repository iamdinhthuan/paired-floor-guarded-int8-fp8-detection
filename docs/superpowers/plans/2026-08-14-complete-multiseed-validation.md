# Complete Multi-seed Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the exact TensorRT 11.1.0.106 runtime on the RTX 5090 host, complete the hash-bound multi-seed validation chain, and integrate only validated multi-seed evidence into the IVC manuscript and Overleaf package.

**Architecture:** Preserve the existing attempt `ivc_multiseed_yolo11m_s5_v1` and its completed training/calibration/export/quantization artifacts. Add an early runtime-layout/identity preflight to the multi-seed controller, restore NVIDIA's CUDA-13.3 TensorRT archive at the already frozen `trt_root`, and resume at the first missing engine. Engine, inference, metric, analysis, and manuscript stages remain fail-closed and are admitted only by their completion reports and SHA-256 bindings.

**Tech Stack:** Python 3.11, pytest, NVIDIA TensorRT 11.1.0.106/CUDA 13.3, NVIDIA ModelOpt, Ultralytics, COCO-style evaluation, LaTeX/Elsevier CAS, SHA-256, SSH/rsync.

## Global Constraints

- Run GPU work only as `thuan@100.111.139.103` under conda environment `qtsd`.
- Do not overwrite completed immutable artifacts; resume only through existing validators.
- Restore exactly TensorRT 11.1.0.106 for CUDA 13.3 and compare `trtexec` SHA-256 with the prior validated value `68b061d276601b7c6d8aafa4d8d75319f32382c85673360407a6d7ee6411aa4d`.
- Keep all manuscript citations in 2023--2026 and exactly five Image and Vision Computing entries.
- Do not add multi-seed claims until a complete, locally revalidated evidence chain exists.
- Never infer Funding, DOI, author, or archive facts.

---

### Task 1: Fail-fast TensorRT runtime preflight

**Files:**
- Modify: `src/run_multiseed_validation.py`
- Test: `tests/test_multiseed_validation.py`

**Interfaces:**
- Consumes: `config["trt_root"]`, expected `bin/trtexec` and `lib/` layout.
- Produces: `validate_trt_runtime(config: dict) -> dict` with resolved path, executable SHA-256, and runtime layout identity; called before any build-stage job is derived or executed.

- [ ] Write a test that creates a missing/incomplete `trt_root` and expects a fail-closed error before command execution.
- [ ] Run the focused test and observe the expected RED failure because no early preflight exists.
- [ ] Implement the minimal validator and call it at controller startup.
- [ ] Add a positive fixture with executable `bin/trtexec` and non-empty `lib/`; assert deterministic SHA identity.
- [ ] Run the focused and full non-completion suites.

### Task 2: Restore and verify the frozen TensorRT runtime

**Files:**
- Create remotely: `/home/thuan/traffic/third_party/TensorRT-11.1.0.106/`
- Create remotely: a download log and archive SHA record under `/home/thuan/topic_c_ivc/outputs/logs/`

**Interfaces:**
- Consumes: NVIDIA archive `TensorRT-Enterprise-11.1.0.106-Linux-x86_64-cuda-13.3-Release-external.tar.zst`.
- Produces: executable `bin/trtexec`, runtime `lib/`, version output, archive digest, and executable digest.

- [ ] Download the archive to a `mktemp` path on the remote host without replacing any target.
- [ ] Record archive size and SHA-256, list its top-level layout, and extract into a temporary sibling directory.
- [ ] Verify `trtexec --help` under the extracted `lib/`, TensorRT version 11.1.0.106, and executable digest against the historical registry digest.
- [ ] Atomically rename the verified directory to the frozen `trt_root`; refuse if the target unexpectedly exists.
- [ ] Run one bounded engine-build probe through the real builder and validate its registry before launching the controller.

### Task 3: Resume the multi-seed evidence chain

**Files:**
- Remote outputs under the existing attempt directories only.
- Completion reports under `outputs/reports/`.

**Interfaces:**
- Consumes: 9 FP32 and 54 quantized ONNX artifacts already validated.
- Produces: 54 TensorRT engines, inference/run records, metrics, cell analyses, and the attempt completion report.

- [ ] Run controller preflight/dry-run and verify completed quantization is reused rather than overwritten.
- [ ] Launch the resumable controller in a persistent background session with a dedicated log and PID record.
- [ ] Validate all 54 engine registries and hashes after the engine phase.
- [ ] Launch/continue inference and metric phases only after engine completion validates.
- [ ] Run the multi-seed analysis and require its exact grid, seed identities, finite values, and completion-report hashes.
- [ ] Synchronize completion/evidence reports to the local workspace and independently rerun the validator.

### Task 4: Evidence-gated manuscript integration

**Files:**
- Modify: `analysis/build_paper_artifacts.py`
- Modify: `paper/main.tex`, direct templates, highlights only if supported by completed evidence.
- Test: `tests/test_build_paper_artifacts.py`, `tests/test_submission_metadata.py`

**Interfaces:**
- Consumes: validated multi-seed completion report and generated summary artifact.
- Produces: concise seed-sensitivity Results/Limitations text, table or interval summary, audit-bound generated assets.

- [ ] Write RED tests that reject missing, incomplete, rehashed, or wrong-grid multi-seed evidence.
- [ ] Implement the multi-seed loader/validator and generated summary artifacts.
- [ ] Integrate the result as a sensitivity analysis without replacing the primary 144-cell direct estimand.
- [ ] Update the single-seed limitation according to the actual replicated scope; do not generalize beyond YOLO11m/severity-5.
- [ ] Regenerate the direct audit so every new manuscript asset is SHA-bound.

### Task 5: Final verification and hand-off

**Files:**
- Rebuild: `paper/main.pdf`, `paper/supplement.pdf`.
- Create: a new versioned `paper/overleaf_upload_ivc_*` directory and ZIP.

**Interfaces:**
- Consumes: final source and validated generated artifacts.
- Produces: independently compilable submission package.

- [ ] Run the full pytest suite, direct evidence regeneration, LaTeX builds, submission audit, and `git diff --check`.
- [ ] Inspect every page affected by the new sensitivity result and reject clipped/late floats.
- [ ] Request a final evidence-focused Claude Code review and resolve only verified defects.
- [ ] Build a new non-overwriting Overleaf folder/ZIP, verify its manifest, and compile main and supplement after extraction.
