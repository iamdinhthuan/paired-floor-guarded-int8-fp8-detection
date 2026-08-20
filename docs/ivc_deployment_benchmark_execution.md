# IVC deployment benchmark execution

## Status

**Complete and independently validated.** The RTX 5090 production benchmark
created 108 immutable repetition records and 108 bound raw-log envelopes for
the exact 36-engine-condition grid. The remote completion report was rebuilt
read-only from all records/logs, then the compact evidence (records, logs,
treatment table, and report) was synchronized locally and validated again by
the fail-closed paper validator. No raw prediction payload was synchronized.

This runtime benchmark measures engine size, latency, and throughput only. It
does not support a memory, power, or energy claim.

## Frozen protocol

- Host gate: `thuan@100.111.139.103:/home/thuan/topic_c_ivc`
- Absolute-layout gate: project root `/home/thuan/topic_c_ivc`, engine root
  `/home/thuan/topic_c_ivc/engines`, and `trtexec`
  `/home/thuan/traffic/third_party/TensorRT-11.1.0.106/bin/trtexec`. These are
  hard validated preconditions, not portable defaults.
- Hostname gate: an authorized read-only remote check on 2026-08-12 returned
  `thuan-B860M-EAGLE-WIFI6`; the config records that exact value and continues
  to refuse a mismatched host. The resulting config SHA-256 is the one bound
  into every repetition record.
- Remote preflight: the exact two-file source package was atomically
  synchronized and SHA-256 verified on 2026-08-12 (`benchmark_trt_engines.py`:
  `5f874cf58f2f52caa4045393bd9b1616169ccb0a6bb1ce818a86e3b4a4950b89`;
  config: `8cd7bd35d84fce1194c34a711fcf8a32b88a8285afd237aa30dd5db62eaa4e7f`).
  `--preflight-only` returned `tasks: 108` with no stderr and no output
  directory, records, logs, treatments, or report.
- Environment gate: `conda activate qtsd`
- Required GPU name: `NVIDIA GeForce RTX 5090`
- Grid: COCO, VOC, KITTI, and TT100K; YOLO11n/m/x; FP32,
  INT8-entropy, and FP8 (36 engine conditions)
- Repetitions: exactly three per engine (108 immutable records and 108 raw
  logs)
- Input binding: each engine registry hash-binds its source ONNX registry. The
  runner verifies the source/output digest chain and inspects the exact ONNX
  graph for one input, its name, rank-four shape, and static/dynamic status.
  For dynamic graphs, every fixed condition-graph axis must agree with the
  selected benchmark shape.
- Each `trtexec` invocation: registry-bound executable and engine,
  `--loadEngine`, `--warmUp=5000`, and `--duration=30`; a `--shapes` flag is
  added only when the engine registry explicitly marks it as required
- After the production invocation validates the full runtime identity, but
  before it creates any output parent: run
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader` and refuse any
  foreign compute PID. This gate is repeated before every repetition to cover
  races between the initial gate and later runs.
- Evidence limitations: this protocol supports latency and throughput only.
  It does not support memory, power, or energy claims.

## Production execution and completed evidence

The production command was run under `qtsd` on
`thuan-B860M-EAGLE-WIFI6`. It started at `2026-08-12T02:28:50.960631+00:00`
and the last validated repetition ended at
`2026-08-12T03:32:44.649323+00:00` (3,833.689 seconds elapsed across the
serial immutable schedule):

```bash
cd /home/thuan/topic_c_ivc
conda activate qtsd
PYTHONPATH=src python src/benchmark_trt_engines.py \
  --project-root /home/thuan/topic_c_ivc \
  --config configs/ivc_deployment_benchmark_v1.json
```

- Runtime identity shared by all 108 records: NVIDIA GeForce RTX 5090;
  driver `610.43.02`; CUDA UMD `13.3`; TensorRT Python `11.1.0.106`; Python
  `3.11.15`; `trtexec`
  `/home/thuan/traffic/third_party/TensorRT-11.1.0.106/bin/trtexec` with SHA-256
  `68b061d276601b7c6d8aafa4d8d75319f32382c85673360407a6d7ee6411aa4d`.
- Every invocation used the registry-bound engine and the fixed timing flags
  `--warmUp=5000 --duration=30`; the runner recorded the exact per-engine argv
  and used a shape flag only when the bound registry required one.
- Remote scheduler log:
  `outputs/logs/ivc_deployment_benchmark_v1_scheduler_20260812T022520Z.log`
  (SHA-256 `3644e3f053d7dd9371ae04de518a3bd038bfdd0ffd35430168375a89447a7cf2`).
- Completion report:
  `outputs/reports/ivc_deployment_benchmark_v1_complete.json`; canonical report
  SHA-256 `b09aac03e37bdf2dc3995e773c1130cb121a790d31b286bd73298e8248380846`.
  It binds 36 engine conditions, three repetitions per condition, 108 records,
  108 raw logs, and the engine-treatment table SHA-256
  `8403777e634c677d68aec7d9859c6eacfce876389c1c8df6842aeb2b288a0c56`.
- Local validation rechecked the canonical completion report, all 108 record
  SHA-256 values, all 108 log SHA-256 values, log envelopes, raw-output hashes,
  parsed throughput/latency values, exact grid, command bindings, one common
  runtime identity, and treatment bindings. It passed with the same canonical
  report SHA-256 above.

## Remote gate and commands

Do not open this gate while another GPU workload is active. The exact remote
`hostname` was recorded by an authorized read-only check; do not run the
benchmark with a guessed SSH address or local hostname. Copy only
`src/benchmark_trt_engines.py` and
`configs/ivc_deployment_benchmark_v1.json`, verify their SHA-256 values on the
remote host, then run the non-writing preflight first:

```bash
cd /home/thuan/topic_c_ivc
conda activate qtsd
PYTHONPATH=src python src/benchmark_trt_engines.py \
  --project-root /home/thuan/topic_c_ivc \
  --config configs/ivc_deployment_benchmark_v1.json \
  --preflight-only
```

The preflight validates 108 tasks and immutable inputs without creating output
parents or querying GPU/runtime state. Only after that static check passes, run:

```bash
cd /home/thuan/topic_c_ivc
conda activate qtsd
PYTHONPATH=src python src/benchmark_trt_engines.py \
  --project-root /home/thuan/topic_c_ivc \
  --config configs/ivc_deployment_benchmark_v1.json
```

The runner refuses an existing record, log, treatment artifact, or completion
report. Any interrupted/failed attempt therefore requires a new versioned
attempt rather than deletion or overwrite.

The production invocation validates the host/GPU/runtime identity, then runs
one fail-closed idle query before creating any output parent. It repeats the
same query before each of the 108 repetitions. A foreign PID at the initial
gate therefore leaves the immutable attempt namespace untouched and retryable.

Each immutable log begins with a runner-owned JSON envelope that records the
condition, repetition, run ID, timestamps, return code, engine digest, exact
argv, and a SHA-256/byte-count binding to the otherwise unchanged raw
`trtexec` output. Completion requires return code zero, one authentic `&&&&
RUNNING TensorRT.trtexec` banner whose parsed argv exactly matches the pinned
executable, engine, timing, and optional shape arguments, and one `&&&& PASSED
TensorRT.trtexec` footer whose repeated argv also matches exactly. Both lines
accept the standard one-or-more bracketed version/build groups before `#
<argv>`. Every `&&&& FAILED TensorRT.trtexec` terminal line is disqualifying,
including the standard multiple-group/argv form. Run IDs, raw-output digests,
and full log digests must be distinct across all 108 repetitions.

Completion parses exactly one bounded, complete performance summary from its
standard timestamp/level-prefixed start through the validated PASSED footer;
the footer, not invented explanation text, is the summary boundary. It selects
only line-start `Throughput:` and primary `Latency:` rows, so Enqueue,
End-to-End, H2D, GPU, and D2H latency rows cannot collide. It requires exact
agreement with the JSON throughput, mean, median, and p99 fields. All seven
primary latency values must be positive and finite, with
`min <= median <= p90 <= p95 <= p99 <= max` and `min <= mean <= max`.
Completion rejects multiple, spliced, incomplete, or internally impossible
summaries. It
also validates the exact 36-condition Cartesian grid, repetitions 1/2/3 per
condition, all registry and ONNX bindings, timestamps, idle-query evidence,
configured host/GPU/runtime identity, and 108 unique record/log paths. The full
runtime identity must be identical across all records; UUID and software
versions are format checked, and the recorded `trtexec --version`, executable,
and executable digest must agree with the pinned registry chain.

## Evidence synchronization boundary

The local attempt directory contains exactly 108 JSON records, 108 raw-log
envelopes, `engine_treatments.json`, and the completion report. The completion
report binds every copied record/log digest. Raw INT8/FP8 prediction payloads
remain remote and were not included in this deployment-evidence transfer.
