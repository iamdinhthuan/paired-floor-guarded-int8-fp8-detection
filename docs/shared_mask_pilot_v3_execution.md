# Shared Q/DQ coverage-policy pilot V3

## Scientific contrast

`shared_mask_pilot_v3` is a three-block, post-hoc mechanistic sensitivity
analysis for VOC/YOLO11m, VOC/RT-DETR-L, and
KITTI/RetinaNet-R50-FPN-v2. It tests whether the recorded INT8--FP8 corruption
interaction changes after aligning INT8 source-compute Q/DQ attachments to the
frozen FP8 attachment contract. It does not retrain checkpoints or estimate a
detector-family population effect.

```text
Omega = DeltaE_default - DeltaE_aligned,
DeltaE = (FP8 - INT8)_corrupt - (FP8 - INT8)_JPEG95-clean.
```

V3 creates only aligned INT8. Its FP8 term reuses the exact existing default
FP8 engine/prediction/metric evidence for all 39 block-condition arms. Each
historical run is bound to one immutable per-block engine registry, the exact
engine bytes, and the configured baseline-FP8 ONNX registry/bytes. No FP8
ONNX, engine, prediction, or metric is rebuilt, copied, hard-linked, or rerun.
The identical FP8 arm therefore cancels algebraically. Each aligned INT8 ONNX
must exactly match the frozen baseline-FP8 source-compute attachment contract
before any engine is built.

## Attempt transition

V1 is an immutable failed method-development attempt. V2 corrected that design
but completed only a source-bound preflight whose package precommitted one
worker and an exclusive-GPU gate. The later request for bounded shared-GPU
parallelism requires a fresh V3 namespace.

Before V3 source synchronization, all 20 files in V2's execution package were
archived and member-hash verified. V3 validates the V1 failure evidence, V2
package/config, V2 source archive and preservation report, absence of V2
progress/scientific artifacts, and a deterministic V2 abandonment manifest.
Nothing from V1 or V2 is admitted as a V3 scientific result.

## Bounded parallel policy

- Two CPU-EP INT8 quantizers, each capped to four OMP/MKL/OpenBLAS/NumExpr/ORT
  threads.
- One TensorRT build at a time to avoid concurrent tactic-selection confounds.
- At most three independent accuracy inference/evaluation workers.
- 4,096 MiB GPU safety margin plus 4,096 MiB reserved per inference: three
  workers require at least 16,384 MiB free at dispatch.
- Serial builds reserve 8,192 MiB and require at least 12,288 MiB free.
- Named GPU-capacity failures remain wait-and-retry conditions. An unexpected
  driver death is retried at most eight times with bounded backoff, but only
  after no attempt-scoped child remains. Explicit scientific/provenance
  failures stop the supervisor.

Unrelated resident GPU services are permitted. Every V3 `runtime_seconds`
value is therefore explicitly **inadmissible** for scientific speed comparison.
Latency, if needed, must be remeasured in a separate isolated-GPU attempt.

```text
bind V1 -> preserve/abandon V2 -> freeze 3 contracts
-> align 3 INT8 ONNX (2 CPU workers) -> verify 3 topology reports
-> build 3 INT8 engines serially -> bind 39 exact default-FP8 bundles
-> infer/evaluate 3 clean + 36 corrupt INT8 arms (<=3 workers)
-> mark timings inadmissible -> paired 2,000-draw analysis (3 workers)
```

## Launch and monitoring

```bash
ssh thuan@100.111.139.103 '
  cd /home/thuan/topic_c_ivc &&
  mkdir -p outputs/logs/shared_mask_pilot_v3 &&
  nohup bash src/supervise_shared_mask_pilot.sh \
    /home/thuan/topic_c_ivc shared_mask_pilot_v3 \
    configs/shared_mask_pilot_v3.json \
    > outputs/logs/shared_mask_pilot_v3/supervisor.launch.log 2>&1 < /dev/null &
'
```

```bash
ssh thuan@100.111.139.103 \
  'tail -n 100 -F /home/thuan/topic_c_ivc/outputs/logs/shared_mask_pilot_v3/{supervisor,driver}.log'
```

```bash
ssh thuan@100.111.139.103 \
  'python3 -m json.tool /home/thuan/topic_c_ivc/outputs/reports/shared_mask_pilot_v3/progress.json'
```

Completion and the supervisor re-hash the V1/V2 transition evidence, V3
execution package, three
masks, three aligned-INT8 ONNX files, three topology reports, three INT8
engines, 39 new INT8 bundles, all source files behind the 39 exact default-FP8
bindings (including the three engine and ONNX registries/bytes), 36 paired draw
caches, both summaries, and the timing-inadmissibility report. Paper text must
not use V3 results until `complete.json` and its marker validate.
