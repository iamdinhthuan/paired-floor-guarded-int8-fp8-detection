# Shared Q/DQ coverage-policy pilot V2

## Scientific question

`shared_mask_pilot_v2` tests whether the recorded INT8--FP8 corruption
interaction is sensitive to their unequal source-compute quantization coverage.
It is a three-block, post-hoc mechanistic sensitivity analysis; it neither
retrains checkpoints nor estimates a detector-family population effect.

For every block, V2 freezes the complete input-edge attachment contract of the
recorded baseline FP8 ONNX graph. It then:

1. copies that FP8 ONNX byte-for-byte into the V2 namespace, without running
   ModelOpt again; and
2. quantizes INT8 from the frozen FP32 source, then removes only surplus
   target-slot Q/DQ attachments until its source-compute input topology exactly
   matches the frozen FP8 contract.

The three diagnostic blocks frozen before V2 execution are VOC/YOLO11m,
VOC/RT-DETR-L, and
KITTI/RetinaNet-R50-FPN-v2. Whole-graph Q/DQ counts remain diagnostics. The
experiment compares complete recorded treatments and does not claim a pure
causal effect of numerical datatype or TensorRT kernel precision.

The primary policy-sensitivity quantity is

```text
Omega = DeltaE_default - DeltaE_aligned,
DeltaE = (FP8 - INT8)_corrupt - (FP8 - INT8)_JPEG95-clean.
```

Positive Omega means the default treatments have the larger format-gap change.
AP is COCO-style AP@[0.50:0.95] on the 0--100 scale.

## Why this is a new attempt

V1 was stopped before engine building because replaying both formats from
ModelOpt did not exactly reproduce the frozen FP8 topology. It produced three
surplus input attachments in YOLO11m, three in RT-DETR-L, and none in
RetinaNet. Those observations are a failed method-development attempt, not
scientific results.

V2 refuses to start unless it verifies both immutable V1 preservation objects:

| Object | Required SHA-256 |
|---|---|
| V1 failure-manifest canonical hash | `de9f8f60bd7ebbe2f3a4995127ee10fa0ec6e2492b42d6a2baf3bad4f891a2d2` |
| V1 failure-manifest file | `759380731f11f426e034f0987ef3571698b5c0fec1c50f1703635e86374b53c2` |
| V1 execution-source archive | `badf87386e66885b2059a2f85b0e5beebb565f60a2c0516419b98684c34dce6f` |

All V2 artifacts use fresh `shared_mask_pilot_v2` paths. V1 masks, ONNX files,
logs, and progress state are never resumed or admitted into V2.

## Fail-closed gates and phase order

The reviewed driver runs one GPU child at a time:

```text
bind V1 failure -> freeze 3 contracts
-> quantize/align 3 INT8 + byte-copy 3 frozen FP8
-> verify 3 exact topology reports
-> build 6 TensorRT engines (--noTF32)
-> infer/evaluate 6 JPEG-95 clean arms -> FP8 replay gate
-> infer/evaluate 72 corrupt arms -> 2,000-draw paired bootstrap
```

Before any engine build, every block must satisfy all of the following:

- V2 FP8 ONNX SHA-256 equals its frozen baseline FP8 ONNX SHA-256;
- the FP8 registry declares a byte-replay control and zero bypasses;
- INT8 and FP8 compute-input attachment contracts are identical;
- their common contract equals the frozen baseline FP8 contract; and
- the topology report is complete, self-hashed, and bound to both registries.

After clean inference, the FP8 replay must be within 0.10 AP point of the
historical matched JPEG-95 FP8 result for AP, AP50, and AP75, and within 0.1%
relative detection count. This is a replay guardrail, not a universal
practical-equivalence margin. Corrupted inference is blocked if it fails.

The same image-position bootstrap draw is reused across the eight
default/aligned x INT8/FP8 x clean/corrupt arms in a cell. Draw indices are also
aligned across conditions sharing a dataset image universe. The percentile
interval covers finite-image variation only.

## Background execution on the RTX 5090 host

After synchronizing the reviewed files to `/home/thuan/topic_c_ivc`, start the
single supervisor. It resumes only transient GPU-contention stops and exits on
scientific or provenance failures:

```bash
ssh thuan@100.111.139.103 '
  cd /home/thuan/topic_c_ivc &&
  mkdir -p outputs/logs/shared_mask_pilot_v2 &&
  nohup bash src/supervise_shared_mask_pilot.sh \
    /home/thuan/topic_c_ivc \
    shared_mask_pilot_v2 \
    configs/shared_mask_pilot_v2.json \
    > outputs/logs/shared_mask_pilot_v2/supervisor.launch.log 2>&1 < /dev/null &
'
```

The supervisor validates its PID identity, the requested attempt/config pair,
the completion marker and canonical report hash, activates `qtsd`, and ignores
only the persistent `sunshine` display context in the GPU-idle check. The
driver independently holds a non-blocking attempt lock.

## Monitoring

Stream the driver log:

```bash
ssh thuan@100.111.139.103 \
  'tail -n 100 -F /home/thuan/topic_c_ivc/outputs/logs/shared_mask_pilot_v2/driver.log'
```

Read compact progress:

```bash
ssh thuan@100.111.139.103 \
  'python -m json.tool /home/thuan/topic_c_ivc/outputs/reports/shared_mask_pilot_v2/progress.json'
```

Read supervisor decisions:

```bash
ssh thuan@100.111.139.103 \
  'tail -n 100 /home/thuan/topic_c_ivc/outputs/logs/shared_mask_pilot_v2/supervisor.log'
```

## Resume, immutability, and completion

Re-running V2 validates and reuses only complete artifacts in the V2 namespace.
An interrupted partial set is moved with hashes and reason under
`outputs/quarantine/shared_mask_pilot_v2/`. A complete artifact whose bytes or
scientific identity differ is never silently replaced; the driver stops.
Changing any reviewed execution-package byte requires another attempt
namespace.

Completion is admitted only when
`outputs/reports/shared_mask_pilot_v2/complete.json` binds the V1 failure
evidence, execution package, 3 masks, 6 ONNX controls/treatments, 3 topology
reports, 6 engines, 6 clean arms, 72 corrupted arms, 36 direct cells, all draw
caches, and both analysis summaries. Paper text must not use V2 numbers until
that report and its `.complete` marker validate.
