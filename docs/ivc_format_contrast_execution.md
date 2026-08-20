# IVC format-contrast execution record

## Status

**Running: reviewed exact CPU-only analysis.**  The historical implementation was stopped for runtime diagnosis before producing evidence.  After the scheduler, provenance, and validation fixes passed independent review and a fresh source-hash preflight, a new immutable `ivc_format_contrast_v1` run was launched.  It is not a retraining, checkpoint rebuild, engine rebuild, or corruption-generation run.

## Remote identity and safety gate

- Host: `thuan-B860M-EAGLE-WIFI6`; launch timestamp: `2026-08-11T18:55:19Z` (UTC).
- Environment: `qtsd`, Python 3.11.15, NumPy 2.4.4, pycocotools 2.0.11, Ultralytics 8.4.95, TensorRT 11.1.0.106.
- GPU: NVIDIA GeForce RTX 5090, zero utilization at preflight; three pre-existing unrelated GPU processes were observed and left untouched.  The analysis reuses frozen JSON payloads and does not invoke CUDA/TensorRT.
- Disk before launch: 30 GB free.  No prior `ivc_format_contrast_v1` directory or completion report existed.

## Fresh reviewed-source preflight (2026-08-12)

After the final Task 2b source-package approval, the controller performed a new remote source synchronization and read-only preflight.  This is distinct from the historical probes and stopped original launch below; it did not run `--execute`, materialize an execution package or schedule, create an attempt artifact, or launch a scratch task.

The exact nine paths in `run_format_contrast_p0.EXECUTION_PACKAGE_FILES` were synchronized to the same relative paths below `/home/thuan/topic_c_ivc`.  A post-sync local/remote comparison proved SHA-256 equality for all nine files.  The source set was `configs/ivc_format_contrast_v1.json` (`38199fc...bcf6`), `src/run_format_contrast_p0.py` (`e9a4582...0316`), `src/format_contrast_scheduler.py` (`4632377...3d63`), `src/bootstrap_format_contrast.py` (`1bcfd36...2b36`), `src/bootstrap_format_contrast_macro.py` (`d20cb76...fccd7e`), `src/validate_format_contrast_evidence.py` (`d283b01...9a545`), `src/paired_bootstrap.py` (`07567eb...bffe`), `src/topic_c/manifest.py` (`2704f5a...758c7`), and `src/topic_c/tt100k_height.py` (`df58c5a...69a415`).  No raw prediction was synchronized.

Under `qtsd`, the exact command `PYTHONPATH=src python src/run_format_contrast_p0.py --project-root /home/thuan/topic_c_ivc --config /home/thuan/topic_c_ivc/configs/ivc_format_contrast_v1.json` exited 0 and reported exactly 12 clean and 144 corrupted tasks, with validated image universes of COCO 5,000, VOC 4,952, KITTI 1,496, and TT100K 3,067.  All 312 runner-derived prediction, input-record, and run-record paths existed.  All four annotations existed and matched their task-bound SHA-256 values, including the external COCO annotation at `/home/thuan/coco_journal/data/coco_src/annotations/instances_val2017.json`.

The fresh attempt work/bootstrap/task-log paths and all execution-package, scheduler-ledger, completion, and local-validation reports were absent before and after.  No production format-contrast process was present.  The historical stopped-launch log remained byte-identical.  At preflight the host had 61,355,118,592 bytes available RAM and 32,022,142,976 bytes free on the 97%-used project filesystem; swap had only 20,574,208 bytes free, an operational caution for a later persistent launch.  Full command, hashes, bindings, resource/process snapshot, and before/after evidence are recorded in `/data_nvme/paper/.superpowers/sdd/2026-08-12-ivc-evidence-first-strengthening/task-2-remote-preflight-rereview.md`.

## Current reviewed execution (2026-08-11T22:14:56Z)

Following the approved preflight, the exact nine-file reviewed package was launched under `qtsd` with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=src`:

```bash
/home/thuan/miniconda3/envs/qtsd/bin/python src/run_format_contrast_p0.py \
  --project-root /home/thuan/topic_c_ivc \
  --config /home/thuan/topic_c_ivc/configs/ivc_format_contrast_v1.json \
  --execute
```

- Parent PID: `2150886`.
- Immutable outer log: `/home/thuan/topic_c_ivc/outputs/logs/ivc_format_contrast_v1_scheduler_20260811T221456Z.log`.
- Immediately after launch, the parent was live and had exactly four clean-cell children: COCO `yolo11m`, COCO `yolo11n`, COCO `yolo11x`, and KITTI `yolo11m`, matching the configured cap of four.  The four schedules and the source/config execution-package manifest were the only new work/report files at that observation; no contrast component JSON had yet completed.
- This run is expected to take roughly 152.5 CPU-hours under the measured linear approximation.  Completion is not claimed until both scheduler ledgers, all 317 scientific files, the completion report, and a separate local validation record pass their hash-bound checks.

### Running-artifact checkpoint (2026-08-12)

A read-only monitor observed 32 immutable component JSON files: all 12 clean
components and 20 corrupted components.  The monitor revalidated every
available component's schema version, `n_boot == 2000`, canonical self-hash,
four input-hash bindings, schedule binding, clean-arm-cache binding, and finite
point estimates.  Every clean component correctly had no temporary draw cache;
every corrupted component had a hash-bound temporary draw cache.  This is an
early integrity checkpoint only, not a completion claim: the remaining 124
corrupted components, joint macro, schedules/caches, execution envelope,
completion report, and local fail-closed validation are still required.

## Historical original-launch package

Before the stopped original launch, the remotely synchronized sources were byte-identical to the then-reviewed local copies:

| File | SHA-256 |
|---|---|
| `src/bootstrap_format_contrast.py` | `38ec19f1ea52fb1f8d88fb69d3468c3e36a1f28089ae4613d8ace5f496186034` |
| `src/bootstrap_format_contrast_macro.py` | `4fc7c8fef282b462eb6d40a740447e38773fd67e880d1851b4b4d3c29ad9069d` |
| `src/run_format_contrast_p0.py` | `2b5589d1ff8cef1f3136173b63c4655e459da6ce63138f8fdb4ee752d5305415` |
| `configs/ivc_format_contrast_v1.json` | `4430997cae52cca4205d3d53b8cc2c23869f80896df38609faa304208c00662d` |

Those historical hashes are not the current corrective source set.  No new source/config package may be synchronized or launched until the cache-evidence and exact-accumulator reviews approve it; its hashes will be recorded before any fresh immutable attempt.

The non-mutating runner preflight reported 12 clean and 144 corrupted cells.  A separate binding pass verified all four annotations, 312 unique prediction files, 312 input records, and 312 run records against their manifest/prediction/image-ID hashes before launch.

## Persistent command and monitoring

```bash
PYTHONPATH=src /home/thuan/miniconda3/envs/qtsd/bin/python src/run_format_contrast_p0.py \
  --project-root /home/thuan/topic_c_ivc \
  --config /home/thuan/topic_c_ivc/configs/ivc_format_contrast_v1.json \
  --execute
```

- Parent PID: `2064928`; initial worker PID: `2064953`.
- Remote log: `outputs/logs/ivc_format_contrast_v1_20260811T185519Z.log`.
- The verified parent/child commands were `run_format_contrast_p0.py --execute` (PID 2064928) and its first-cell `bootstrap_format_contrast.py` worker (PID 2064953).  They were sent `TERM` only after command-line verification; both were subsequently confirmed stopped.  The target attempt contained zero contrast JSON artifacts and no completion report, so no evidence was removed or overwritten.
- Profiling one real COCO evaluator showed 18.182 s to build it and 0.917 s for one reference `accumulate_ap` call.  Since a cell requires four arms times 2,000 replicates, the reference serial estimator needs about 2.04 CPU-hours per COCO cell before macro recomputation.  The dominant work was the original per-replicate construction/concatenation and sorting of 320 category-area streams.

## Historical diagnosis and completion gate

The historical run was blocked on review of a corrective, test-driven execution path: deterministic shared per-dataset schedules; precommitted clean-arm and per-cell draw-cache evidence; a cache-fed joint macro that does not rebuild 144 evaluators; and bounded **cross-cell** subprocess parallelism.  That path has now passed independent review and is the running package above.  The static unique-score accelerator remains ineligible because the representative tie audit found ties in 1,272/1,280 streams; no static backend is used.  Final evidence remains gated on hash-bound completion and local validation.

### Bounded equivalence probe

The permitted one-replicate probe used the frozen four-arm `coco__yolo11n__clean-s0` cell, 5,000 images, and the derived COCO dataset seed `789174179`.  It wrote only `outputs/logs/ivc_format_contrast_probe_20260811T192000Z.log`, not an attempt artifact.  Serial (`workers=1`) and forked (`workers=8`) outputs were bitwise equal for every `Delta Q`, `Delta E`, and `Delta Psi` endpoint.  Evaluator construction took 71.661 s; the one-replicate serial and forked timings were 3.274 s and 4.075 s respectively (worker startup dominates at one replicate); maximum parent RSS was 6,602,216 KiB.  The probe parent and all worker PIDs were confirmed stopped.

The revised implementation additionally materializes and SHA-binds one legacy-compatible 2,000-row bootstrap schedule per dataset, validates that all cells of a dataset share one ordered image universe before evaluation, computes each dataset/model's two clean AP arms once, and reuses those arrays for its twelve corruption cells.  This reduces the planned arm-draw count from 1,248,000 to 624,000 without changing the paired estimator.  It is still awaiting independent code review and has not been run at 2,000 replicates.

### Bounded B=8 memory/timing probe

On the same frozen four-arm COCO `yolo11n` / `gaussian_noise-s1` cell, a fixed eight-row COCO schedule was evaluated serially and with fork workers.  The probe log is `outputs/logs/ivc_format_contrast_workers_b8_20260811T200539Z.log`; it used no attempt artifact and left no worker PIDs.  All three outputs had the identical draw digest `944624ba4b38f867b10ec204f30ef29d4b56e85a988dd9377de7cb02f2a6d90b`.

| Workers | Bootstrap time (B=8) | Peak parent+child RSS | Peak children |
|---:|---:|---:|---:|
| 1 | 27.538 s | 6.774 GB | 0 |
| 2 | 14.913 s | 20.240 GB | 2 |
| 4 | 8.783 s | 33.755 GB | 4 |

The 5090 host had approximately 62 GB RAM / 50 GB available at review.  This established four as the upper bound for the former within-cell fork probe.  The later cross-cell scheduler fixes one bootstrap worker and separately measures/enforces its own four-cell cap below.  Its B=8 four-cell probe processed 64 arm-draws in 56.315 s, so linear extrapolation to the 624,000 arm-draw plan is about 152.5 hours.  This is a transparent approximate limitation rather than a forecast (larger runs may differ in I/O and contention), but it confirms the full grid must not launch until the scheduler evidence contract receives independent review.

### Exact B=8 bounded cross-cell scheduler probe

The static backend remains out of scope.  Instead, the permitted read-only probe at `outputs/logs/ivc_format_contrast_multicell_b8_20260811T205255Z.log` (SHA-256 `d47023dd92140a16bc9d0a34a8f9a49044462d45b4c92b8771f0affae76bb280`) ran `source /home/thuan/miniconda3/bin/activate qtsd && PYTHONPATH=src python /tmp/ivc_multicell_b8_20260811T205255Z.py`.  It tested exact legacy `accumulate_ap` execution for the two corrupt arms of four frozen COCO cells: `yolo11n`, `yolo11m`, and `yolo11x` on `gaussian_noise-s1`, plus `yolo11n` on `fog-s1`.  All cells read the same temporary eight-row COCO schedule, generated with seed `789174179`, image-ID SHA-256 `da6fc732d63651c32997a20b8f1197813a923613f059986297c5ddd5c6325190`, and schedule SHA-256 `d8bbeee3b7b74731e2c79fe82b63a3f8a4bebe8e4d65e7769b6504370c4f3644`.  It created no attempt artifact.

| Cell cap | Wall time (four cells) | Peak aggregate parent+child RSS | Exact comparison | Residual child PIDs |
|---:|---:|---:|---|---|
| 1 | 176.022 s | 3.553 GB | reference | none |
| 2 | 105.490 s | 6.272 GB | bitwise equal per cell | none |
| 4 | 56.315 s | 11.673 GB | bitwise equal per cell | none |

The host reported 67.116 GB total and 61.234 GB available memory at start.  The reviewed local configuration consequently uses one bootstrap worker and a maximum of four cross-cell workers; the runner rejects nested parallelism and any cell cap above four.  `src/format_contrast_scheduler.py` logs each deterministic immutable cell command/PID/timestamps/exit status/output SHA-256 in a self-hashed ledger and stops new launches after a failure while preserving all existing evidence.  `--resume-verified` is intentionally refused until a fully testable provenance-complete resume implementation is approved.  This measured cap is not authorization for the full grid.

### Static-accumulator tie feasibility audit

The read-only audit log `outputs/logs/ivc_format_contrast_tie_audit_20260811T201140Z.log` inspected every category/area/arm score stream for frozen COCO `yolo11n` clean and `gaussian_noise-s1` four-arm cells.  Both cells had 1,272 tied streams and eight unique-score streams out of 1,280; all scores were finite and every descending/stable-mergesort check passed.  Clean had 7,792,120 tie-excess detections out of 9,666,392; corrupted had 7,809,356 out of 9,668,868.  Thus a unique-score-only fast path would cover only 0.625% of representative streams and is rejected.  The exact multiplicity/legacy-fallback design and its parity gates are in `docs/ivc_exact_static_accumulator_design.md`.  The audit produced no final contrast artifact and left no worker PID.

The follow-up stable-tie sensitivity audit log `outputs/logs/ivc_format_contrast_stable_tie_sensitivity_20260811T202554Z.log` classified tied score groups at every IoU/category/area stream.  Mixed TP+FP groups—the only groups whose internal ordering affects the TP/FP sequence—were 3.464% of clean groups (6.227% of tied detections) and 3.495% of corrupted groups (6.454% of tied detections).  A score-group replay backend must preserve the full stable sequence for every mixed group and fall back explicitly otherwise; it has not yet been implemented or used for an artifact.

### Required completion envelope after the fix round

The scientific evidence contract remains exactly **317 hash-bound compact files**: 157 JSON scientific artifacts (156 components plus the joint macro), four materialized schedules, 12 clean-arm caches, and 144 component draw caches.  The completion report is separate from that scientific count and binds the full 317-file map without any raw prediction payload.

The reviewed scheduler adds a separate immutable execution envelope, which must be synchronized and validated alongside the scientific chain: one precommitted execution-package manifest, two self-hashed scheduler ledgers, and 156 immutable task logs (12 clean plus 144 corrupted).  The manifest records SHA-256 values for the reviewed configuration and **eight** production source files, including the dynamically imported `src/topic_c/tt100k_height.py`; it also records the absolute remote `execution_root` used to construct child commands.  Each ledger binds that manifest and records the deterministic task command, PID, timestamps, exit status, every declared output's SHA-256, any declared missing output, and its task-log SHA-256.  The completion report binds the manifest and both ledger file/self-hashes in `execution_package` and `scheduler_ledgers`.

Before the first schedule/cache/artifact/log/ledger/report write, the runner requires the exact reviewed attempt name and resolves every planned path below the project root; traversal, absolute paths, and symlink escapes are refused.  A child failure stops new launches, drains already-started children, hashes every preserved declared byte even on nonzero exit, and writes a failure ledger.  The scheduler and child path additionally recheck the precommitted package so changed reviewed source bytes refuse execution.

After an approved remote job finishes, synchronize the 317 scientific files, the completion report, the 159-file execution envelope, the **nine** manifest-listed source/config files, and the relevant hash-bound run-manifest witnesses.  Never synchronize raw prediction JSON payloads.  Then run `src/validate_format_contrast_evidence.py` to create `outputs/reports/ivc_format_contrast_v1_local_validation.json`; it fails closed on any source, task-log, ledger, schedule, cache, artifact, endpoint-plan, macro-statistic, or binding mutation.  It also requires scheduler schema/name, `cell_workers == 4`, bounded `peak_active`, exact 12/144 phase membership, and command grammar.  Project-owned command paths remain compared to the manifest-recorded remote `execution_root`, never the local validation root; `--annotations` instead must be an exact, normalized absolute match to the component's hash-bound `annotation.path`, allowing the committed external COCO annotations without admitting a forged path.
