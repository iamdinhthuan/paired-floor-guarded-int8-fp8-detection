# Additional CVIU experiments — 2026-09-05

User approved additional experiments following the manuscript revision plan.
Historical result artifacts remain unchanged. The completed follow-ups were
integrated into the canonical CVIU main/S1 on 2026-09-07; see the current
`Thuan_paper_3_CVIU_revised/REVIEW_FINAL_CVIU.md` for publication verification.

## Completed: full TT100K fixed-universe sensitivity

- Host: thuan@100.111.139.103, /home/thuan/topic_c_ivc, conda qtsd.
- Existing tested runner: src/run_fixed_universe_sensitivity.py; no changes.
- Universe: n/m/x × fog/noise/JPEG/motion blur × severities 1/3/5 = 36 cells.
- 3,067 images; 10,000 positive-weight draws; 2,000-draw Monte Carlo checkpoint.
- Shared seed 20260905 and one shared weight matrix across all cells and arms.
- Four processes, one numerical-library thread each; no GPU inference.
- tmux session: cviu_tt100k_full_20260905.
- Outputs: outputs/bootstrap/tt100k_fixed_universe_full_v1_20260905/.
- Queue log: outputs/logs/tt100k_fixed_universe_full_v1_20260905/queue.log.
- Each completed cell is validated by validate_fixed_universe_artifact before
  DONE is recorded. A failure stops new dispatch; already-running cells finish.
- No automatic retry, deletion, overwrite or substitution of historical outputs.
- Queue completed 2026-09-06 at 16:16:11 +07:00: all 36 cells validated.
- Joint aggregation completed on 2026-09-06 with
  `src/aggregate_fixed_universe.py`. The aggregator revalidated every original
  prediction/run/input binding, report/draw/schedule hash, the exact 36-cell
  membership, common ordered image IDs and annotations. It averaged values
  within each shared draw, not interval endpoints. The ordinary reference was
  independently aggregated from its retained paired draw caches.

Runner SHA256: d2a60b6f145fb8f0342300feba377e405bc128b06a3e4ff6e61422affd0c4320

Accumulator SHA256: a81e4ebedaf733ee3e39e71918a8ac65e5f40bde947b5ab4b5d10bb9a91224cc

Weight schedule identity: 2b786e398bfdec5e1bcf91ccc46e94b56a6884d07963d5563341e1dbad614b9f

## Joint TT100K result (AP points, not native 0–1 AP)

| Endpoint | Uniform-weight point | Ordinary 2,000-draw interval | Fixed-universe 10,000-draw interval |
| --- | ---: | ---: | ---: |
| Overall Delta E | +0.606598 | [-0.149039, +1.099021] | [+0.200219, +1.104022] |
| Original-height Delta Psi | -1.377754 | [-2.508938, -0.140123] | [-2.319033, -0.605104] |

The height endpoint remains negative under both constructions. The overall
interval's zero-crossing status changes, so the result must be reported as
bootstrap-construction sensitivity, not universal class-composition invariance
or a corrected replacement of the ordinary interval. The full-sample point is
unchanged by construction. This does not recompute the four-dataset macro.

Largest shift across the three percentile summaries (2.5/50/97.5), comparing
the first 2,000 versus all 10,000 fixed-weight draws: 0.018907 AP for Delta E,
0.015439 AP for height Delta Psi. These are Monte Carlo checkpoint diagnostics,
not additional independent replications.

Remote/local report:
`outputs/bootstrap/tt100k_fixed_universe_full_v1_20260905/joint_macro.json`

SHA256: `ad54064979321d1e67600827aaacd37b2364f6f70bc86b94f0c8f8ee08515ff8`.

All 36 reports and compact draw caches were copied locally and their joint
percentiles independently recomputed with NumPy after file-hash checks.
The 112-MiB full remote folder retains the positive-weight schedule; local
compact evidence does not include that schedule or remote raw predictions.
Focused runner/macro/build/queue tests: 31 passed on 2026-09-06.

## Completed: controlled non-YOLO rebuild (launch history below)

The completed v3 attachment branch changed build/TF32 policy as well as
attachment. Rerunning that branch unchanged would not remove the confound.
A new comparison must rebuild default and aligned INT8 under the same current
build policy, freeze the FP8 reference and input/decoder/calibration identities,
verify source/reference parity, and report clean plus corrupted AP regardless
of whether alignment rescues accuracy. A successful clean-competitive non-YOLO
comparison must not be assumed or selected using corrupted outcomes.

At 17:18 +07:00 on 2026-09-06, the controlled-build queue was launched in tmux
session `cviu_controlled_retinanet_20260906`. Initial status is WAIT_GPU; no
engine build or inference had started. An unrelated compute process occupies
most VRAM. It is not terminated, paused or modified by this queue.

- Scope fixed before new metrics: KITTI / RetinaNet-R50-FPN-v2 only; this is a
  post-hoc mechanism check motivated by prior results, not a prospective study.
- Three serial engine builds: strict TF32-off FP32 diagnostic, default INT8,
  aligned INT8. Both INT8 graphs share source ONNX, calibration, decoder and
  geometry; both are rebuilt with the same existing builder/current TRT policy.
- 27 inference/evaluation bundles: FP32 matched-clean only; two INT8 treatments
  each on matched-clean plus all 12 corrupted conditions. Each bundle is checked
  against the corresponding frozen FP8 manifest and ordered image universe.
- 13 original FP8 prediction/run/input/metric bundles were validated before
  dispatch; 103 retained input/source files are hash-frozen in the execution
  config. No FP8 rebuilding, new quantization or training is requested.
- GPU admission before every GPU command: no unrelated compute context except
  the known Sunshine display process, at least 12,288 MiB free VRAM, and at
  least 20 GiB free disk. Checks repeat every 30 seconds in the background.
- All 84 build/infer/evaluate/validate commands execute serially and continuously
  once admitted. A command or provenance failure stops the queue, preserving
  partial evidence; no overwrite or automatic retry of failed scientific steps.
- New results are not selected by clean/corrupted AP. Timings are inadmissible
  for speed claims. Strict TRT FP32 clean AP alone does NOT establish full
  PyTorch/ONNX Runtime/TensorRT parity; that and inspector diagnostics remain
  follow-up checks before any mechanistic manuscript claim.
- Queue completion means three engines and 27 validated metric bundles, not a
  finished paired statistical analysis or an updated manuscript.

Frozen execution config, including all argv steps and the launch shell:
`configs/controlled_retinanet_tf32off_v1_20260906.json`.

Remote output directory:
`outputs/controlled_build/controlled_retinanet_tf32off_v1_20260906/`.

Monitor:

```bash
ssh thuan@100.111.139.103 'tail -n 40 -F /home/thuan/topic_c_ivc/outputs/logs/controlled_retinanet_tf32off_v1_20260906.queue.log'
```

The live queue log is mirrored from tmux; the first two WAIT_GPU lines remain
in tmux scrollback. All step-level logs are written directly to the output
directory. The existing manuscript and published archive remain unchanged.

## Automatic postprocessing (2026-09-06)

The controlled GPU queue completed at 23:07:22 +07:00: all three engines and
27 metric bundles were validated. No new training was performed.

`src/run_controlled_postprocess.py` now implements the continuous CPU tail:

1. Revalidate frozen source files, three engine payloads/build logs/registries,
   common TF32-off build policy, all new bundles and 13 frozen FP8 bundles.
2. Freeze an analysis manifest and one 2,000-draw ordinary image schedule
   (seed 20260906; sorted evaluator image IDs). This is a post-hoc sensitivity,
   not a prospectively registered analysis. No seed search is performed.
3. Compute 39 overall-AP draw caches (three clean + three arms for each of 12
   corruptions/severities), using at most four CPU processes. Every uniform
   sample must reconstruct its recorded COCO AP before bootstrapping. Repeated
   image positions remain repeated evaluator entries.
4. Reuse the exact FP8 vector across policy contrasts; check Omega cancellation
   algebraically. Aggregate the 12 aligned draws first, then take percentiles.
   Report clean AP, corrupted AP and signed losses for both policies and FP8.
5. Recheck all source hashes and automatically publish `summary.json`,
   `cells.csv`, `macro_table.tex`, `report.md`, and a hash-bound `complete.json`.

Completed AP caches can be resumed only after validating identities and hashes.
Partial/mismatched caches are preserved and rejected, never silently retried or
overwritten. Only one driver is admitted by a file lock. Scientific/provenance
failures stop pending dispatch; already-running CPU tasks may finish. A rerun
of the same entry point reuses completed caches and validates existing reports.

Remote tmux: `cviu_controlled_analysis_20260906`.

```bash
ssh thuan@100.111.139.103 'tail -n 50 -F /home/thuan/topic_c_ivc/outputs/logs/controlled_retinanet_tf32off_v1_20260906.analysis.log'
```

Outputs are kept together under:
`outputs/controlled_build/controlled_retinanet_tf32off_v1_20260906/analysis_v1/`.

The automation ends at verified analysis/report generation. It does not claim
source-PyTorch/ORT parity, publish data, edit the manuscript's interpretation,
or issue a submission-ready PDF. Those boundaries are also embedded in reports.

Verification: 22 local focused tests passed (one evaluator test skipped locally);
22 focused tests passed on the server, including the real COCO evaluator test
for repeated positions, reconstruction and reusable/corruption-rejecting caches.

Runner SHA256:
`19810947cc1802d7777a801523529a7c12bec5d374ff7018036dfed7869668f4`.
