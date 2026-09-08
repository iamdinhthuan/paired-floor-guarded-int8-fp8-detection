# CVIU V4 — execution and handoff

Approved 2026-09-07. Canonical manuscript: `Thuan_paper_3_CVIU_revised/`.
Scientific plan: `CVIU_V4_Ke_hoach_nang_cap.md` in that directory. The remote
`configs/cviu_v4_analysis_plan.md` is its frozen execution copy, not a second
editable paper. No GitHub/Zenodo publication or DOI changes are authorized here.

## Completed clean-control queue

Host: `thuan@100.111.139.103`, project `/home/thuan/topic_c_ivc`, conda `qtsd`.
Attempt: `outputs/cviu_v4_clean_control_v1/`.
Registry frozen at **2026-09-07 03:58:01 UTC / 10:58:01 +07** before V4 AP evaluation.
Terminal completion and recovery markers verified on 7 September; GPU usage
was 0% at the 11:49 +07 status check. Inference, metrics and bootstrap are each
32/32 complete. The six-primary-block shift is +0.0573566 AP with paired
percentiles [−0.1839256, +0.2517661]. All eight individual intervals cross zero;
TT100K n/x diagnostics (+0.8995/+1.1878 AP) remain outside the primary macro.

- Six primary VOC/KITTI final-holdout blocks; two separate TT100K n/x diagnostics.
- 32 newly executed original-source/JPEG-95 × INT8/FP8 clean arms.
- One GPU process, then up to eight CPU bootstrap workers, BLAS/OpenMP threads=1.
- A resident `/usr/bin/sunshine` desktop service below 1 GiB is allowed; other
  compute processes cause waiting. No unrelated process is terminated.
- 2,000 paired draws per dataset. Seeds VOC202609071/KITTI202609072/TT100K202609073.
- No scientific results are overwritten or silently retried with changed settings.
- CPU AP and bootstrap start automatically after inference. A `complete.json`
  marker binds terminal outputs; `registry.complete.json` means only the inputs
  were frozen, not that the experiment finished.

Live log:

```bash
ssh thuan@100.111.139.103 'tail -f /home/thuan/topic_c_ivc/outputs/logs/cviu_v4/clean_control_v1.log'
```

Read-only progress (number of committed inference arms):

```bash
ssh thuan@100.111.139.103 'find /home/thuan/topic_c_ivc/outputs/cviu_v4_clean_control_v1/inference -name complete.json -type f | wc -l'
```

After an interruption, review the log and preserved partial files first. Only
fully hash-validated completed arms/caches may be reused. The runner's `--run`
mode resumes a frozen registry; **do not rerun `--prepare` after freezing**.

## Completed audits and implementation evidence

1. `outputs/analysis/cviu_v4/holdout_provenance/holdout_provenance.json`:
   12 retained treatment chains and 156 clean/corrupted run chains verified.
   This checks checkpoint/calibration/graph/build/engine/prediction/metric
   consistency, not independent execution attestation. Historical evaluator
   hashes remain unrecorded. All historical final clean inputs are **original
   source**, not JPEG-95; main/S1 and the new tables use that definition.
2. `outputs/analysis/cviu_v4/tt100k_annotations/complete_audit/`:
   no retained positive-label conversion defects. Of 7,641 validation records,
   6,544 have explicit empty original object lists. Four test-list and two
   train-list identifiers have source JPEGs but no original JSON records;
   neither negative labels nor missing annotated objects are inferred. Sixteen
   exactly-96px boxes occupy both inclusive L/XL component ranges. No dataset,
   selection or checkpoint changes were made.
3. `outputs/analysis/cviu_v4/holdout_example/`:
   fresh extraction of 34 payload files reproduced KITTI final1197/YOLO11m/fog1
   from predictions and annotations without AP caches. Original-clean APs
   67.328098/70.528749 and corrupted APs63.589215/66.509381 yield
   DeltaE−0.280484AP, paired interval[−1.269237,+0.784413]. Point error0;
   interval error8.22e−15AP. This is a specific tested example, not a claim that
   every historical inference can be regenerated. Dataset-derived material is
   not covered by the code's MIT license; review rights before public deposit.
4. `outputs/analysis/cviu_v4/holdout_synthesis/`:
   six-block point/AP and paired DeltaE ledger reconstructs macro−0.5498074216AP.
   The initial missing-CI summary is preserved. The completed separate recovery
   regenerates all 12 clean AP vectors on the exact historical schedules before
   combining paired vectors. Corrupted gap is +1.050221 AP with interval
   [+0.910530, +1.155391]. Adding interval endpoints is prohibited.

The historical clean-only recovery tail completed after the hash-valid
clean-control run. Its `holdout_synthesis/recovered/recovery.complete.json`
confirms 12 clean vectors and all six block intervals. The original missing-CI
summary and matching source snapshot remain preserved. Historical NumPy and
evaluator versions were not recorded; the current recovery environment is
identified and is not retroactively assigned to historical runs.

```bash
ssh thuan@100.111.139.103 'tail -f /home/thuan/topic_c_ivc/outputs/analysis/cviu_v4/holdout_synthesis/recovery_queue.log'
```

## Final integration verification

- All 32 clean arms and 12 recovery arms are complete; both V4 publication
  tables are generated from verified source records and draw vectors.
- Main/S1 integrated with the three-contribution narrative and all completed
  V4 results, including the small primary S and uncertain selected diagnostics.
- Final main21/S1 19 pages compile and have been visually inspected; both source
  ZIPs independently reproduce all 40 pages pixel-for-pixel. The evidence ZIP
  verifies 1,848 bound files and its actual final1197 example runs without caches.
- See canonical `REVIEW_FINAL_CVIU.md` for the five-task acceptance audit,
  final hashes, test results and deliberately unclaimed external-replication limits.
- No new public release, archive or DOI is created automatically.

The explicit evidence build command, **after** terminal remote results have
been downloaded into the corresponding local V4 namespaces, is:

```bash
python analysis/build_reviewer_evidence.py --root . --include-v4 --output /tmp/CVIU_V4_Reviewer_Evidence.zip
```

The output path must not already exist. This mode checks all32 clean AP vectors
and their common schedules, recomputes block and six-primary-block S intervals,
requires all recovered holdout gap intervals, validates audit/example hashes,
and excludes duplicate old/full example payloads. It currently refuses an
incomplete V4 build without creating an archive. Download only clean-control
registry/seal, points, bootstrap jobs, summary, completion and `bootstrap/`,
plus the synthesis `recovered/` directory. The other32 raw prediction sets stay
on the research host; only the tested holdout example is in the compact package.

Temporary compilation/backup files are outside the project at
`/tmp/cviu-v4-integration.aWHI4u/`. New work stays in the one canonical paper
directory, dedicated code/tests and the V4 evidence namespace.
