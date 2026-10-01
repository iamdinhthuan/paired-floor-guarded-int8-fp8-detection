# HANDOFF — Neural Networks manuscript: quantization × corruption fragility

Read this file first. It is the single self-contained briefing for continuing
work on this repository. Target: *Neural Networks* (Elsevier), submission-ready
state as of release **v3.0.3**.

---

## 1. Current state (verified)

- Manuscript: `paper/main_nn.tex` (22 pp) + `paper/supplement.tex` (10 pp),
  Elsevier `cas-dc` class, STIX fonts, `\pdfgentounicode` text layer fixed.
- Published reproducibility release: **v3.0.3**, version DOI
  `10.5281/zenodo.23082850`, concept DOI `10.5281/zenodo.22031663`.
- Git: tag `v3.0.3` = commit `951b081` (the archived Zenodo snapshot).
  `main` = `91f0b03` — post-release cleanup that removed all CVIU/IVC-era
  files; Zenodo still archives the v3.0.3 tree.
- `submission_package/` = upload bundle (Elsevier Editorial Manager):
  01 main PDF, 02 supplement PDF, 03 highlights docx+txt, 04 graphical
  abstract tif, 05 cover letter pdf+txt, 06 LaTeX source zip,
  README_FIRST.txt, SHA256SUMS.txt. Items 01/02/03/06 **byte-match** the
  Zenodo v3.0.3 assets.
- Test baseline: `python3 -m pytest -q` → **110 passed, 4 skipped**
  (suite was trimmed to NN-only tests in the cleanup commit).
- Paper build: `cd paper && PATH=/data_nvme/texlive/2026/bin/x86_64-linux:$PATH
  ./verify.sh` → builds both PDFs + regenerates `SOURCE_MANIFEST.sha256`.

## 2. What the paper claims (and how far each claim reaches)

Title: *Quantization fragility under image corruption is recipe-dependent:
paired evidence from INT8 and FP8 object detectors.*

Estimand: `ΔE = (AP_FP8,corrupt − AP_INT8,corrupt) − (AP_FP8,clean −
AP_INT8,clean)` on paired, codec-controlled (JPEG-q95) inputs, common-image
bootstrap B=2000, shared draw schedule per dataset×model block. Interpretation
order is ΔE second, absolute AP and clean gap first.

| Claim | Status |
|---|---|
| FP8–INT8 corruption interaction flips sign across 9 YOLO11 blocks | Frozen paired evidence, bootstrap intervals, Holm-corrected |
| RetinaNet (R50-FPN-v2 recipe) has a severe INT8 deficit localized to the regression head | Frozen paired + decomposition, intervals |
| Deficit is ~all activation quantization; W-only ≈ full recovery, A-only ≈ full deficit | Post-hoc operand factorial, paired bootstrap but unregistered |
| Regression-head activations are heavy-tailed at fine FPN levels (P3/P4 excess kurtosis ≈ 427.5/208.1 vs cls-head ≈ 20/19) | Post-hoc activation captures; distribution-level, NOT causal proof |
| Q/DQ quantizers are per-level per-tensor (rejects shared head-wide amax) | ONNX structure report |
| Head-local max calibration does NOT repair the deficit (KITTI 27.2/19.6, VOC 52.3/37.9) → favors coarse-grid resolution over calibration-set clipping; rounding bias and eval-time clipping remain unresolved | Post-hoc counterfactual, point estimates |
| FCOS (anchor-free) shows no comparable head deficit and has much lighter P3/P4 tails (14.4/11.3 KITTI, 3.8/4.7 VOC) | Post-hoc, distribution-level consistency — not tail-free, not causal isolation |
| Pattern recurs on off-the-shelf COCO-pretrained checkpoints | Post-hoc replication arm |
| Corruption-aware calibration transfers inconsistently across folds — not a reliable repair | Intervention, complementary folds, intervals |
| KITTI final holdout (1,197 imgs): matched 34.70 / selective 65.25 / W-only 65.30 / A-only 34.64 clean AP; selective−matched = +30.54 clean, +21.18 corrupt mean | **Post-hoc point estimates only — no bootstrap intervals; frame-level split, NOT sequence-disjoint** |

## 3. Repository map (post-cleanup, NN-only)

```
paper/            main_nn.tex, supplement.tex, references.bib (63 entries,
                  strict cite closure), generated/nn_*.tex (GENERATED — never
                  hand-edit), figures/, verify.sh, build.sh,
                  scripts/make_manifest.py, .zenodo.json, CITATION.cff,
                  highlights.txt, cover_letter.txt, Highlights.docx,
                  Cover_Letter.docx, preview/ (built PDFs)
analysis/         nn_final_stats.py   (ledgers -> nn_final_stats.json)
                  build_nn_tables.py  (-> generated/nn_*.tex + nn figures)
                  collect_cell_points.py, compute_arm_points.py,
                  nn_contrasts.py, nn_actstats_report.py, nn_qdq_structure_report.py,
                  nn_corruptcalib_report.py, nn_rebuild_variance_report.py,
                  nn_shared_schedule_correlation.py, nn_latency_aggregate.py,
                  nn_paired_protocol_summary.py, build_nn_framework_fig.py,
                  regression_head_macs.py, kitti_calib_eval_phash.py,
                  validate_fcos_decoder.py, build_cviu_submission_docx.py
                  (name is legacy; it builds the NN Highlights/Cover_Letter
                  docx), submission_metadata.py, submission_package.py,
                  verify_local_archive.py, build_artifact_inventory.py,
                  run_fold2_merge.sh
src/              run_nn_*_queue.sh / *_bootstrap.sh (eval + bootstrap queues),
                  run_nn_paired_bootstrap.py, run_nn_preprocessing_pilot.py,
                  cross_family_infer_trt.py (TensorRT eval driver),
                  coco_eval.py, coco_infer_trt.py, evaluate_paired_subset.py,
                  materialize_codec_control.py (JPEG-q95 codec control),
                  generate_corruption.py, build_corrupt_calibration.py,
                  build_paired_subset.py, build_q95_calibration.py,
                  export_cross_family_onnx.py, export_yolo_onnx.py,
                  quantize_yolo_onnx.py, build_yolo_trt_engine.py,
                  build_trt_engine_registry.py, build_reghead_wa_arms.py
                  (W-only/A-only ONNX variants), acquire_ultralytics_dataset.py,
                  generate_coco_manifest.py, train_retinanet_dataset.py,
                  train_fcos_dataset.py, benchmark_trt_engines.py (latency),
                  paired_bootstrap.py, capture_provenance.py,
                  validate_manifest.py, validate_predictions.py,
                  validate_run_registry.py, topic_c/{manifest,cross_family,
                  coco_data,shared_quantization_mask}.py
configs/          nn_* protocol/intervention configs + corruptions.json +
                  ivc_deployment_benchmark_v1.json (latency; legacy name) +
                  training/cross_family_retinanet_r50_fpn_v2_v1.json
manifests/        hash-bound input/calibration/image/annotation manifests
outputs/          metrics/ + nn_* attempt ledgers (hash-bound evidence)
submission_support_20260911/
                  frozen evidence: phase_a_arms, phase_b_results,
                  phase_cd_results, phase_e_q95_results, phase_f_fold2_results,
                  phase_g_wa_results (operand factorial), phase_h_fcos_results,
                  phase_i_coco_pretrained_results, phase_j_maxcalib_results
                  (graph-wide max), phase_k_maxreg_results (head-local max),
                  phase_l_holdout_results (KITTI holdout, MANIFEST.sha256),
                  nn_actstats/ (incl. kitti_fcos.json, voc_fcos.json),
                  nn_rebuild_variance_v1_20260930/, nn_latency_v1_20260930/,
                  merged_cd_e/, shared_schedule/, nn_final_stats.json,
                  NN_EVIDENCE_README.md, review_notes/NN_*
submission_package/  journal upload bundle (see §1)
tests/            15 NN-guard test files (110 tests)
docs/             README, paired_excess_gap_method.md, coco_checkpoint_inventory.md
scripts/          publish_zenodo_archive.py
```

## 4. Regeneration commands

```bash
python3 analysis/nn_final_stats.py        # ledgers -> submission_support_20260911/nn_final_stats.json
python3 analysis/build_nn_tables.py       # -> paper/generated/nn_*.tex + figures
cd paper && PATH=/data_nvme/texlive/2026/bin/x86_64-linux:$PATH ./verify.sh
python3 -m pytest -q                      # expect: 110 passed, 4 skipped
```

- All in-text numbers are `\nn<Name>` macros from `generated/nn_numbers.tex`.
  **Never hard-write a computed number into prose** — add a macro in
  `build_nn_tables.py` and regenerate.
- Local conda TeX Live is broken; the repo's TeX Live 2026 at
  `/data_nvme/texlive/2026/bin/x86_64-linux` works. `verify.sh` prefers it and
  fails closed if `stix.sty` is missing.
- Remote GPU host: `ssh thuan@100.111.139.103`, mirror
  `/home/thuan/topic_c_ivc/`. Launch long queues with `setsid` (session
  children are killed on disconnect). `rsync -az --delete` `paper/`, `src/`,
  `analysis/`, `manifests/` before remote runs.

## 5. Review history — do not regress

Three adversarial review rounds already happened. Fixed items:

- **PDF text layer**: must keep `\input{glyphtounicode}\pdfgentounicode=1` and
  build with STIX Type1 (TeX Live 2026). `pdftotext` must yield clean
  "deficit/defined/off-the-shelf" and em dashes.
- **Max-calibration is head-local** (reg-head quantizers only), not graph-wide
  — graph-wide max destroyed VOC and is a separate negative control.
- **Shared-amax hypothesis is falsified** in the text: per-level per-tensor
  scales are reported; heavy tails concentrate at P3/P4.
- **KITTI split is frame-level** (seed 20260818), DontCare dropped, 8 classes;
  near-duplicate audit: ~3% calib images hash-collide with eval at aHash-16
  d=0 — disclosed in §3.4.
- **Rebuild audit** lists exactly 7 RetinaNet engines; YOLO11 engines were
  NOT rebuilt — text says so.
- **Calibration-draw variance is unmeasured** — stated as limitation, not hidden.
- **AI declaration** names Cognition Devin, OpenAI Codex/ChatGPT, Anthropic
  Claude Code with bounded roles; AI is never an author/creator anywhere.
- **Abstract/Conclusion** were tightened; conclusion is 5 short sentences.

## 6. Known limitations = levers for strengthening (ranked)

1. **Holdout has no uncertainty bars.** Phase L is point-estimate only.
   Cheapest credible upgrade: run the same paired-bootstrap machinery on the
   1,197 holdout images (records + cell_points exist; add bootstrap stage to
   `run_nn_holdout_queue.sh`/`run_nn_paired_bootstrap.py` path, then emit
   intervals via `nn_final_stats.py`).
2. **Split is frame-level, not sequence-disjoint.** KITTI adjacent frames can
   cross partitions. A sequence-level re-split + re-eval would answer the
   strongest reviewer objection; needs new manifests + full re-run (costly).
3. **Resolution vs rounding bias vs eval-time clipping not separated.** The
   head-local max counterfactual bounds the story but leaves the mechanism
   decomposition incomplete; text already says this — any further claim needs
   a new counterfactual (e.g., finer activation grid / piecewise-linear
   calibrator on the head only).
4. **Calibration-draw variance unmeasured.** Currently stated as a limitation;
   a small multi-draw sensitivity (rebuild engines under 2–3 calibration seeds)
   would quantify it.
5. **FCOS evidence is distribution-level.** Activation stats are consistent
   with the mechanism but not causal isolation; a FCOS head-local counterfactual
   would strengthen the contrast if attempted.
6. **COCO-pretrained replication covers 4 arms only** — bounded accordingly.
7. **Scale**: one RetinaNet recipe, one FCOS recipe, fixed calibration budget.
   Do not broaden claims without new evidence.

## 7. Hard rules

- Keep the five evidence layers distinct in prose: diagnostic, frozen paired,
  intervention, historical exploratory, post-hoc extensions (operand
  factorial, FCOS, COCO-pretrained, actstats, max-calib, holdout).
- Never turn post-hoc point estimates into interval claims; never call FCOS
  activations tail-free; never claim sequence-independent KITTI
  generalization; never claim universal INT8/FP8 ordering.
- `paper/generated/` is build output — regenerate, don't hand-edit.
- Secrets `git_token.txt` / `zenodo_key.txt` are gitignored — never commit,
  never print.
- Elsevier requires: highlights 3–5 bullets ≤85 chars each, graphical
  abstract TIFF, cover letter, editable LaTeX source archive — all live in
  `submission_package/` and are regenerated, not hand-fixed.

## 8. Verification gates before claiming done

```bash
python3 -m pytest -q                                  # 110 passed, 4 skipped
cd paper && PATH=/data_nvme/texlive/2026/bin/x86_64-linux:$PATH ./verify.sh
sha256sum -c SOURCE_MANIFEST.sha256                   # inside paper/
cd submission_package && sha256sum -c SHA256SUMS.txt  # all OK
pdfinfo paper/main_nn.pdf | grep Pages                # 22
pdfinfo paper/supplement.pdf | grep Pages             # 10
pdftotext paper/main_nn.pdf - | grep -c 'deficit'     # >0, no broken ligatures
```

Independent compile: unzip `submission_package/06_LaTeX_source_NN.zip` to a
clean dir and `pdflatex` both `main_nn.tex` and `supplement.tex`.

## 9. Next-version release procedure (v3.0.4 template)

1. Finish manuscript edits; regenerate stats/tables; `verify.sh`; full pytest.
2. `git commit` on main, `git tag v3.0.4`, push both.
3. Zenodo: create new-version draft under concept record `22031663`, reserve
   its DOI, then update `.zenodo.json`/`CITATION.cff`/README/cover-letter/
   tests to that DOI **before** publishing.
4. Build release assets to `/data_nvme/release_v3.0.4/`:
   `01_NN_main.pdf`, `02_NN_supplement.pdf`, `03_highlights_NN.txt`,
   `04_Overleaf_Source_NN.zip` (self-contained LaTeX: tex+bbl+bib+bst+cls/sty
   +figures+generated), `05_Source_Code_v3.0.4.zip` (`git archive` of the tag),
   `06_NN_Evidence_v3.0.4.zip` (git archive of `submission_support_20260911/`
   phase/ledger paths + `NN_EVIDENCE_README.md` + `paper/SOURCE_MANIFEST.sha256`),
   `README_v3.0.4.md`, `RELEASE_SHA256SUMS.txt`.
5. Upload to the Zenodo draft via bucket PUT (md5 is returned — verify against
   local), set metadata `version` + `related_identifiers` → `tree/v3.0.4`,
   publish, then verify the public record without a token.
6. Rebuild `submission_package/` from the same bytes; regen `SHA256SUMS.txt`.
7. Never announce push/publish without an authoritative check
   (`git ls-remote`, `curl https://zenodo.org/api/records/<id>`).

## 10. Environment notes

- Python: repo code is stdlib + NumPy/SciPy/matplotlib/yaml/Pillow;
  `requirements-remote.txt` is intentionally minimal (not a locked env).
- GPU work was done on remote RTX (TensorRT, ModelOpt). Engine rebuilds are
  not byte-guaranteed across stacks — provenance is recorded in manifests.
- The pytest suite runs offline in ~3 s; no GPU required.
