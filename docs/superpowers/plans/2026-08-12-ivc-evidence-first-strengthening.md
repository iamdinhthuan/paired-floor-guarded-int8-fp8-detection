# IVC Evidence-First Strengthening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reviewer-ready IVC manuscript with a direct paired INT8--FP8 contrast, RTX 5090 deployment evidence, recent claim-matched citations, readable presentation, and verified submission/reproducibility packages.

**Architecture:** New remote analysis artifacts form a separate `ivc_format_contrast_v1` completion chain and never overwrite P0. The paper builder validates that chain, derives all tables/figures/macros from it, and rejects inconsistent evidence. Manuscript and packaging consume only generated artifacts and versioned source records.

**Tech Stack:** Python 3.10, NumPy, pandas, pycocotools, TensorRT `trtexec`, NVIDIA ModelOpt, pytest, LaTeX/latexmk, rsync/ssh, SHA-256, BibTeX.

## Global Constraints

- Keep the frozen P0 primary analysis unchanged; create new attempts rather than overwrite files.
- No retraining, checkpoint rebuild, corruption-byte change, calibration change, split change, or evaluator change.
- All new remote runs use `thuan@100.111.139.103:/home/thuan/topic_c_ivc` and `conda activate qtsd`.
- New direct-format inference is exploratory; never state equivalence from a confidence interval containing zero.
- Use exactly 2,000 paired bootstrap replicates and the exact 12 clean / 144 corrupted-condition grids.
- Use only bibliography entries with verified publication/release years 2023--2026; cite exactly five claim-matched IVC articles.
- Do not falsify years for COCO, VOC, KITTI, TT100K, or foundational methods; describe frozen manifests/self-contained algorithms when the date rule precludes historical provenance citations.
- Preserve fail-closed SHA-256 validation, including the recorded narrow historical-COCO-config exception.
- Use final figure lettering of at least 7 pt and do not claim latency, memory, or energy unless directly measured.
- Do not publish a DOI, alter author order/roles, or invent funding/ORCID information.

---

### Task 1: Implement and test the fail-closed paired INT8--FP8 contrast pipeline

**Files:**
- Create: `src/bootstrap_format_contrast.py`
- Create: `src/run_format_contrast_p0.py`
- Create: `configs/ivc_format_contrast_v1.json`
- Create: `tests/test_format_contrast.py`
- Modify: `src/topic_c/tt100k_height.py` only if a reusable public helper is strictly required by both area and height runners.

**Interfaces:**
- Consumes four quantized prediction/input pairs per corrupted cell: `int8_clean`, `fp8_clean`, `int8_corrupt`, `fp8_corrupt`; the records must have one identical ordered unique image-ID list.
- Produces one JSON per `dataset__model__corruption-s{severity}` with `schema_version: 1`, `n_boot: 2000`, linked SHA-256 input bindings, `point.delta_q`, `point.delta_e`, `point.delta_psi`, percentile intervals, seed, endpoint type, and sign convention.
- Produces `outputs/reports/ivc_format_contrast_v1_complete.json` with exactly 12 clean contrasts, 144 corrupted contrasts, every artifact SHA-256, config SHA-256, and a canonical report hash.

- [ ] **Step 1: Write failing algebra and integrity tests**

Add tests that import `compute_contrast_from_aps`, `validate_linked_inputs`, and `canonical_hash` from `bootstrap_format_contrast` and assert:

```python
def test_delta_e_cancels_fp32_and_has_positive_int8_worsening_sign():
    values = compute_contrast_from_aps(
        int8_clean=[0.50, 0.20, 0.45], fp8_clean=[0.54, 0.22, 0.49],
        int8_corrupt=[0.30, 0.07, 0.31], fp8_corrupt=[0.37, 0.12, 0.38],
    )
    assert values["delta_q"]["all"] == pytest.approx(0.04)
    assert values["delta_e"]["all"] == pytest.approx(0.03)
    assert values["delta_psi"] == pytest.approx(
        values["delta_e"]["small"] - values["delta_e"]["large"]
    )

def test_linked_inputs_reject_different_image_order(tmp_path: Path):
    with pytest.raises(ValueError, match="identical ordered image IDs"):
        validate_linked_inputs([{"image_ids": [1, 2]}, {"image_ids": [2, 1]}])
```

Add tests that construct a temporary run-record tree with a missing FP8 corruption cell and assert `run_format_contrast_p0.build_tasks` refuses it; construct the full grid and assert 144 condition tasks plus 12 clean tasks.

- [ ] **Step 2: Run the focused tests and confirm red**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_format_contrast.py
```

Expected: collection fails because `bootstrap_format_contrast` and `run_format_contrast_p0` do not exist.

- [ ] **Step 3: Implement area and TT100K contrast calculation**

Implement one generic command-line runner with `--endpoint area|tt100k-height`. Build `COCOeval` or native TT100K height evaluators once per input arm, then apply one shared resample to all four arms per replicate. For area endpoints compute `delta_q`, all/small/medium/large `delta_e`, and `delta_psi = delta_e.small - delta_e.large`. For TT100K compute all AP for `delta_q`/`delta_e` and height small-like/large-like `delta_psi` using existing `HEIGHT_GROUPS`.

Reject any duplicate image IDs, input list mismatch, missing/empty prediction file, mismatched input-manifest SHA-256, an output path that already exists, non-positive `n_boot`, or non-finite point estimates. Store only interval summaries, not all replicates.

- [ ] **Step 4: Implement orchestration and immutable completion report**

Define the `ivc_format_contrast_v1` config with datasets, annotation paths, expected image counts, source attempts, models, quant precisions, 2,000 replicates, and one fixed seed namespace. Derive records from the existing P0 run manifests. Build and validate exactly 12 clean tasks and 144 condition tasks, run serially by default, and write only a new output attempt. Completion must be refused if task cardinality or any expected artifact hash is wrong.

- [ ] **Step 5: Run focused and full local tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_format_contrast.py
PYTHONPATH=src pytest -q
```

Expected: all contrast tests pass and no existing tests regress.

- [ ] **Step 6: Write task report**

Record files changed, commands, red/green evidence, interfaces, and any remote-only prerequisites in `.superpowers/sdd/2026-08-12-ivc-evidence-first-strengthening/task-1-report.md`.

### Task 2: Execute and validate the remote 2,000-replicate paired-format analysis

**Files:**
- Create remotely then sync locally: `outputs/bootstrap/ivc_format_contrast_v1/*.json`
- Create remotely then sync locally: `outputs/reports/ivc_format_contrast_v1_complete.json`
- Create: `outputs/reports/ivc_format_contrast_v1_local_validation.json`
- Create: `docs/ivc_format_contrast_execution.md`
- Modify: `configs/ivc_format_contrast_v1.json` only to record the immutable remote config hash if it was not already frozen in Task 1.

**Interfaces:**
- Consumes Task 1 scripts/config and the immutable remote prediction/input/run-manifest artifacts.
- Produces 156 contrast JSON files and one completion report; every local copy must SHA-256 match the remote source.

- [ ] **Step 1: Write the failing local validation test**

Add to `tests/test_format_contrast.py`:

```python
def test_complete_remote_contrast_report_has_exact_grid_and_hashes(project_root: Path):
    report = load_complete_contrast(project_root)
    assert report["clean_cells"] == 12
    assert report["condition_cells"] == 144
    assert report["n_boot"] == 2000
    assert len(report["artifacts_sha256"]) == 156
```

Expected before remote execution: fail because the local completion report is absent.

- [ ] **Step 2: Preflight the remote host without mutating evidence**

Copy only Task 1 code/config/tests needed for the run. Over SSH, activate `qtsd`, inspect GPU utilization, confirm all raw prediction/input paths and their linked manifest records exist, confirm no conflicting GPU process, and run one 10-replicate scratch task into a unique temporary directory. Verify the scratch output contains the four expected arm hashes and correct sign fields; delete only that unique scratch directory after inspecting it.

- [ ] **Step 3: Run the immutable remote analysis**

Run `src/run_format_contrast_p0.py` on the remote project with `--config configs/ivc_format_contrast_v1.json`, serial execution, and `--resume-verified`. Do not change existing P0 folders. Persist the command, package versions, host/GPU identity, timestamps, and report hash to the execution documentation.

- [ ] **Step 4: Synchronize compact evidence and validate hashes locally**

Sync only 156 contrast summaries, the completion report, config, relevant run manifests, and command log; do not copy raw prediction payloads. Validate each local artifact SHA-256 against the remote completion report and write `ivc_format_contrast_v1_local_validation.json` with `status: valid`.

- [ ] **Step 5: Run green tests and report**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_format_contrast.py
```

Write execution status, exact artifact counts, remote command, validation result, and elapsed time to `docs/ivc_format_contrast_execution.md` and the task report.

### Task 3: Add repeatable RTX 5090 deployment measurements and engine-treatment evidence

**Files:**
- Create: `src/benchmark_trt_engines.py`
- Create: `configs/ivc_deployment_benchmark_v1.json`
- Create: `tests/test_benchmark_trt_engines.py`
- Create remotely then sync locally: `outputs/benchmarks/ivc_deployment_benchmark_v1/*.json`
- Create remotely then sync locally: `outputs/benchmarks/ivc_deployment_benchmark_v1/logs/*.log`
- Create: `outputs/reports/ivc_deployment_benchmark_v1_complete.json`
- Create: `docs/ivc_deployment_benchmark_execution.md`

**Interfaces:**
- Consumes the 36 validated engine registries from COCO and transfer ladders.
- Produces 108 immutable repetition records and one completion report. Each record includes engine SHA-256/bytes, input name/shape, GPU/runtime identity, exact `trtexec` command, log SHA-256, median and percentile latency, and throughput.

- [ ] **Step 1: Write failing parser/grid tests**

Add tests for `parse_trtexec_latency`, `build_benchmark_tasks`, and `validate_idle_gpu`:

```python
def test_parse_trtexec_latency_reads_milliseconds_and_throughput():
    record = parse_trtexec_latency("Throughput: 412.50 qps\\nLatency: min = 1.20 ms, max = 2.50 ms, mean = 1.60 ms, median = 1.55 ms, percentile(99%) = 2.10 ms")
    assert record == {"throughput_qps": 412.5, "latency_mean_ms": 1.6, "latency_median_ms": 1.55, "latency_p99_ms": 2.1}

def test_build_benchmark_tasks_requires_36_unique_engines(tmp_path: Path):
    with pytest.raises(ValueError, match="36 unique engine conditions"):
        build_benchmark_tasks(tmp_path)
```

- [ ] **Step 2: Run the focused tests and confirm red**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_benchmark_trt_engines.py
```

Expected: import/collection failure before the implementation exists.

- [ ] **Step 3: Implement benchmark runner and completion validation**

Resolve one input tensor/shape from each engine registry. Use the registry's `trtexec` binding; execute exactly three runs per engine with `--loadEngine`, `--warmUp=5000`, `--duration=30`, and input shape flags only when required by the engine. Before each run, query `nvidia-smi --query-compute-apps=pid --format=csv,noheader`; refuse if a foreign compute PID is present. Parse raw output; reject omitted/zero/non-finite latency or throughput. Never overwrite a record/log. Build a report only if 36 engines × 3 repetitions validate.

- [ ] **Step 4: Implement engine-treatment table extraction**

Create a pure function that joins ONNX and engine registries with `quantize_yolo_onnx.py` versioned fields. Emit a machine-readable `engine_treatments.json` later consumed by the paper builder. Report unavailable fields as `unrecorded`; do not infer FP8 encoding/granularity beyond recorded data.

- [ ] **Step 5: Run remote benchmark and synchronize compact artifacts**

Copy the runner/config to the 5090 project and invoke it under `qtsd`. Sync JSON/log evidence and completion report only after remote hash verification. Write exact host, GPU, TensorRT/CUDA version, command, start/end, and the 108-record check to `docs/ivc_deployment_benchmark_execution.md`.

- [ ] **Step 6: Run green tests and report**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_benchmark_trt_engines.py
```

Write the task report with raw log count, engine repetition count, limitations (no memory/power claim), and test output.

### Task 4: Extend the fail-closed paper builder, generated evidence, and figures

**Files:**
- Modify: `analysis/build_paper_artifacts.py`
- Modify: `tests/test_build_paper_artifacts.py`
- Modify: `tests/test_format_contrast.py`
- Create: `paper/generated/format_contrast_summary.tex`
- Create: `paper/generated/deployment_summary.tex`
- Create: `paper/generated/engine_treatment.tex`
- Create: `paper/generated/format_contrast_cells.csv`
- Create: `paper/generated/deployment_benchmark.csv`
- Create: `paper/figures/fig_format_contrast.pdf`
- Modify: `paper/figures/fig_interaction_distributions.pdf`
- Modify: `paper/figures/fig_extreme_forest.pdf`
- Modify: `paper/figures/fig_interaction_heatmap.pdf`
- Modify: `paper/generated/artifact_audit.json`

**Interfaces:**
- Consumes Task 2's exact 156-item contrast chain and Task 3's 108-item benchmark chain.
- Produces validated DataFrames, macros/tables/CSV/vector figures, and an audit that records both new chains and their code/config/report hashes.

- [ ] **Step 1: Write failing builder tests**

Add tests that copy valid fixture reports then mutate one condition grid, one contrast SHA-256, and one latency record. Assert `validate_format_contrast`, `validate_deployment_benchmark`, and `validate_and_load` raise `RuntimeError`. Add a valid fixture assertion for 12 clean, 144 condition, 2,000 replicates, 36 engines, 108 repetitions, and no FP8/INT8 pooled `Psi` endpoint.

- [ ] **Step 2: Run the builder tests and confirm red**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_build_paper_artifacts.py tests/test_format_contrast.py
```

Expected: failures because the builder lacks both validators and generated outputs.

- [ ] **Step 3: Implement loading, validation, and derived summaries**

Validate canonical report hashes, artifact hashes, task grids, `n_boot=2000`, shared input list digests, engine identities, and three repetitions per engine. Generate format-contrast tables that distinguish `Delta Q`, all-dataset `Delta E`, area `Delta Psi`, and TT100K height `Delta Psi`; label all as targeted exploratory intervals. Generate deployment tables with medians across the three timing runs and explicit engine-size/latency/throughput units.

- [ ] **Step 4: Regenerate readable vector figures**

Set the actual final figure width/height and 8--9 pt base font. Replace dense or illegible panels with either a two-panel figure or table/CSV supplement. Remove raw pivot-index labels, use mathematical display labels, and retain the full machine-readable ledger. Do not alter scientific values.

- [ ] **Step 5: Run green builder checks**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_build_paper_artifacts.py tests/test_format_contrast.py tests/test_benchmark_trt_engines.py
python analysis/build_paper_artifacts.py
```

Assert the printed counts include the new contrast/benchmark chains and inspect `paper/generated/artifact_audit.json` for `status: valid`.

- [ ] **Step 6: Write task report**

Document validators, generated artifacts, figure physical sizes, exact new counts, and green command output.

### Task 5: Rewrite the manuscript, bibliography, supplement, and submission metadata

**Files:**
- Modify: `paper/main.tex`
- Modify: `paper/references.bib`
- Modify: `paper/highlights.txt`
- Create: `paper/supplement.tex`
- Create: `paper/appendix_tables.tex`
- Create: `paper/CITATION.cff`
- Create: `paper/zenodo.json`
- Modify: `paper/README.md`
- Modify: `paper/REVIEWER_NOTES.md`
- Modify: `analysis/submission_audit.py`
- Create: `tests/test_submission_metadata.py`

**Interfaces:**
- Consumes only generated Task 4 artifacts and verified DOI metadata.
- Produces a main paper with five IVC citations, all references in 2023--2026, an explicit direct-format result, engine-treatment table, supplement, and author-side archive metadata without a fabricated DOI.

- [ ] **Step 1: Write failing citation/submission tests**

Add a parser test that asserts every BibTeX `year` is in `[2023, 2026]`, every cited key has an entry, no bibliography key is orphaned, exactly five entries have journal `Image and Vision Computing`, and their DOI set is exactly:

```python
{
 "10.1016/j.imavis.2023.104692",
 "10.1016/j.imavis.2024.105035",
 "10.1016/j.imavis.2024.105054",
 "10.1016/j.imavis.2024.105095",
 "10.1016/j.imavis.2025.105740",
}
```

Add tests that require a Funding declaration heading, a Data availability statement that says a DOI is pending rather than inventing one, valid `CITATION.cff` fields, and `zenodo.json` with non-empty title/description/creators placeholders marked for author confirmation.

- [ ] **Step 2: Run tests and confirm red**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_submission_metadata.py
```

Expected: fail because current bibliography contains pre-2023 entries and no IVC entries or metadata files.

- [ ] **Step 3: Rewrite literature and novelty positioning**

Remove every pre-2023 citation and rewrite associated sentences so no recent source is credited for an older origin. Retain or add only verified 2023--2026 sources. Cite exactly the five required IVC articles at their claim-specific locations. State that Karimov et al. already found condition/model dependence; distinguish this paper by codec-matched `E`, executable FP8, direct paired contrasts, size endpoints, and hash-bound artifact chain.

- [ ] **Step 4: Integrate new evidence and repair claims**

Replace “similar” INT8/FP8 language with the direct contrast/interval result. Add a compact treatment table, deployment table, and direct-contrast figure/table. Confine the quantified AP-floor finding to `E`. Replace deployment preference language with measured secondary latency/throughput/engine-size facts. Move audit inventories and full generated tables to `supplement.tex`; keep only concise validation prose in the main Results.

- [ ] **Step 5: Reflow and copy edit**

Use the shorter title `Paired Excess-Gap Evaluation of INT8 and FP8 YOLO11 Detectors under Image Corruption` unless the direct contrast results require a more precise wording. Put all appendix heading/table explanations adjacent in the supplement. Remove float-only pages where possible, use no figure text below 7 pt, standardize `E`, `Delta E`, `Psi`, `Delta Psi`, and use direct, non-causal language.

- [ ] **Step 6: Add archive-ready metadata without fabricating author facts**

Write `CITATION.cff` with existing title/authors/affiliations and a `message` requiring final author verification; write `zenodo.json` with the same title, description, keywords, and creators marked for verification. Include a `Funding` section with the journal's no-specific-funding sentence only if the project notes explicitly support that fact; otherwise write a clearly marked author-action statement to `REVIEWER_NOTES.md` and leave no factual funding assertion in the manuscript.

- [ ] **Step 7: Run green metadata/audit tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_submission_metadata.py
python analysis/submission_audit.py
```

Expected: audit passes with zero old references, exactly five IVC citations, abstract/highlight limits, and no orphaned bibliography keys.

- [ ] **Step 8: Write task report**

Record bibliography count/year distribution, exact five IVC DOI values, manuscript claim changes, unresolved author-controlled gates, and tests.

### Task 6: Build the supplement and flat Editorial Manager/reproducibility packages, then verify end-to-end

**Files:**
- Create: `paper/submission_package_ivc_v2/`
- Create: `paper/submission_package_ivc_v2/Manuscript.pdf`
- Create: `paper/submission_package_ivc_v2/Supplement.pdf`
- Create: `paper/submission_package_ivc_v2/Editorial_Manager_Flat_Source.zip`
- Create: `paper/submission_package_ivc_v2/Reproducibility_Package.tar.gz`
- Create: `paper/submission_package_ivc_v2/SHA256SUMS`
- Create: `paper/submission_package_ivc_v2/README.md`
- Modify: `paper/main.pdf`
- Create: `paper/supplement.pdf`

**Interfaces:**
- Consumes the complete validated evidence, manuscript, supplement, flat sources, archive metadata, and all generated figures/tables.
- Produces independently compilable main/supplement PDFs and two clean archives whose checksums validate.

- [ ] **Step 1: Write a failing flat-package verification test**

Add a test to `tests/test_submission_metadata.py` that unzips a fixture flat source archive, asserts no archive member contains `/`, checks `main.tex`, `references.bib`, all `\\input` files, and all `\\includegraphics` targets live in that one directory, and runs `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` when TeX is available.

- [ ] **Step 2: Run test and confirm red**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_submission_metadata.py
```

Expected: fail because no flat archive exists.

- [ ] **Step 3: Build PDFs and flat source archive**

Compile `paper/main.tex` and `paper/supplement.tex`. Assemble a separate flat directory containing renamed or flattened generated TeX and figure PDF files; adjust copied `main.tex` paths only inside that directory. Include `main.bbl` if BibTeX is needed for an Editorial Manager compile. Validate compilation from a new `mktemp -d` extraction directory.

- [ ] **Step 4: Assemble reproducibility archive**

Include source code, tests, frozen configs, manifests, reports, compact metric/bootstrap/contrast/benchmark evidence, builder/submission audit, generated artifacts, citation metadata, and instructions. Exclude raw datasets, engine binaries, checkpoints, and raw predictions, and state their access/regeneration boundary accurately.

- [ ] **Step 5: Run all verification gates**

Run:

```bash
PYTHONPATH=src pytest -q
python analysis/build_paper_artifacts.py
python analysis/submission_audit.py
PATH=/data_nvme/texlive/2026/bin/x86_64-linux:$PATH latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
PATH=/data_nvme/texlive/2026/bin/x86_64-linux:$PATH latexmk -g -pdf -interaction=nonstopmode -halt-on-error supplement.tex
sha256sum -c paper/submission_package_ivc_v2/SHA256SUMS
```

Render every page of both PDFs and inspect final text/figure legibility. Reject unresolved citations/references, fatal LaTeX errors, overfull boxes, stale artifact hashes, missing package members, or a PDF/source mismatch.

- [ ] **Step 6: Write task report and completion ledger entry**

Record page counts, package byte sizes/checksums, independent extracted-build evidence, visual-inspection coverage, and remaining external submission gates.

## Plan self-review

- Spec coverage: Tasks 1--2 implement and execute direct paired contrasts; Task 3 implements deployment/treatment evidence; Task 4 integrates fail-closed generated outputs and figures; Task 5 rewrites scientific narrative/references; Task 6 compiles and packages every deliverable.
- Placeholder scan: no TBD/TODO/“implement later” instructions remain; external author facts are explicitly bounded rather than guessed.
- Type consistency: Task 1 outputs are named `ivc_format_contrast_v1`; Task 2 synchronizes that exact name; Task 4 consumes the same name; Task 3 uses `ivc_deployment_benchmark_v1`; Tasks 4--6 consume that exact name.
