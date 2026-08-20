# Absolute Accuracy Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hash-bound absolute corrupted AP and \(D\) guardrails to the direct IVC manuscript without rerunning inference.

**Architecture:** Extend the existing evidence builder with a fail-closed adapter from direct prediction arms to validated P0 metric records. Keep the direct contrast and absolute metric chains separate, then bind their outputs into one machine-readable direct audit consumed by the manuscript and Overleaf packager.

**Tech Stack:** Python 3, pandas, JSON/SHA-256 provenance ledgers, pytest, LaTeX/CAS class, Make.

## Global Constraints

- Use exactly 288 corrupted arms and exactly 312 distinct validated metric files.
- Require exact prediction, input-manifest, ordered-image, and run-record hash equality.
- Require \(\Delta E=D_{INT8}-D_{FP8}\) within absolute tolerance \(10^{-12}\).
- Store AP values as native fractions in CSV and render them as AP points in prose/tables.
- Do not rerun training, quantization, inference, metric evaluation, or bootstrap.
- Fail before output creation when any evidence or grid validation fails.

---

### Task 1: Direct-to-metric evidence adapter

**Files:**
- Modify: `analysis/build_paper_artifacts.py`
- Test: `tests/test_build_paper_artifacts.py`

**Interfaces:**
- Consumes: `load_direct_evidence(Path)` output and the validated P0 metric/completion directories.
- Produces: `load_direct_accuracy_guardrail(Path) -> tuple[pandas.DataFrame, dict[str, str], dict[str, str]]`.

- [ ] **Step 1: Write a failing integration test** that requires 288 unique arms, 312 metric hashes, both completion reports, the independently hand-checked median loss `0.055125`, and 35 arms below native AP `0.10`.
- [ ] **Step 2: Run the targeted test** with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q tests/test_build_paper_artifacts.py::test_direct_absolute_guardrail_binds_exact_metric_arms_and_reconstructs_delta_e` and confirm failure because the adapter/output is absent.
- [ ] **Step 3: Implement the minimal adapter** with exact provenance/semantic comparisons, exact grid checks, and the algebraic reconstruction gate.
- [ ] **Step 4: Add adversarial tests** that mutate one direct provenance hash and one metric AP value while preserving valid JSON, then require fail-closed errors.
- [ ] **Step 5: Run the targeted guardrail tests** and confirm all pass.

### Task 2: Generated artifacts and audit binding

**Files:**
- Modify: `analysis/build_paper_artifacts.py`
- Modify: `analysis/create_overleaf_upload.py`
- Test: `tests/test_build_paper_artifacts.py`

**Interfaces:**
- Consumes: the validated 288-row guardrail frame and 312 metric hashes from Task 1.
- Produces: `direct_absolute_guardrail.csv`, `direct_absolute_guardrail_summary.tex`, `direct_absolute_guardrail_narrative.tex`, and extended `direct_evidence_audit.json`.

- [ ] **Step 1: Write a failing builder test** requiring the three generated files, 16 total audited generated assets, and guardrail counts `288` and `312`.
- [ ] **Step 2: Run the builder test** and confirm failure because the artifacts/audit fields are absent.
- [ ] **Step 3: Implement the minimal writers and audit fields**, including AP unit metadata and completion-report hashes.
- [ ] **Step 4: Add a packager test** that rejects a missing or byte-mutated guardrail asset referenced by the audit.
- [ ] **Step 5: Run the builder and packager tests** and confirm all pass.

### Task 3: Manuscript integration

**Files:**
- Modify: `paper/main.tex`
- Modify: `paper/direct_methods_tail.tex`
- Modify: `paper/direct_results_template.tex`
- Modify: `paper/direct_discussion_template.tex`
- Modify: `paper/direct_availability_template.tex`
- Modify: `paper/supplement.tex`
- Modify: `paper/README.md`

**Interfaces:**
- Consumes: the generated guardrail table/narrative and audit from Task 2.
- Produces: a direct paper that distinguishes absolute degradation from the bootstrapped format contrast.

- [ ] **Step 1: Add manuscript contract checks** to the existing submission/build tests for the guardrail inputs and direct-only wording.
- [ ] **Step 2: Run the manuscript checks** and confirm failure because the new guardrail is not referenced.
- [ ] **Step 3: Integrate the guardrail** into RQ3, Methods, Results, Discussion, abstract, conclusion, supplement, and availability text without turning it into a superiority claim.
- [ ] **Step 4: Regenerate direct evidence** with `make -C paper direct-artifacts` and confirm the audit binds 288 arms, 312 metric records, and 16 generated assets.
- [ ] **Step 5: Rebuild both PDFs** with `make -C paper direct-final-pdf` and `make -C paper supplement.pdf`.

### Task 4: Final verification and Overleaf handoff

**Files:**
- Create: a uniquely named ZIP under `paper/`
- Verify: `paper/main.pdf`, `paper/supplement.pdf`, extracted ZIP contents

**Interfaces:**
- Consumes: all completed artifacts and manuscript sources.
- Produces: a self-contained Overleaf package that rebuilds the same main/supplement documents.

- [ ] **Step 1: Run the full test suite** with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q`.
- [ ] **Step 2: Run the submission audit** with `python analysis/submission_audit.py` and require success.
- [ ] **Step 3: Render page images** from both PDFs and inspect the new table plus adjacent pages for overflow, clipping, or duplicate floats.
- [ ] **Step 4: Build a unique Overleaf ZIP** with `python analysis/create_overleaf_upload.py --project-root . --output paper/overleaf_upload_ivc_direct_absolute_guardrail_20260813_v1.zip`.
- [ ] **Step 5: Extract the ZIP to a new temporary directory and compile main and supplement there**, requiring no undefined references/citations and matching audited generated-file hashes.

## Plan self-review

The plan covers every evidence-boundary, output, failure, manuscript, and packaging requirement in the design. It contains no unresolved placeholders. Function names, counts, units, and completion-chain interfaces are consistent across tasks.
