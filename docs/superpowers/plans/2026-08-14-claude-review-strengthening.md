# Claude-Review Manuscript Strengthening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the verified Reviewer-2 manuscript gaps using only the current validated direct-evidence chain.

**Architecture:** Extend the existing fail-closed paper-artifact generator so numerical prose, tables, and Figure 2 are regenerated from validated CSVs. Keep author metadata and the unfinished multi-seed experiment behind explicit gates. Regression tests inspect both source artifacts and rendered PDF ordering.

**Tech Stack:** Python 3.11, pandas, matplotlib/seaborn, LaTeX cas-dc, pytest, pdftotext, SHA-256 evidence ledgers.

## Global Constraints

- Do not add multi-seed results before its completion report validates.
- Do not invent Funding or DOI facts.
- All citations must remain 2023--2026 and exactly five must be IVC entries.
- All direct Results floats must render before Discussion.

---

### Task 1: Figure 2 AP-point unit integrity

**Files:**
- Modify: `analysis/build_paper_artifacts.py`
- Modify: `tests/test_build_paper_artifacts.py`

**Interfaces:**
- Consumes: native-fraction columns in `direct_format_contrast_cells.csv`.
- Produces: `paper/fig_paired_excess_gap.pdf` with both panels in AP points.

- [ ] Add a failing source-level test requiring `delta_e_all` conversion by 100 before the strip plot and an `AP points` axis label.
- [ ] Run the focused test and observe the unit failure.
- [ ] Implement the one-column conversion and precise label.
- [ ] Regenerate direct artifacts and verify the focused test passes.

### Task 2: RQ3 descriptive summaries and disclosures

**Files:**
- Modify: `analysis/build_paper_artifacts.py`
- Modify: `paper/direct_results_template.tex`
- Modify: `paper/direct_methods_tail.tex`
- Modify: `paper/direct_discussion_template.tex`
- Modify: `tests/test_build_paper_artifacts.py`
- Modify: `tests/test_submission_metadata.py`

**Interfaces:**
- Consumes: the 144-cell direct CSV, validated FP16 table, and bound completion-chain facts.
- Produces: compact capacity/corruption summaries and active-branch disclosure text.

- [ ] Add failing tests requiring capacity and corruption summaries in direct Results plus active PDF text for FP16 pass, TT100K class-composition, COCO digest exception, and P0 guardrail lineage.
- [ ] Run focused tests and observe the missing-output failures.
- [ ] Generate direct RQ3 summary tables from the 144 cells and include them before deployment Results.
- [ ] Add narrowly scoped verified disclosures to Methods/Discussion.
- [ ] Rebuild and verify focused tests and float-order regression.

### Task 3: Guardrail distribution and quantitative highlights

**Files:**
- Modify: `analysis/build_paper_artifacts.py`
- Modify: `paper/highlights.txt`
- Modify: `tests/test_build_paper_artifacts.py`
- Modify: `tests/test_submission_metadata.py`

**Interfaces:**
- Consumes: 288-row direct absolute guardrail CSV.
- Produces: mean-and-median loss prose and five quantitative highlights.

- [ ] Add failing tests for the recomputed mean 15.24 AP points, retained median 5.51, and quantitative highlights.
- [ ] Run focused tests and confirm the expected failures.
- [ ] Extend generated guardrail prose and revise highlights without exceeding the submission character limit.
- [ ] Regenerate artifacts and verify focused tests pass.

### Task 4: Full rebuild and completion audit

**Files:**
- Rebuild: `paper/main.pdf`, `paper/supplement.pdf`
- Regenerate later: final Overleaf folder/ZIP only after multi-seed integration.

**Interfaces:**
- Consumes: Tasks 1--3 outputs.
- Produces: audited current PDFs; does not claim final submission readiness.

- [ ] Run the full pytest suite, submission audit, `py_compile`, and `git diff --check`.
- [ ] Build main and supplement twice and require no undefined citations/references.
- [ ] Inspect every page and verify all Results floats precede Discussion.
- [ ] Record remaining author gates: Funding and public DOI/archive.

## Plan self-review

Every verified reviewer finding that can be fixed without author facts or new
experiments maps to a task. The plan contains no placeholder implementation
step and preserves the in-flight multi-seed evidence boundary.
