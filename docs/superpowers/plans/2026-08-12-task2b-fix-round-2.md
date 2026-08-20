# Task2b Fix Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the TT100K evaluator source and make local completion validation prove the recorded cap-four scheduler command contract.

**Architecture:** The execution package gains the dynamically imported TT100K evaluator and a remote execution-root string. The validator reads the strict production configuration, verifies scheduler metadata, then reconstructs each phase command from fixed task/component bindings and compares it to the ledger without substituting local paths.

**Tech Stack:** Python 3.11, pytest, JSON/SHA-256 evidence chain.

## Global Constraints

- No remote action, grid execution, GPU work, training, rebuild, raw-prediction synchronization, or destructive deletion.
- Preserve stock evaluator/statistics, immutable 2,000-row schedules, and the 317 scientific-evidence count.
- Source edits use `apply_patch`; tests precede each production behavior change.
- The validated production scheduler remains `cell_workers == 4`, `bootstrap_workers == 1`.

---

### Task 1: Complete and bind the execution package

**Files:**
- Modify: `src/run_format_contrast_p0.py: EXECUTION_PACKAGE_FILES, validate_execution_package, materialize_execution_package`
- Test: `tests/test_format_contrast.py`
- Modify: `docs/ivc_format_contrast_execution.md`, `.superpowers/sdd/2026-08-12-ivc-evidence-first-strengthening/task-2-report.md`

**Interfaces:**
- Produces: manifest `execution_root` string plus exactly nine `files_sha256` entries (configuration plus eight source files).
- Consumes: remote project root passed to `materialize_execution_package`.

- [ ] **Step 1: Write failing source-set/source-mutation tests**

```python
assert set(EXECUTION_PACKAGE_FILES) == {
    "configs/ivc_format_contrast_v1.json", ..., "src/topic_c/tt100k_height.py"
}
# Mutate tmp_path / "src/topic_c/tt100k_height.py" after manifest materialization.
with pytest.raises(ValueError, match="source SHA-256 mismatch"):
    validate_complete_report(tmp_path, config)
```

- [ ] **Step 2: Run tests and observe package omission failure**

Run: `PYTHONPATH=src pytest -q tests/test_format_contrast.py -k 'execution_package_source_set or tt100k_source_mutation'`

Expected: FAIL because the literal required set contains the absent TT100K source.

- [ ] **Step 3: Implement minimal manifest completion/root binding**

```python
EXECUTION_PACKAGE_FILES = (..., "src/topic_c/tt100k_height.py")
document["execution_root"] = str(root)
```

Require a non-empty absolute `execution_root` in both runner and local manifest validation; include it in canonical manifest hashing.

- [ ] **Step 4: Run the same tests green**

Run: `PYTHONPATH=src pytest -q tests/test_format_contrast.py -k 'execution_package_source_set or tt100k_source_mutation'`

Expected: PASS.

### Task 2: Validate exact cap and command semantics

**Files:**
- Modify: `src/validate_format_contrast_evidence.py: _validate_execution_package, _validate_scheduler_ledger, validate_complete_report`
- Test: `tests/test_format_contrast.py`

**Interfaces:**
- Consumes: validated config, package `execution_root`, component plan/documents, and a ledger phase.
- Produces: fail-closed validation of schema/name, cap/peak, 12/144 phase count, and exact phase-specific commands.

- [ ] **Step 1: Write rebound-ledger/report mutation tests**

```python
for field, value, message in [
    ("cell_workers", 99, "scheduler worker cap"),
    ("peak_active", 99, "scheduler peak"),
    ("clean_command", ["python", "wrong.py"], "scheduler clean command"),
    ("corrupt_command", ["python", "wrong.py"], "scheduler corrupt command"),
]:
    # Rehash affected ledger and completion report.
    with pytest.raises(ValueError, match=message):
        validate_complete_report(tmp_path, config)
```

- [ ] **Step 2: Run tests and observe validator acceptance failure**

Run: `PYTHONPATH=src pytest -q tests/test_format_contrast.py -k 'scheduler_ledger_rebound_contract'`

Expected: FAIL because the prior validator accepts synthetic commands and altered cap/peak values.

- [ ] **Step 3: Implement deterministic contract checks**

```python
validate_runtime_config(config)
assert ledger["schema_version"] == 1
assert ledger["scheduler"] == "bounded independent immutable format-contrast cells"
assert ledger["cell_workers"] == config["cell_workers"] == 4
assert 1 <= ledger["peak_active"] <= ledger["cell_workers"]
```

Generate expected clean/corrupt command lists from `execution_root`, config, task/component cache/schedule bindings, and record fields. Compare them as exact lists; compare only against the recorded remote root, never the local validator root.

- [ ] **Step 4: Run the same tests green**

Run: `PYTHONPATH=src pytest -q tests/test_format_contrast.py -k 'scheduler_ledger_rebound_contract'`

Expected: PASS.

### Task 3: Document and verify

**Files:**
- Modify: `docs/ivc_format_contrast_execution.md`, `.superpowers/sdd/2026-08-12-ivc-evidence-first-strengthening/task-2-report.md`
- Create: `.superpowers/sdd/2026-08-12-ivc-evidence-first-strengthening/task-2b-fix-round-2.md`

- [ ] **Step 1: Correct package counts and document command portability**

State eight production source files plus config (nine package files), that command paths are compared with manifest-recorded remote `execution_root`, and exact cap-four/one-bootstrap-worker validation.

- [ ] **Step 2: Run final verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -k 'not complete_remote_contrast_report'
PYTHONDONTWRITEBYTECODE=1 python -m py_compile src/run_format_contrast_p0.py src/validate_format_contrast_evidence.py
git diff --check
```

Expected: full local noncompletion suite passes; only remote-completion gate is deselected.
