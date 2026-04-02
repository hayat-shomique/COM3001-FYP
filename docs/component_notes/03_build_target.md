# Component Note: build_target.py

**File:** `src/data/build_target.py`
**Date:** 2026-04-02

---

## Purpose

Reads the cleaned interim dataset, constructs a binary next-day direction label from CRSP's adjusted return field, drops the final row (no observable next-day outcome), and saves to `data/processed/spy_targeted.csv`.

This module crosses the boundary from data engineering into modelling assumptions. The target definition — binary next-day direction — is an explicit supervised learning design choice, not a data property. Making it a separate module ensures the assumption is visible, auditable, and independent of both the upstream data cleaning and the downstream feature engineering.

---

## Target definition

The target is constructed as:

    target_t = 1 if return_{t+1} > 0, else 0

where `return_{t+1}` is the CRSP-adjusted simple return on the next trading day. This uses `shift(-1)` on the return column: the target at row t is determined by the return at row t+1.

The final row in the dataset has no next-day return (the series ends), so its target is undefined and the row is dropped. This preserves the integrity of the label — no imputation or assumption is made about the missing outcome.

---

## Why binary direction

The project compares three paradigms on a common prediction task. Binary direction (up/down) is the simplest non-trivial prediction target that:

1. Is well-defined for all three paradigms (GeoBM can compute P(up), the GA can evolve rules for up/down, XGBoost can classify)
2. Has a clear naive baseline (majority-class frequency)
3. Avoids the additional complexity of regression targets (predicting return magnitude), multi-class targets, or event-labelling frameworks

The 0-threshold (return > 0) is the natural boundary — it separates positive from negative returns without imposing an arbitrary threshold. Days with exactly zero return are classified as "down" (not up), but this is rare in practice for a liquid ETF.

---

## What this module does NOT do

- **No feature engineering.** Features are computed from price/volume data downstream, not from the target.
- **No train/test splitting.** The temporal boundary is an evaluation design decision.
- **No multi-horizon targets.** The target is strictly 1-day-ahead. Multi-day labels would introduce overlapping outcome windows and require purging at the split boundary.

---

## Evidence produced

| Metric | Value |
|---|---|
| Input rows | 3,774 |
| Output rows | 3,773 (1 dropped — final row, no next-day outcome) |
| Target definition | 1 if next-day return > 0, else 0 |
| Class balance | ~55% up / ~45% down (full dataset) |
| Output | data/processed/spy_targeted.csv |

---

## Rubric mapping

| Criterion | How satisfied |
|---|---|
| **Design issues discussed** | Binary direction target justified. Boundary between data engineering and modelling marked explicitly. Final-row drop explained. |
| **Replication detail** | Target type, shift value, and paths all in config. Exact row counts logged. |
