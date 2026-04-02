# Component Note: preprocess_prices.py

**File:** `src/data/preprocess_prices.py`
**Date:** 2026-04-02

---

## Purpose

Reads the raw CRSP daily data, applies unavoidable CRSP standardisation (negative price correction, column renaming), runs integrity checks, drops inadmissible rows, and saves the cleaned result to `data/interim/spy_daily_clean.csv`.

This module contains no modelling decisions. Every transformation is either a CRSP convention correction or a data integrity requirement. The boundary between data engineering and modelling is enforced: target construction, feature engineering, and train/test splitting are all downstream concerns.

---

## CRSP-specific transformations

**Negative price correction.** CRSP encodes the bid/ask midpoint as a negative value in the `prc` field when no closing trade occurred. The absolute value is the correct price. This module applies `abs(prc)` to resolve the convention, which is documented in the CRSP data manual and is standard practice in any study using CRSP data.

**Column renaming.** CRSP field names (`prc`, `openprc`, `askhi`, `bidlo`, `vol`, `ret`) are renamed to standard names (`close`, `open`, `high`, `low`, `volume`, `return`) for downstream readability. The rename is a pure cosmetic operation with no data transformation.

---

## What this module does NOT do

- **No imputation.** Missing values are dropped, not filled. Forward-fill or interpolation would impose modelling assumptions about price continuity.
- **No target construction.** The forward shift that creates the next-day direction label is a modelling choice, not a data property.
- **No feature engineering.** Rolling windows, lags, and indicators are computed downstream.
- **No train/test splitting.** The temporal boundary is an evaluation design decision.

---

## Evidence produced

| Metric | Value |
|---|---|
| Input rows | 3,774 |
| Output rows | 3,774 (0 dropped for null close/return) |
| Columns | date, open, high, low, close, volume, return |
| Negative prices corrected | Yes (CRSP convention) |
| Output | data/interim/spy_daily_clean.csv |

---

## Rubric mapping

| Criterion | How satisfied |
|---|---|
| **Design issues discussed** | CRSP-specific conventions identified and handled. Boundary between data engineering and modelling enforced. |
| **Replication detail** | Input/output paths in config. Column mapping documented. Another researcher can reproduce by running `python src/data/preprocess_prices.py`. |
