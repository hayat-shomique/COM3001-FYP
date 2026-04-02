# Component Note: Pipeline Overview — Data Flow and Row Tracking

**Date:** 2026-04-02

---

## Purpose

This note documents the complete data pipeline from raw CRSP ingestion through to model-ready train/test splits, with exact row counts at every stage. It serves as the single reference for the dissertation's data pipeline description and provides the chain of evidence that no rows were lost or gained without explicit justification.

---

## Pipeline stages

```
Module                     Input → Output                           Rows In → Rows Out   Delta   Reason
─────────────────────────  ──────────────────────────────────────   ─────────────────────  ──────  ──────────────────────────
1. fetch_wrds.py           WRDS CRSP → data/raw/spy_daily_raw.csv  —        → 3,774      —       SQL query, 2010-01-04 to 2024-12-31
2. preprocess_prices.py    raw → data/interim/spy_daily_clean.csv  3,774    → 3,774       0       CRSP convention fixes, no rows dropped
3. build_target.py         clean → data/processed/spy_targeted.csv 3,774    → 3,773      -1       Final row dropped (no next-day outcome)
4. build_features.py       targeted → data/processed/spy_featured  3,773    → 3,753      -20      Warm-up rows (20-day rolling windows)
5. temporal_split.py       featured → spy_train + spy_test          3,753    → 3,252+501    0       Chronological split at 2022-12-31
```

**Total rows lost:** 21 out of 3,774 (0.56%). Every loss is justified:
- 1 row: no observable next-day outcome (series end)
- 20 rows: rolling-window warm-up period (earliest observations, Feb 2010)

**No rows were imputed, forward-filled, or synthetically generated at any stage.**

---

## Column evolution

| Stage | Columns | New columns added |
|---|---|---|
| Raw | date, prc, openprc, askhi, bidlo, vol, ret | — |
| Clean | date, open, high, low, close, volume, return | Renamed only |
| Targeted | date, open, high, low, close, volume, return, **target** | target (binary direction) |
| Featured | date, open, high, low, close, volume, return, target, **14 features** | 14 engineered features |
| Train/Test | Same 22 columns | Split only, no column changes |

---

## Anti-leakage checkpoints

| Checkpoint | Module | What is verified |
|---|---|---|
| Target uses `shift(-1)` only | build_target.py | No same-day or future return in target |
| Features use `shift(1)` for return-derived signals | build_features.py | RSI, volatility exclude same-day return |
| No feature correlates >0.95 with target | build_features.py | Automated leakage scan |
| Train dates strictly before test dates | temporal_split.py | Zero-overlap verification |
| Feature ranges from training only | ga_strategy.py | GA threshold decoding uses train-only ranges |
| Model fitted on training only | all model modules | No test data in parameter estimation |

---

## Dissertation-ready pipeline diagram

> *Ready-to-lift wording:* "The data pipeline consists of five sequential stages, each implemented as a single-responsibility Python module with config-driven paths and logged diagnostics. Raw CRSP daily data (3,774 observations, January 2010 to December 2024) was standardised for CRSP conventions, labelled with a binary next-day direction target (losing 1 row at the series boundary), and augmented with 14 engineered features across 6 categories (losing 20 warm-up rows from the earliest observations). The resulting 3,753-row feature-bearing dataset was partitioned into a training set (3,252 rows, February 2010 to December 2022) and a held-out test set (501 rows, January 2023 to December 2024) using a strict chronological split with zero date overlap. No rows were imputed or synthetically generated. The total data loss across all pipeline stages was 21 rows (0.56%), each with explicit justification."
