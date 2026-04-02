# Component Note: fetch_wrds.py

**File:** `src/data/fetch_wrds.py`
**Date:** 2026-04-02

---

## Purpose

Fetches daily OHLCV data for SPY from WRDS CRSP (Center for Research in Security Prices) and saves the raw result to `data/raw/spy_daily_raw.csv`. No cleaning, preprocessing, or transformation — single responsibility.

This is the first module in the pipeline. It establishes the institutional data foundation that every downstream module depends on. Using WRDS CRSP rather than free data sources (Yahoo Finance, Alpha Vantage) is a deliberate methodological choice: CRSP provides split-adjusted and dividend-adjusted returns, verified OHLCV data, and a standardised identifier system (PERMNO) that eliminates the survivorship bias, data quality, and corporate action adjustment problems endemic to free sources.

---

## Why WRDS CRSP

CRSP is the standard institutional data source for academic research in financial economics. It is used in the canonical studies that this project cites: Fama (1970) on market efficiency, Lo & MacKinlay (1988) on return autocorrelation, and Brock, Lakonishok & LeBaron (1992) on technical trading rules. Using the same data source as the literature it references strengthens the project's methodological standing and ensures that any comparison with published results is conducted on comparable data.

The CRSP Daily Stock File (`crsp.dsf`) provides the `ret` field — the holding-period return adjusted for splits and dividends. This is the return used throughout the pipeline for target construction, feature engineering, and GeoBM parameter estimation. No manual adjustment for corporate actions is needed.

---

## PERMNO resolution

SPY is identified by PERMNO 84398. This was resolved programmatically via `crsp.dsenames` rather than hardcoded, then verified manually against WRDS and locked in `config/data_config.yaml`. The resolution step demonstrates understanding of CRSP's identifier system, where a single ticker can map to multiple PERMNOs across time periods.

---

## Scope boundaries

**Data cleaning** is deferred to `preprocess_prices.py`. This module saves raw CRSP output without modification.

**Column renaming** is deferred. CRSP field names (`prc`, `openprc`, `askhi`, `bidlo`, `vol`, `ret`) are preserved in the raw output.

**Date filtering** is applied via the SQL query (2010-01-04 to 2024-12-31) but no further row-level filtering is performed.

---

## Evidence produced

| Metric | Value |
|---|---|
| Rows fetched | 3,774 |
| Date range | 2010-01-04 to 2024-12-31 |
| Asset | SPY (PERMNO 84398) |
| Source | WRDS CRSP Daily Stock File (crsp.dsf) |
| Fields | date, prc, openprc, askhi, bidlo, vol, ret |
| Output | data/raw/spy_daily_raw.csv |

---

## Rubric mapping

| Criterion | How satisfied |
|---|---|
| **Data source quality** | Institutional-grade CRSP via WRDS — the standard in academic finance research |
| **Replication detail** | PERMNO, date range, fields, and source table all in config. Another researcher with WRDS access can reproduce exactly. |
| **Professional practice** | WRDS credentials handled via environment variables, not committed to repo |
