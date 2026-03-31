"""
COM3001 — CRSP Price Standardisation and Cleaning
==================================================
Reads the raw CRSP daily data, applies unavoidable CRSP standardisation
(negative price correction, column renaming), runs integrity checks,
drops inadmissible rows, and saves the cleaned result to data/interim/.

This module contains NO modelling decisions:
  - No target variable construction (forward shift is a modelling choice)
  - No feature engineering (indicators, rolling windows)
  - No train/test splitting (temporal boundary is a modelling choice)
  - No imputation or forward-fill (imputation is a modelling assumption)

Every transformation here is either a CRSP convention correction or a
data integrity requirement. Nothing downstream-specific.

Usage:
    python src/data/preprocess_prices.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CRSP Standardisation
# ---------------------------------------------------------------------------

def resolve_negative_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct CRSP negative price convention.

    CRSP encodes the bid/ask midpoint as a negative value in the PRC
    field when no closing trade occurred on that day. The absolute value
    is the appropriate price to use. For SPY this should affect zero rows
    (confirmed by ingestion diagnostics), but the correction is applied
    unconditionally for methodological completeness.

    See CRSP documentation: 'A negative price indicates that the price
    is the average of the bid and ask, not an actual closing price.'
    """
    n_negative = (df["prc"] < 0).sum()
    if n_negative > 0:
        logging.info(
            f"Correcting {n_negative} negative PRC values "
            f"(CRSP bid/ask midpoint convention)."
        )
    else:
        logging.info("No negative PRC values to correct.")
    df["prc"] = df["prc"].abs()
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CRSP field names to standard OHLCV convention.

    This makes downstream code readable without CRSP domain knowledge.
    The mapping is:
        prc     -> close    (closing price, after abs correction)
        openprc -> open     (opening price)
        askhi   -> high     (daily high)
        bidlo   -> low      (daily low)
        vol     -> volume   (trading volume in shares)
        ret     -> return   (holding period return, split/dividend adjusted)
    """
    column_map = {
        "prc": "close",
        "openprc": "open",
        "askhi": "high",
        "bidlo": "low",
        "vol": "volume",
        "ret": "return",
    }
    df = df.rename(columns=column_map)
    logging.info(f"Renamed columns: {list(column_map.keys())} -> {list(column_map.values())}")
    return df


# ---------------------------------------------------------------------------
# Integrity Checks
# ---------------------------------------------------------------------------

def parse_and_verify_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date column to datetime and verify temporal ordering.

    Asserts that dates are monotonically increasing after sorting.
    A non-monotonic date column would indicate a data integrity failure
    in the raw CRSP extract.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_monotonic_increasing, (
        "FATAL: dates are not monotonically increasing after sorting. "
        "This indicates a data integrity failure in the raw CRSP extract."
    )
    logging.info(
        f"Date range verified: {df['date'].min().date()} to "
        f"{df['date'].max().date()} (monotonically increasing)."
    )
    return df


def check_duplicate_dates(df: pd.DataFrame) -> None:
    """
    Assert no duplicate dates exist in the cleaned dataset.

    This is checked in the raw data by fetch_wrds.py, but re-checked
    here as a post-cleaning integrity gate.
    """
    n_dupes = df["date"].duplicated().sum()
    if n_dupes > 0:
        dupes = df[df["date"].duplicated(keep=False)]["date"].unique()
        logging.error(f"Duplicate dates found after cleaning: {dupes}")
        raise ValueError(
            f"Data integrity failure: {n_dupes} duplicate dates in cleaned data."
        )
    logging.info("No duplicate dates in cleaned data.")


# ---------------------------------------------------------------------------
# Admissibility Filtering
# ---------------------------------------------------------------------------

def drop_inadmissible_rows(
    df: pd.DataFrame, required_columns: list[str]
) -> pd.DataFrame:
    """
    Drop rows where required columns contain null values.

    Close and return are required by every downstream model. A row
    without a closing price or return is inadmissible — no model in
    the framework can use it. Open/high/low nulls are preserved
    because imputation is a modelling choice, not a cleaning step.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe after CRSP standardisation.
    required_columns : list[str]
        Column names that must be non-null for a row to be admissible.

    Returns
    -------
    pd.DataFrame
        Dataframe with inadmissible rows removed.
    """
    n_before = len(df)

    # Log null counts per required column before dropping
    for col in required_columns:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            logging.warning(f"  Nulls in '{col}': {n_null} rows will be dropped.")

    df = df.dropna(subset=required_columns).reset_index(drop=True)
    n_dropped = n_before - len(df)

    if n_dropped > 0:
        logging.warning(
            f"Dropped {n_dropped} inadmissible rows "
            f"(null in {required_columns})."
        )
    else:
        logging.info(
            f"No inadmissible rows — all {len(df)} rows have "
            f"non-null {required_columns}."
        )
    return df


# ---------------------------------------------------------------------------
# Output Diagnostics
# ---------------------------------------------------------------------------

def log_clean_diagnostics(df: pd.DataFrame) -> None:
    """
    Log summary diagnostics for the cleaned dataset.

    Reports column inventory, remaining nulls, value ranges, and
    basic statistics. This provides an auditable record of the
    cleaned data state before any modelling decisions are applied.
    """
    logging.info("-" * 40)
    logging.info("CLEANED DATA DIAGNOSTICS")
    logging.info(f"  Columns: {list(df.columns)}")
    logging.info(f"  Rows: {len(df)}")

    # Remaining nulls per column
    nulls = df.isnull().sum()
    for col, count in nulls.items():
        if count > 0:
            logging.info(f"  Remaining nulls in {col}: {count} ({count/len(df)*100:.1f}%)")
    if nulls.sum() == 0:
        logging.info("  No remaining nulls in any column.")

    # Key statistics
    logging.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logging.info(f"  Close range: ${df['close'].min():.2f} — ${df['close'].max():.2f}")
    logging.info(f"  Mean daily return: {df['return'].mean():.6f}")
    logging.info(f"  Std daily return: {df['return'].std():.6f}")
    logging.info(f"  Trading days: {len(df)}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_clean(df: pd.DataFrame, output_path: str) -> Path:
    """Save cleaned data to CSV at the configured output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved cleaned data: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Full preprocessing pipeline: load raw, standardise, check, filter, save."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 50)
    logging.info("COM3001 — CRSP Price Standardisation and Cleaning")
    logging.info("=" * 50)

    # Load config
    config = load_config()
    input_path = config["preprocessing"]["input_path"]
    output_path = config["preprocessing"]["output_path"]
    required_cols = config["preprocessing"]["drop_if_null"]

    # Load raw data
    logging.info(f"Loading raw data: {input_path}")
    df = pd.read_csv(input_path)
    raw_row_count = len(df)
    logging.info(f"Loaded {raw_row_count} rows.")

    # Step 1: Parse and verify dates
    df = parse_and_verify_dates(df)

    # Step 2: Resolve CRSP negative price convention
    df = resolve_negative_prices(df)

    # Step 3: Rename CRSP fields to standard OHLCV
    df = rename_columns(df)

    # Step 4: Drop inadmissible rows (null close or return)
    df = drop_inadmissible_rows(df, required_cols)

    # Step 5: Post-cleaning integrity gate
    check_duplicate_dates(df)

    # Final column order
    df = df[["date", "open", "high", "low", "close", "volume", "return"]]

    # Diagnostics
    log_clean_diagnostics(df)

    # Save
    save_clean(df, output_path)

    # Summary
    n_dropped = raw_row_count - len(df)
    logging.info("=" * 50)
    logging.info("PREPROCESSING COMPLETE")
    logging.info(f"  Input:   {input_path} ({raw_row_count} rows)")
    logging.info(f"  Output:  {output_path} ({len(df)} rows)")
    logging.info(f"  Dropped: {n_dropped} rows")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
