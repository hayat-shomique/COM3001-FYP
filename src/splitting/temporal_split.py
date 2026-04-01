"""
COM3001 — Temporal Train/Test Split for SPY Direction Prediction
================================================================
Reads the feature-bearing dataset (data/processed/spy_featured.csv),
splits it into train and test sets at a strict chronological boundary
(train ≤ 2022-12-31, test ≥ first trading day 2023), and saves the
two partitions to separate CSV files.

ANTI-LEAKAGE DISCIPLINE:
  The split is purely chronological — no randomisation, no shuffling.
  No row from the test period can appear in the training set.
  No row from the training period can appear in the test set.
  No scaling, normalisation, or feature engineering is performed here.
  The split boundary is read from config/data_config.yaml.

This module does NOT:
  - Shuffle or randomise rows (would destroy temporal ordering)
  - Scale or normalise features (model-dependent, done downstream)
  - Engineer features or modify existing columns
  - Create or modify the target column
  - Perform stratified sampling (violates chronological discipline)
  - Perform cross-validation folding (separate evaluation concern)

Usage:
    python src/splitting/temporal_split.py

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
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Split Logic
# ---------------------------------------------------------------------------

def parse_and_verify_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the date column is datetime, sorted chronologically,
    and contains no duplicates.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if not df["date"].is_monotonic_increasing:
        logging.warning("Date column is not sorted — sorting chronologically.")
        df = df.sort_values("date").reset_index(drop=True)

    n_dupes = df["date"].duplicated().sum()
    if n_dupes > 0:
        raise ValueError(
            f"FATAL: {n_dupes} duplicate date(s) found. "
            "Cannot perform unambiguous temporal split."
        )

    logging.info("Date verification passed: sorted, no duplicates.")
    return df


def temporal_split(
    df: pd.DataFrame, train_end_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset at a strict chronological boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a datetime 'date' column, already sorted.
    train_end_date : str
        Last date (inclusive) for the training set, e.g. '2022-12-31'.

    Returns
    -------
    (train_df, test_df) : tuple of pd.DataFrame
    """
    boundary = pd.Timestamp(train_end_date)

    train_df = df[df["date"] <= boundary].reset_index(drop=True)
    test_df = df[df["date"] > boundary].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("FATAL: Training set is empty after split.")
    if len(test_df) == 0:
        raise ValueError("FATAL: Test set is empty after split.")

    return train_df, test_df


def verify_no_overlap(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Confirm zero date overlap between train and test sets."""
    train_dates = set(train_df["date"])
    test_dates = set(test_df["date"])
    overlap = train_dates & test_dates

    if overlap:
        raise ValueError(
            f"LEAKAGE DETECTED: {len(overlap)} date(s) appear in both "
            f"train and test: {sorted(overlap)[:5]}"
        )

    # Verify strict ordering: last train date < first test date
    train_max = train_df["date"].max()
    test_min = test_df["date"].min()
    if train_max >= test_min:
        raise ValueError(
            f"LEAKAGE DETECTED: Train max date ({train_max.date()}) >= "
            f"test min date ({test_min.date()})."
        )

    logging.info(
        f"Zero-overlap confirmed: train ends {train_max.date()}, "
        f"test starts {test_min.date()}."
    )


def log_class_balance(df: pd.DataFrame, label: str) -> None:
    """Log class distribution for a given partition."""
    n = len(df)
    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    logging.info(f"  {label} class balance:")
    logging.info(f"    Up (1):   {n_up:>5d} ({100 * n_up / n:.1f}%)")
    logging.info(f"    Down (0): {n_down:>5d} ({100 * n_down / n:.1f}%)")


def log_split_diagnostics(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log comprehensive split diagnostics."""
    logging.info("-" * 50)
    logging.info("TEMPORAL SPLIT DIAGNOSTICS")
    logging.info(f"  Input rows:      {len(df)}")
    logging.info(f"  Train rows:      {len(train_df)}")
    logging.info(f"  Test rows:       {len(test_df)}")
    logging.info(
        f"  Train date range: "
        f"{train_df['date'].min().date()} to {train_df['date'].max().date()}"
    )
    logging.info(
        f"  Test date range:  "
        f"{test_df['date'].min().date()} to {test_df['date'].max().date()}"
    )
    log_class_balance(train_df, "Train")
    log_class_balance(test_df, "Test")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run temporal train/test split pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — Temporal Train/Test Split")
    logging.info("=" * 60)

    config = load_config()
    split_cfg = config["splitting"]
    input_path = split_cfg["input_path"]
    train_output_path = split_cfg["train_output_path"]
    test_output_path = split_cfg["test_output_path"]
    train_end_date = split_cfg["train_end_date"]

    logging.info(f"Input:          {input_path}")
    logging.info(f"Train output:   {train_output_path}")
    logging.info(f"Test output:    {test_output_path}")
    logging.info(f"Train end date: {train_end_date}")

    # ---- Load ----
    df = pd.read_csv(input_path, parse_dates=["date"])
    logging.info(f"Loaded {len(df)} rows from {input_path}")
    logging.info(
        f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}"
    )

    # ---- Verify dates ----
    df = parse_and_verify_dates(df)

    # ---- Split ----
    train_df, test_df = temporal_split(df, train_end_date)

    # ---- Verify no overlap ----
    verify_no_overlap(train_df, test_df)

    # ---- Diagnostics ----
    log_split_diagnostics(df, train_df, test_df)

    # ---- Save ----
    for path, partition, label in [
        (train_output_path, train_df, "train"),
        (test_output_path, test_df, "test"),
    ]:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        partition.to_csv(out, index=False)
        logging.info(f"Saved {label} set: {out} ({len(partition)} rows)")

    # ---- Final summary ----
    logging.info("=" * 50)
    logging.info("TEMPORAL SPLIT COMPLETE")
    logging.info(f"  Input:  {input_path} ({len(df)} rows)")
    logging.info(f"  Train:  {train_output_path} ({len(train_df)} rows)")
    logging.info(f"  Test:   {test_output_path} ({len(test_df)} rows)")
    logging.info(f"  Split:  {split_cfg['description']}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
