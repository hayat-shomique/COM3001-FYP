"""
COM3001 — Supervised Learning Target Construction
==================================================
Reads the cleaned interim dataset, constructs the binary next-day
direction label from CRSP's adjusted return field, drops the final
row (no observable next-day outcome), and saves to data/processed/.

This module crosses the boundary from data engineering into modelling
assumptions. The target definition — binary next-day direction — is
an explicit supervised learning design choice, not a data property.

This module contains NO feature engineering and NO train/test splitting.

Usage:
    python src/data/build_target.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def construct_binary_target(df: pd.DataFrame, shift: int = 1) -> pd.DataFrame:
    df = df.copy()
    future_return = df["return"].shift(-shift)
    df["target"] = (future_return > 0).astype(float)
    df.loc[future_return.isna(), "target"] = float("nan")
    logging.info(
        f"Target constructed: binary direction with shift={shift}. "
        f"Rows with observable target: {df['target'].notna().sum()}, "
        f"rows without (final {shift}): {df['target'].isna().sum()}."
    )
    return df


def drop_unobservable_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    n_dropped = n_before - len(df)
    logging.info(f"Dropped {n_dropped} unobservable row(s) (no next-day outcome).")
    return df


def log_target_diagnostics(df: pd.DataFrame) -> None:
    logging.info("-" * 40)
    logging.info("TARGET DIAGNOSTICS")
    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    total = len(df)
    logging.info(f"  Class 1 (up):   {n_up} ({n_up/total*100:.1f}%)")
    logging.info(f"  Class 0 (down): {n_down} ({n_down/total*100:.1f}%)")
    logging.info(f"  Total rows:     {total}")
    n_nulls = df.isnull().sum().sum()
    if n_nulls > 0:
        logging.error(f"  UNEXPECTED: {n_nulls} null values remain.")
    else:
        logging.info("  No null values in any column.")
    logging.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logging.info(f"  Columns: {list(df.columns)}")


def save_targeted(df: pd.DataFrame, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved targeted dataset: {path}")
    return path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("=" * 50)
    logging.info("COM3001 — Supervised Learning Target Construction")
    logging.info("=" * 50)
    config = load_config()
    target_cfg = config["target"]
    input_path = target_cfg["input_path"]
    output_path = target_cfg["output_path"]
    shift = target_cfg["shift"]
    logging.info(f"Loading interim data: {input_path}")
    df = pd.read_csv(input_path)
    input_row_count = len(df)
    logging.info(f"Loaded {input_row_count} rows.")
    df = construct_binary_target(df, shift=shift)
    df = drop_unobservable_rows(df)
    log_target_diagnostics(df)
    save_targeted(df, output_path)
    logging.info("=" * 50)
    logging.info("TARGET CONSTRUCTION COMPLETE")
    logging.info(f"  Input:   {input_path} ({input_row_count} rows)")
    logging.info(f"  Output:  {output_path} ({len(df)} rows)")
    logging.info(f"  Dropped: {input_row_count - len(df)} row(s)")
    logging.info(f"  Target:  {target_cfg['description']}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
