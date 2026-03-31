"""
COM3001 — WRDS CRSP Data Ingestion for SPY
===========================================
Fetches daily OHLCV data for SPY from WRDS CRSP and saves the RAW
result to data/raw/. No cleaning or preprocessing — single responsibility.

CRSP identifier resolution:
  SPY's PERMNO is resolved at runtime via crsp.dsenames rather than
  hardcoded, demonstrating programmatic use of CRSP's identifier system.

Usage:
    python src/data/fetch_wrds.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import os
import sys
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
# WRDS Connection
# ---------------------------------------------------------------------------

def connect_wrds():
    """
    Establish connection to WRDS PostgreSQL database.

    Uses ~/.pgpass for cached credentials after first interactive login.
    """
    try:
        import wrds
        db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
        logging.info("WRDS connection established.")
        return db
    except Exception as e:
        logging.error(f"WRDS connection failed: {e}")
        logging.error(
            "Run: python -c \"import wrds; wrds.Connection()\" "
            "to cache credentials."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# PERMNO Resolution
# ---------------------------------------------------------------------------

def resolve_permno(db, ticker: str) -> int:
    """
    Resolve a ticker symbol to its CRSP PERMNO via crsp.dsenames.

    CRSP uses PERMNO as its permanent security identifier. Rather than
    hardcoding this, we resolve it programmatically to demonstrate
    proper use of the CRSP identifier system.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS database connection.
    ticker : str
        Ticker symbol to resolve (e.g. 'SPY').

    Returns
    -------
    int
        CRSP PERMNO for the given ticker.

    Raises
    ------
    SystemExit
        If ticker cannot be resolved to exactly one PERMNO.
    """
    query = f"""
        SELECT DISTINCT permno
        FROM crsp.dsenames
        WHERE ticker = '{ticker}'
    """
    result = db.raw_sql(query)

    if result.empty:
        logging.error(f"No PERMNO found for ticker '{ticker}' in crsp.dsenames.")
        sys.exit(1)

    if len(result) > 1:
        logging.error(
            f"Ambiguous: multiple PERMNOs for '{ticker}': "
            f"{result['permno'].tolist()}. "
            f"Cannot proceed — resolve manually and set permno in config."
        )
        sys.exit(1)

    permno = int(result["permno"].iloc[0])
    logging.info(f"Resolved {ticker} -> PERMNO {permno}")
    return permno


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_spy_daily(db, config: dict, permno: int) -> pd.DataFrame:
    """
    Fetch SPY daily data from CRSP Daily Stock File.

    Returns the RAW data exactly as CRSP provides it — no cleaning,
    no transformations. Preprocessing is handled by a separate module.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS database connection.
    config : dict
        Data configuration with period and field parameters.
    permno : int
        Resolved CRSP PERMNO for the target asset.

    Returns
    -------
    pd.DataFrame
        Raw daily data from crsp.dsf.
    """
    start = config["period"]["start_date"]
    end = config["period"]["end_date"]

    query = f"""
        SELECT
            a.date,
            a.prc,
            a.openprc,
            a.askhi,
            a.bidlo,
            a.vol,
            a.ret
        FROM
            crsp.dsf AS a
        WHERE
            a.permno = {permno}
            AND a.date BETWEEN '{start}' AND '{end}'
        ORDER BY
            a.date ASC
    """

    logging.info(f"Querying CRSP: PERMNO={permno}, {start} to {end}")
    df = db.raw_sql(query)
    logging.info(f"Retrieved {len(df)} rows from crsp.dsf.")

    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def log_diagnostics(df: pd.DataFrame) -> None:
    """
    Log data quality diagnostics for the raw CRSP data.

    Reports null counts, duplicate dates, negative prices (CRSP
    convention), and basic summary statistics. This provides an
    auditable record of what CRSP returned before any preprocessing.
    """
    logging.info("-" * 40)
    logging.info("RAW DATA DIAGNOSTICS")

    # Null counts per column
    nulls = df.isnull().sum()
    for col, count in nulls.items():
        if count > 0:
            logging.info(f"  Nulls in {col}: {count} ({count/len(df)*100:.1f}%)")
    if nulls.sum() == 0:
        logging.info("  No null values in any column.")

    # Duplicate dates
    n_dupes = df["date"].duplicated().sum()
    if n_dupes > 0:
        logging.warning(f"  Duplicate dates found: {n_dupes}")
    else:
        logging.info("  No duplicate dates.")

    # Negative prices (CRSP bid/ask midpoint convention)
    n_neg = (df["prc"] < 0).sum()
    logging.info(
        f"  Negative PRC values: {n_neg} "
        f"({n_neg/len(df)*100:.1f}% — CRSP bid/ask midpoint convention)"
    )

    # Date range and basic stats
    dates = pd.to_datetime(df["date"])
    logging.info(f"  Date range: {dates.min().date()} to {dates.max().date()}")
    logging.info(f"  Trading days: {len(df)}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame, output_path: str) -> Path:
    """Save raw data to CSV at the configured output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved raw data: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Full data ingestion pipeline: connect, resolve, fetch, diagnose, save."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 50)
    logging.info("COM3001 — SPY Data Ingestion (WRDS CRSP)")
    logging.info("=" * 50)

    # Load config
    config = load_config()
    ticker = config["asset"]["ticker"]
    logging.info(f"Target: {ticker} ({config['period']['start_date']} to {config['period']['end_date']})")

    # Connect to WRDS
    db = connect_wrds()

    try:
        # Resolve ticker -> PERMNO via crsp.dsenames
        permno = config["asset"]["permno"]
        if permno is None:
            permno = resolve_permno(db, ticker)
        else:
            logging.info(f"Using configured PERMNO: {permno}")

        # Fetch raw data
        raw_df = fetch_spy_daily(db, config, permno)

        if raw_df.empty:
            logging.error("No data returned from CRSP. Check PERMNO and date range.")
            sys.exit(1)

        # Diagnostics on raw data
        log_diagnostics(raw_df)

        # Save
        output_path = config["output"]["raw_path"]
        save_raw(raw_df, output_path)

        # Summary
        logging.info("=" * 50)
        logging.info("INGESTION COMPLETE")
        logging.info(f"  Asset:    {ticker} (PERMNO {permno})")
        logging.info(f"  Rows:     {len(raw_df)}")
        logging.info(f"  Output:   {output_path}")
        logging.info("=" * 50)

    finally:
        db.close()
        logging.info("WRDS connection closed.")


if __name__ == "__main__":
    main()
