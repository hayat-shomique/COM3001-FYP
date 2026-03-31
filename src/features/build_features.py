"""
COM3001 — Feature Engineering for SPY Daily Direction Prediction
================================================================
Reads the target-bearing dataset (data/processed/spy_targeted.csv),
engineers a controlled set of technical features from price and volume
data, drops warm-up rows where rolling lookback windows produce NaN,
and saves the feature-bearing dataset to data/processed/spy_featured.csv.

ANTI-LEAKAGE DISCIPLINE:
  Every feature uses ONLY data available at or before time t.
  No feature depends on the target column.
  No feature uses future prices, returns, or volume.
  All rolling windows use .shift(1) or look backward only.
  RSI is computed from lagged return series — no same-day information
  leaks into the signal.

This module does NOT:
  - Perform train/test splitting (downstream concern)
  - Scale or normalise features (model-dependent)
  - Select or rank features (evaluation-stage concern)
  - Modify the target column (already constructed upstream)
  - Impute missing values with future-dependent methods

Feature set (14 features):
  Lagged returns:    ret_lag_1, ret_lag_2, ret_lag_5
  Realised vol:      vol_5, vol_10, vol_20
  Momentum:          mom_5, mom_10
  MA ratios:         close_sma5_ratio, close_sma20_ratio, sma5_sma20_ratio
  RSI:               rsi_14
  Volume signals:    volume_chg_1, volume_sma20_ratio

Usage:
    python src/features/build_features.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------

def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged daily returns: ret_lag_1, ret_lag_2, ret_lag_5.

    Each lag_k feature contains the return from k trading days ago.
    shift(k) ensures that row t receives the return from row t-k,
    which was observable at time t. No future information is used.

    Rationale: Lagged returns capture short-term serial dependence
    (autocorrelation) in equity returns. While the EMH implies returns
    are unpredictable, empirical evidence shows weak but exploitable
    autocorrelation at daily frequency, particularly for broad-market
    ETFs (Lo & MacKinlay, 1988, *Review of Financial Studies*).
    """
    for lag in [1, 2, 5]:
        df[f"ret_lag_{lag}"] = df["return"].shift(lag)
    return df


def add_realised_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling realised volatility: vol_5, vol_10, vol_20.

    Computed as the rolling standard deviation of daily returns over
    windows of 5, 10, and 20 trading days. Each window uses .shift(1)
    so that the volatility estimate at time t uses returns from
    t-window to t-1 (inclusive) — no same-day return leakage.

    Rationale: Realised volatility is a core input for both the GeoBM
    baseline (where sigma parameterises the diffusion term) and the
    XGBoost classifier (where volatility regime may predict direction).
    Multiple windows capture short-term (weekly), medium-term
    (fortnightly), and monthly volatility regimes. Volatility clustering
    is one of the most robust stylised facts in financial time series
    (Cont, 2001, *Quantitative Finance*).
    """
    for window in [5, 10, 20]:
        df[f"vol_{window}"] = (
            df["return"]
            .shift(1)
            .rolling(window=window, min_periods=window)
            .std()
        )
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price momentum: mom_5, mom_10.

    Computed as (close_t / close_{t-k}) - 1, using the closing price
    from k days ago. Since close_{t-k} is historical, no future data
    is used.

    Rationale: Momentum — the tendency of recent winners to continue
    outperforming — is one of the most documented anomalies in finance
    (Jegadeesh & Titman, 1993, *Journal of Finance*). Short-term
    momentum (5-day and 10-day) is particularly relevant for daily
    direction prediction on broad-market ETFs.
    """
    for period in [5, 10]:
        df[f"mom_{period}"] = df["close"] / df["close"].shift(period) - 1
    return df


def add_ma_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moving average ratios: close_sma5_ratio, close_sma20_ratio,
    sma5_sma20_ratio.

    close_sma{n}_ratio = close_t / SMA(n)_t, where SMA(n) is the
    n-day simple moving average of closing prices computed up to and
    including day t. This uses today's close, which is observable at
    time t (the prediction target is NEXT-day direction).

    sma5_sma20_ratio = SMA(5)_t / SMA(20)_t — a continuous version
    of the classic short/long MA crossover signal.

    Rationale: MA crossover strategies are among the most widely
    studied systematic trading rules (Brock, Lakonishok & LeBaron,
    1992, *Journal of Finance*). Expressing the signal as a
    continuous ratio rather than a binary cross avoids the information
    loss of thresholding and provides the XGBoost classifier with a
    richer input space.
    """
    sma5 = df["close"].rolling(window=5, min_periods=5).mean()
    sma20 = df["close"].rolling(window=20, min_periods=20).mean()

    df["close_sma5_ratio"] = df["close"] / sma5
    df["close_sma20_ratio"] = df["close"] / sma20
    df["sma5_sma20_ratio"] = sma5 / sma20
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (RSI-14): rsi_14.

    Computed using Wilder's smoothing method (exponential moving average
    with alpha = 1/period) on lagged returns. The return series is
    shifted by 1 before RSI computation, so the RSI at time t reflects
    price changes up to t-1 only — no same-day return leakage.

    The calculation:
      1. Compute daily return changes (shifted by 1 to prevent leakage)
      2. Separate gains (positive changes) and losses (absolute negative)
      3. Apply Wilder's EMA (span=period, adjust=False) to both series
      4. RS = avg_gain / avg_loss
      5. RSI = 100 - (100 / (1 + RS))

    Rationale: RSI is the most widely used bounded oscillator in
    technical analysis (Wilder, 1978, *New Concepts in Technical
    Trading Systems*). It provides a normalised measure of recent
    price momentum that the GA's chromosome can threshold against
    and that XGBoost can learn non-linear splits on.
    """
    # Use lagged returns to prevent same-day leakage
    delta = df["return"].shift(1)

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: EMA with span = period, adjust=False
    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume signals: volume_chg_1, volume_sma20_ratio.

    volume_chg_1 = (volume_t / volume_{t-1}) - 1  (daily volume change)
    volume_sma20_ratio = volume_t / SMA(20, volume)_t

    Today's volume is observable at time t (market close), and the
    prediction target is next-day direction, so using today's volume
    does not create look-ahead bias.

    Rationale: Volume confirms price moves — high volume on an up day
    suggests conviction, low volume suggests fragility. The volume-price
    relationship is a core tenet of technical analysis (Karpoff, 1987,
    *Journal of Financial and Quantitative Analysis*) and provides
    information complementary to the price-based features.
    """
    df["volume_chg_1"] = df["volume"].pct_change(periods=1)
    sma20_vol = df["volume"].rolling(window=20, min_periods=20).mean()
    df["volume_sma20_ratio"] = df["volume"] / sma20_vol
    return df


# ---------------------------------------------------------------------------
# Anti-Leakage Validation
# ---------------------------------------------------------------------------

def validate_no_leakage(df: pd.DataFrame, feature_cols: list) -> None:
    """
    Verify that no feature column is correlated with the target in a
    way that would indicate look-ahead bias. This is a sanity check,
    not a proof — but it catches the most common leakage patterns.

    Checks:
    1. No feature column is identical to the target column.
    2. No feature column has correlation > 0.95 with the target
       (which would suggest the target or next-day return leaked in).
    """
    target = df["target"]
    for col in feature_cols:
        # Check 1: Exact match
        if df[col].equals(target):
            raise ValueError(
                f"LEAKAGE DETECTED: Feature '{col}' is identical to target."
            )
        # Check 2: Suspiciously high correlation
        corr = df[col].corr(target)
        if abs(corr) > 0.95:
            raise ValueError(
                f"LEAKAGE SUSPECTED: Feature '{col}' has correlation "
                f"{corr:.4f} with target."
            )
    logging.info("Anti-leakage validation passed: no feature mirrors the target.")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

# Canonical feature list — the single source of truth for column names
FEATURE_COLS = [
    # Lagged returns
    "ret_lag_1", "ret_lag_2", "ret_lag_5",
    # Realised volatility
    "vol_5", "vol_10", "vol_20",
    # Momentum
    "mom_5", "mom_10",
    # Moving average ratios
    "close_sma5_ratio", "close_sma20_ratio", "sma5_sma20_ratio",
    # RSI
    "rsi_14",
    # Volume
    "volume_chg_1", "volume_sma20_ratio",
]


def build_features(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Reads the target-bearing dataset, adds all 14 features, drops
    warm-up rows, validates anti-leakage, and saves the result.

    Parameters
    ----------
    input_path : str
        Path to spy_targeted.csv.
    output_path : str
        Path to write spy_featured.csv.

    Returns
    -------
    pd.DataFrame
        The feature-bearing dataset.
    """
    # ---- Load ----
    df = pd.read_csv(input_path, parse_dates=["date"])
    n_input = len(df)
    logging.info(f"Loaded {n_input} rows from {input_path}")
    logging.info(
        f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}"
    )
    logging.info(f"  Columns: {list(df.columns)}")

    # ---- Engineer features ----
    df = add_lagged_returns(df)
    df = add_realised_volatility(df)
    df = add_momentum(df)
    df = add_ma_ratios(df)
    df = add_rsi(df)
    df = add_volume_features(df)

    # ---- Log null status BEFORE dropping ----
    null_counts = df[FEATURE_COLS].isnull().sum()
    logging.info("-" * 50)
    logging.info("NULL STATUS (before warm-up drop):")
    for col in FEATURE_COLS:
        nulls = null_counts[col]
        if nulls > 0:
            logging.info(f"  {col:25s} {nulls:>5d} NaN rows")
        else:
            logging.info(f"  {col:25s}     0 NaN rows")

    # ---- Drop warm-up rows ----
    # Rows where ANY feature is NaN are warm-up artefacts from rolling
    # windows. These rows cannot be used by any model.
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    n_warmup = n_before - len(df)
    n_final = len(df)

    logging.info("-" * 50)
    logging.info(f"WARM-UP ROWS DROPPED: {n_warmup}")
    logging.info(f"  (Driven by 20-day rolling windows; vol also shifts by 1)")
    logging.info(f"FINAL RETAINED ROWS:  {n_final}")
    logging.info(
        f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}"
    )

    # ---- Post-drop null verification ----
    remaining_nulls = df[FEATURE_COLS].isnull().sum().sum()
    assert remaining_nulls == 0, (
        f"FATAL: {remaining_nulls} NaN values remain after warm-up drop."
    )
    logging.info(f"  Null features after drop: {remaining_nulls} (verified clean)")

    # ---- Anti-leakage validation ----
    validate_no_leakage(df, FEATURE_COLS)

    # ---- Feature inventory ----
    logging.info("-" * 50)
    logging.info(f"FEATURE INVENTORY ({len(FEATURE_COLS)} features):")
    for i, col in enumerate(FEATURE_COLS, 1):
        mn = df[col].min()
        mx = df[col].max()
        md = df[col].median()
        logging.info(f"  {i:2d}. {col:25s}  min={mn:>10.4f}  median={md:>10.4f}  max={mx:>10.4f}")

    # ---- Class balance after warm-up drop ----
    up = (df["target"] == 1).sum()
    down = (df["target"] == 0).sum()
    logging.info("-" * 50)
    logging.info(f"CLASS BALANCE (post warm-up):")
    logging.info(f"  Up (1):   {up:>5d} ({100*up/n_final:.1f}%)")
    logging.info(f"  Down (0): {down:>5d} ({100*down/n_final:.1f}%)")

    # ---- Save ----
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved featured dataset to {output_path}")

    # ---- Final summary ----
    logging.info("=" * 50)
    logging.info("FEATURE ENGINEERING SUMMARY")
    logging.info(f"  Input rows:          {n_input}")
    logging.info(f"  Warm-up dropped:     {n_warmup}")
    logging.info(f"  Final rows:          {n_final}")
    logging.info(f"  Features added:      {len(FEATURE_COLS)}")
    logging.info(f"  Total columns:       {len(df.columns)}")
    logging.info(f"  Output:              {output_path}")
    logging.info("=" * 50)

    return df


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """Run feature engineering pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — Feature Engineering for SPY Direction Prediction")
    logging.info("=" * 60)

    config = load_config()
    features_cfg = config["features"]
    input_path = features_cfg["input_path"]
    output_path = features_cfg["output_path"]

    logging.info(f"Input:  {input_path}")
    logging.info(f"Output: {output_path}")

    build_features(input_path, output_path)


if __name__ == "__main__":
    main()
