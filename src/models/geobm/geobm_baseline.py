"""
COM3001 — Geometric Brownian Motion Directional Baseline
=========================================================
Estimates drift (mu) and volatility (sigma) from the training-period
log returns, then derives a one-step-ahead directional prediction for
every day in the test set using the analytic probability that the next
log return exceeds zero under the GBM assumptions.

MODEL ASSUMPTIONS:
  GeoBM assumes that log returns are IID draws from a normal distribution:
      log(S_{t+1} / S_t) ~ N(mu - 0.5 * sigma^2, sigma^2)
  where mu is the drift rate and sigma is the volatility, both estimated
  from the training-period log-return series at daily frequency.

PREDICTION RULE:
  The probability that the next day's simple return is positive equals:
      P(r_{t+1} > 0) = P(log(S_{t+1}/S_t) > 0)
                      = Phi((mu - 0.5 * sigma^2) / sigma)
  where Phi is the standard normal CDF.

  If this probability exceeds 0.5, predict up (1); otherwise predict
  down (0). Under GBM, this probability is constant across all days —
  the model produces the same prediction for every test day. This is
  not a bug; it is a direct consequence of the IID assumption and is
  itself an important finding about the limits of drift-based
  directional prediction.

ANTI-LEAKAGE DISCIPLINE:
  Parameters are estimated from training data only.
  No test-period return, price, or feature enters parameter estimation.
  The engineered feature columns are not used — GeoBM operates on the
  raw return series, consistent with its role as a stochastic baseline
  that assumes returns are IID draws from a known distribution.

This module does NOT:
  - Use engineered features as predictive inputs
  - Perform rolling or adaptive parameter re-estimation
  - Simulate price paths or trading strategies
  - Compute PnL, Sharpe ratios, or portfolio-level metrics
  - Scale or normalise any data

Usage:
    python src/models/geobm/geobm_baseline.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Parameter Estimation
# ---------------------------------------------------------------------------

def estimate_parameters(train_df: pd.DataFrame) -> dict:
    """
    Estimate GBM drift and volatility from training-period returns.

    The CRSP 'return' column contains simple daily returns:
        r_t = (S_t - S_{t-1}) / S_{t-1}

    GeoBM operates on log returns:
        x_t = log(1 + r_t) = log(S_t / S_{t-1})

    Under GBM, log returns are IID normal:
        x_t ~ N(mu - 0.5 * sigma^2, sigma^2)

    The sample mean of x_t estimates (mu - 0.5 * sigma^2), and the
    sample standard deviation estimates sigma. From these, mu is
    recovered as:
        mu = mean(x) + 0.5 * std(x)^2

    All estimates are at daily frequency — no annualisation is applied,
    because the prediction horizon is also daily.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with a 'return' column (simple daily returns).

    Returns
    -------
    dict with keys: mu, sigma, log_return_mean, log_return_std,
                    n_train, p_up_analytic
    """
    simple_returns = train_df["return"].values

    # Convert simple returns to log returns
    log_returns = np.log(1 + simple_returns)

    # Check for any invalid values
    n_invalid = np.sum(~np.isfinite(log_returns))
    if n_invalid > 0:
        raise ValueError(
            f"FATAL: {n_invalid} non-finite log return(s) found. "
            "Cannot estimate GBM parameters."
        )

    # Sample statistics of log returns (daily frequency)
    log_return_mean = np.mean(log_returns)  # estimates (mu - 0.5*sigma^2)
    log_return_std = np.std(log_returns, ddof=1)  # estimates sigma

    # Recover GBM parameters
    sigma = log_return_std
    mu = log_return_mean + 0.5 * sigma**2

    # Analytic probability that next log return > 0
    # P(x > 0) = P(Z > -log_return_mean / sigma) = Phi(log_return_mean / sigma)
    p_up = norm.cdf(log_return_mean / sigma)

    return {
        "mu": mu,
        "sigma": sigma,
        "log_return_mean": log_return_mean,
        "log_return_std": log_return_std,
        "n_train": len(log_returns),
        "p_up_analytic": p_up,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def generate_predictions(
    test_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Generate directional predictions for the test set.

    Under GBM, the probability of a positive next-day return is constant
    and equal to Phi((mu - 0.5*sigma^2) / sigma). This probability is
    the same for every test day — the model produces a constant
    prediction vector.

    The prediction rule is:
        predict 1 (up) if p_up > 0.5
        predict 0 (down) if p_up <= 0.5

    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with 'date' and 'target' columns.
    params : dict
        Estimated GBM parameters from estimate_parameters().

    Returns
    -------
    pd.DataFrame with columns: date, target, predicted, p_up, model
    """
    p_up = params["p_up_analytic"]
    predicted_class = 1 if p_up > 0.5 else 0

    predictions = pd.DataFrame({
        "date": test_df["date"].values,
        "target": test_df["target"].values,
        "predicted": predicted_class,
        "p_up": p_up,
        "model": "geobm",
    })

    return predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(predictions: pd.DataFrame, params: dict) -> dict:
    """
    Evaluate directional predictions against actual outcomes and the
    majority-class baseline.

    Returns
    -------
    dict with evaluation metrics.
    """
    n_test = len(predictions)
    correct = (predictions["predicted"] == predictions["target"]).sum()
    accuracy = correct / n_test

    # Majority-class baseline
    n_up_actual = (predictions["target"] == 1).sum()
    n_down_actual = (predictions["target"] == 0).sum()
    majority_baseline = max(n_up_actual, n_down_actual) / n_test

    # Prediction class balance
    n_pred_up = (predictions["predicted"] == 1).sum()
    n_pred_down = (predictions["predicted"] == 0).sum()

    # Is the model a constant predictor?
    is_constant = (n_pred_up == 0) or (n_pred_down == 0)
    constant_class = None
    if is_constant:
        constant_class = "up" if n_pred_up == n_test else "down"

    return {
        "n_test": n_test,
        "correct": correct,
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "exceeds_baseline": accuracy > majority_baseline,
        "n_up_actual": n_up_actual,
        "n_down_actual": n_down_actual,
        "n_pred_up": n_pred_up,
        "n_pred_down": n_pred_down,
        "is_constant_predictor": is_constant,
        "constant_class": constant_class,
        "p_up_analytic": params["p_up_analytic"],
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_parameter_diagnostics(params: dict) -> None:
    """Log estimated GBM parameters."""
    logging.info("-" * 50)
    logging.info("GBM PARAMETER ESTIMATES (daily frequency)")
    logging.info(f"  Training observations:     {params['n_train']}")
    logging.info(f"  Return definition:         log(1 + r_simple)")
    logging.info(f"  Log-return mean:           {params['log_return_mean']:.8f}")
    logging.info(f"  Log-return std (sigma):    {params['log_return_std']:.8f}")
    logging.info(f"  Drift (mu):                {params['mu']:.8f}")
    logging.info(f"  Volatility (sigma):        {params['sigma']:.8f}")
    logging.info(f"  Drift correction term:     {0.5 * params['sigma']**2:.8f}")
    logging.info(f"  P(next return > 0):        {params['p_up_analytic']:.6f}")


def log_evaluation_diagnostics(metrics: dict) -> None:
    """Log evaluation results."""
    logging.info("-" * 50)
    logging.info("EVALUATION RESULTS")
    logging.info(f"  Test observations:         {metrics['n_test']}")
    logging.info(f"  Correct predictions:       {metrics['correct']}")
    logging.info(f"  Accuracy:                  {metrics['accuracy']:.4f} "
                 f"({100 * metrics['accuracy']:.1f}%)")
    logging.info(f"  Majority-class baseline:   {metrics['majority_baseline']:.4f} "
                 f"({100 * metrics['majority_baseline']:.1f}%)")

    if metrics['exceeds_baseline']:
        logging.info("  Result: EXCEEDS majority-class baseline")
    else:
        logging.info("  Result: DOES NOT exceed majority-class baseline")

    logging.info("-" * 50)
    logging.info("PREDICTION CLASS BALANCE")
    logging.info(f"  Predicted up (1):          {metrics['n_pred_up']:>5d} "
                 f"({100 * metrics['n_pred_up'] / metrics['n_test']:.1f}%)")
    logging.info(f"  Predicted down (0):        {metrics['n_pred_down']:>5d} "
                 f"({100 * metrics['n_pred_down'] / metrics['n_test']:.1f}%)")

    if metrics['is_constant_predictor']:
        logging.info(f"  WARNING: Model is a constant '{metrics['constant_class']}' "
                     f"predictor.")
        logging.info(f"  This is expected when the estimated drift is positive: "
                     f"P(up) = {metrics['p_up_analytic']:.6f} > 0.5, so the "
                     f"analytic rule predicts 'up' on every day.")
        logging.info(f"  The model's accuracy ({100 * metrics['accuracy']:.1f}%) "
                     f"equals the up-day fraction in the test set.")

    logging.info("-" * 50)
    logging.info("ACTUAL CLASS BALANCE (test set)")
    logging.info(f"  Actual up (1):             {metrics['n_up_actual']:>5d} "
                 f"({100 * metrics['n_up_actual'] / metrics['n_test']:.1f}%)")
    logging.info(f"  Actual down (0):           {metrics['n_down_actual']:>5d} "
                 f"({100 * metrics['n_down_actual'] / metrics['n_test']:.1f}%)")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run GeoBM directional baseline pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — Geometric Brownian Motion Directional Baseline")
    logging.info("=" * 60)

    config = load_config()
    geobm_cfg = config["geobm"]
    seed = config["reproducibility"]["random_seed"]

    train_path = geobm_cfg["train_path"]
    test_path = geobm_cfg["test_path"]
    predictions_path = geobm_cfg["predictions_path"]

    logging.info(f"Train input:       {train_path}")
    logging.info(f"Test input:        {test_path}")
    logging.info(f"Predictions output: {predictions_path}")
    logging.info(f"Random seed:       {seed} (not used — analytic method)")

    # ---- Load data ----
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    logging.info(f"Loaded train: {len(train_df)} rows "
                 f"({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logging.info(f"Loaded test:  {len(test_df)} rows "
                 f"({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # ---- Estimate parameters from training data only ----
    params = estimate_parameters(train_df)
    log_parameter_diagnostics(params)

    # ---- Generate predictions ----
    predictions = generate_predictions(test_df, params)

    # ---- Evaluate ----
    metrics = evaluate_predictions(predictions, params)
    log_evaluation_diagnostics(metrics)

    # ---- Save predictions ----
    out = Path(predictions_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    logging.info(f"Saved predictions: {out} ({len(predictions)} rows)")

    # ---- Final summary ----
    logging.info("=" * 50)
    logging.info("GeoBM BASELINE COMPLETE")
    logging.info(f"  Train rows:              {params['n_train']}")
    logging.info(f"  Test rows:               {metrics['n_test']}")
    logging.info(f"  Estimated mu (daily):    {params['mu']:.8f}")
    logging.info(f"  Estimated sigma (daily): {params['sigma']:.8f}")
    logging.info(f"  P(next return > 0):      {params['p_up_analytic']:.6f}")
    logging.info(f"  Prediction method:       analytic (no simulation)")
    logging.info(f"  Accuracy:                {100 * metrics['accuracy']:.1f}%")
    logging.info(f"  Majority baseline:       {100 * metrics['majority_baseline']:.1f}%")
    logging.info(f"  Constant predictor:      {metrics['is_constant_predictor']}")
    logging.info(f"  Output:                  {predictions_path}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
