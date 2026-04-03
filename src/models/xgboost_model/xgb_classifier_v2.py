"""
COM3001 — XGBoost-v2: Validated, Early-Stopped Directional Classifier
======================================================================
Addresses the 18.6pp overfitting gap from XGBoost-v1 by applying
chronological validation within the training set, early stopping on
validation logloss, and grid search over a small hyperparameter space.

DESIGN:
  1. Split training set (2010–2022) chronologically:
     - Fitting set: 2010 to validation_end_date (config, default 2020-12-31)
     - Validation set: validation_end_date+1 to 2022-12-30
  2. Grid search over max_depth × n_estimators × learning_rate:
     - For each combination, fit on fitting set with early_stopping on
       validation logloss
     - Record validation accuracy, MCC, and trees actually used
  3. Select best configuration by validation accuracy
  4. Retrain on FULL training set (2010–2022) with selected hyperparameters
     and the early-stopped n_estimators
  5. Evaluate once on test set (2023–2024)

ANTI-LEAKAGE DISCIPLINE:
  The test set is used exactly once, after all tuning and retraining.
  The validation set is carved from training data only — it is posterior
  to the fitting set but anterior to the test set.
  No test data enters fitting, tuning, or model selection at any point.

This module does NOT:
  - Replace v1 — it is a disciplined refinement alongside v1
  - Use test data in any tuning or selection step
  - Scale or normalise features
  - Simulate trading strategies or compute PnL

Usage:
    python src/models/xgboost_model/xgb_classifier_v2.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


FEATURE_COLS = [
    "ret_lag_1", "ret_lag_2", "ret_lag_5",
    "vol_5", "vol_10", "vol_20",
    "mom_5", "mom_10",
    "close_sma5_ratio", "close_sma20_ratio", "sma5_sma20_ratio",
    "rsi_14",
    "volume_chg_1", "volume_sma20_ratio",
]


# ---------------------------------------------------------------------------
# Chronological Validation Split
# ---------------------------------------------------------------------------

def split_train_val(
    train_df: pd.DataFrame, validation_end_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data chronologically into fitting and validation sets.

    The validation set contains all rows strictly after validation_end_date
    (which is the last date of the fitting set). This preserves temporal
    ordering — no shuffling.
    """
    boundary = pd.Timestamp(validation_end_date)
    fit_df = train_df[train_df["date"] <= boundary].reset_index(drop=True)
    val_df = train_df[train_df["date"] > boundary].reset_index(drop=True)
    return fit_df, val_df


# ---------------------------------------------------------------------------
# Grid Search with Early Stopping
# ---------------------------------------------------------------------------

def compute_mcc(target: np.ndarray, predicted: np.ndarray) -> float:
    """Matthews Correlation Coefficient from arrays."""
    tp = int(((predicted == 1) & (target == 1)).sum())
    tn = int(((predicted == 0) & (target == 0)).sum())
    fp = int(((predicted == 1) & (target == 0)).sum())
    fn = int(((predicted == 0) & (target == 1)).sum())
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return num / den if den > 0 else 0.0


def grid_search(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    seed: int,
) -> list[dict]:
    """
    Exhaustive grid search with early stopping on validation logloss.

    For each hyperparameter combination:
      - Fit on fitting set with eval_set=validation, early_stopping
      - Record validation accuracy, MCC, and number of trees used

    Returns list of result dicts sorted by validation accuracy (descending).
    """
    X_fit = fit_df[FEATURE_COLS].values
    y_fit = fit_df["target"].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["target"].values

    depths = cfg["grid_max_depth"]
    estimators = cfg["grid_n_estimators"]
    rates = cfg["grid_learning_rate"]
    early_stop = cfg["early_stopping_rounds"]

    results = []

    for depth, n_est, lr in product(depths, estimators, rates):
        model = xgb.XGBClassifier(
            max_depth=depth,
            n_estimators=n_est,
            learning_rate=lr,
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=early_stop,
            random_state=seed,
            use_label_encoder=False,
            verbosity=0,
        )

        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        trees_used = model.best_iteration + 1
        val_preds = model.predict(X_val)
        val_acc = float((val_preds == y_val).mean())
        val_mcc = compute_mcc(y_val, val_preds)

        results.append({
            "max_depth": depth,
            "n_estimators": n_est,
            "learning_rate": lr,
            "trees_used": trees_used,
            "val_accuracy": val_acc,
            "val_mcc": val_mcc,
        })

    results.sort(key=lambda r: r["val_accuracy"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run XGBoost-v2 validated classifier pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — XGBoost-v2: Validated, Early-Stopped Classifier")
    logging.info("=" * 60)

    config = load_config()
    cfg = config["xgboost_v2"]
    seed = config["reproducibility"]["random_seed"]

    train_path = cfg["train_path"]
    test_path = cfg["test_path"]
    predictions_path = cfg["predictions_path"]
    val_end = cfg["validation_end_date"]

    logging.info(f"Train input:        {train_path}")
    logging.info(f"Test input:         {test_path}")
    logging.info(f"Predictions output: {predictions_path}")
    logging.info(f"Validation boundary: {val_end}")
    logging.info(f"Random seed:        {seed}")

    # ---- Load data ----
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    logging.info(f"Loaded train: {len(train_df)} rows "
                 f"({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logging.info(f"Loaded test:  {len(test_df)} rows "
                 f"({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # ---- Chronological validation split ----
    fit_df, val_df = split_train_val(train_df, val_end)

    logging.info("-" * 50)
    logging.info("CHRONOLOGICAL VALIDATION SPLIT (within training)")
    logging.info(f"  Fitting set:     {len(fit_df)} rows "
                 f"({fit_df['date'].min().date()} to {fit_df['date'].max().date()})")
    logging.info(f"  Validation set:  {len(val_df)} rows "
                 f"({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    logging.info(f"  Total:           {len(fit_df) + len(val_df)} "
                 f"(= {len(train_df)} training rows)")

    # ---- Grid search configuration ----
    n_combos = (len(cfg["grid_max_depth"]) *
                len(cfg["grid_n_estimators"]) *
                len(cfg["grid_learning_rate"]))

    logging.info("-" * 50)
    logging.info("GRID SEARCH CONFIGURATION")
    logging.info(f"  max_depth:         {cfg['grid_max_depth']}")
    logging.info(f"  n_estimators:      {cfg['grid_n_estimators']}")
    logging.info(f"  learning_rate:     {cfg['grid_learning_rate']}")
    logging.info(f"  subsample:         {cfg['subsample']}")
    logging.info(f"  colsample_bytree:  {cfg['colsample_bytree']}")
    logging.info(f"  early_stopping:    {cfg['early_stopping_rounds']} rounds")
    logging.info(f"  Total combinations: {n_combos}")

    # ---- Run grid search ----
    results = grid_search(fit_df, val_df, cfg, seed)

    logging.info("-" * 50)
    logging.info("GRID SEARCH RESULTS (all 18 combinations)")
    logging.info(f"  {'depth':>5s} {'n_est':>5s} {'lr':>6s} {'trees':>5s} "
                 f"{'val_acc':>8s} {'val_mcc':>8s}")
    logging.info(f"  {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*8} {'-'*8}")

    for r in results:
        marker = " <-- BEST" if r is results[0] else ""
        logging.info(
            f"  {r['max_depth']:>5d} {r['n_estimators']:>5d} "
            f"{r['learning_rate']:>6.2f} {r['trees_used']:>5d} "
            f"{100 * r['val_accuracy']:>7.1f}% "
            f"{r['val_mcc']:>8.4f}{marker}"
        )

    best = results[0]
    logging.info("-" * 50)
    logging.info("BEST CONFIGURATION")
    logging.info(f"  max_depth:      {best['max_depth']}")
    logging.info(f"  n_estimators:   {best['trees_used']} "
                 f"(early-stopped from {best['n_estimators']})")
    logging.info(f"  learning_rate:  {best['learning_rate']}")
    logging.info(f"  val_accuracy:   {100 * best['val_accuracy']:.1f}%")
    logging.info(f"  val_mcc:        {best['val_mcc']:.4f}")

    # ---- Retrain on full training set with best config ----
    logging.info("-" * 50)
    logging.info("RETRAINING ON FULL TRAINING SET")

    final_model = xgb.XGBClassifier(
        max_depth=best["max_depth"],
        n_estimators=best["trees_used"],
        learning_rate=best["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        use_label_encoder=False,
        verbosity=0,
    )

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values
    final_model.fit(X_train, y_train)

    train_preds = final_model.predict(X_train)
    train_accuracy = float((train_preds == y_train).mean())
    train_mcc = compute_mcc(y_train, train_preds)
    logging.info(f"  Training accuracy: {train_accuracy:.4f} "
                 f"({100 * train_accuracy:.1f}%)")
    logging.info(f"  Training MCC:      {train_mcc:.4f}")

    # ---- Evaluate on test set ----
    X_test = test_df[FEATURE_COLS].values
    p_up = final_model.predict_proba(X_test)[:, 1]
    test_preds = (p_up >= 0.5).astype(int)

    predictions = pd.DataFrame({
        "date": test_df["date"].values,
        "target": test_df["target"].values,
        "predicted": test_preds,
        "p_up": p_up,
        "model": "xgboost_v2",
    })

    n_test = len(predictions)
    target_arr = predictions["target"].values
    correct = int((test_preds == target_arr).sum())
    test_accuracy = correct / n_test
    test_mcc = compute_mcc(target_arr, test_preds)

    n_up_actual = int((target_arr == 1).sum())
    n_down_actual = int((target_arr == 0).sum())
    majority_baseline = max(n_up_actual, n_down_actual) / n_test

    n_pred_up = int((test_preds == 1).sum())
    n_pred_down = int((test_preds == 0).sum())
    is_constant = (n_pred_up == 0) or (n_pred_down == 0)

    train_test_gap = train_accuracy - test_accuracy

    logging.info("-" * 50)
    logging.info("TEST EVALUATION")
    logging.info(f"  Test accuracy:         {test_accuracy:.4f} "
                 f"({100 * test_accuracy:.1f}%)")
    logging.info(f"  Test MCC:              {test_mcc:.4f}")
    logging.info(f"  Majority baseline:     {majority_baseline:.4f} "
                 f"({100 * majority_baseline:.1f}%)")
    logging.info(f"  GeoBM baseline:        0.5749 (57.5%)")
    logging.info(f"  GA baseline:           0.5669 (56.7%)")
    logging.info(f"  XGBoost-v1:            0.5309 (53.1%)")

    if test_accuracy > majority_baseline:
        delta = test_accuracy - majority_baseline
        logging.info(f"  Result: EXCEEDS majority baseline by "
                     f"{100 * delta:.1f}pp")
    else:
        logging.info("  Result: DOES NOT exceed majority-class baseline")

    logging.info("-" * 50)
    logging.info("PREDICTION CLASS BALANCE (test set)")
    logging.info(f"  Predicted up (1):      {n_pred_up:>5d} "
                 f"({100 * n_pred_up / n_test:.1f}%)")
    logging.info(f"  Predicted down (0):    {n_pred_down:>5d} "
                 f"({100 * n_pred_down / n_test:.1f}%)")
    if is_constant:
        constant_class = "up" if n_pred_up == n_test else "down"
        logging.info(f"  WARNING: Constant '{constant_class}' predictor.")
    else:
        logging.info("  Model produces non-constant predictions.")

    logging.info("-" * 50)
    logging.info("OVERFITTING CHECK")
    logging.info(f"  Training accuracy:     {train_accuracy:.4f} "
                 f"({100 * train_accuracy:.1f}%)")
    logging.info(f"  Test accuracy:         {test_accuracy:.4f} "
                 f"({100 * test_accuracy:.1f}%)")
    logging.info(f"  Train-test gap:        {train_test_gap:+.4f} "
                 f"({100 * train_test_gap:+.1f}pp)")
    logging.info(f"  v1 train-test gap:     +18.6pp")
    if train_test_gap < 0.186:
        logging.info(f"  Gap REDUCED by {100 * (0.186 - train_test_gap):.1f}pp "
                     f"vs v1")
    else:
        logging.info("  Gap NOT reduced vs v1")

    # ---- v1-vs-v2 comparison table ----
    logging.info("-" * 70)
    logging.info("V1-VS-V2 COMPARISON TABLE")
    logging.info(f"  {'Model':<14s} {'Train':>7s} {'Test':>7s} "
                 f"{'Gap':>8s} {'MCC':>7s} {'Pred↑%':>7s}")
    logging.info(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")
    logging.info(f"  {'XGBoost-v1':<14s} {'71.7%':>7s} {'53.1%':>7s} "
                 f"{'+18.6pp':>8s} {'-0.037':>7s} {'80.4%':>7s}")
    logging.info(f"  {'XGBoost-v2':<14s} "
                 f"{100 * train_accuracy:>6.1f}% "
                 f"{100 * test_accuracy:>6.1f}% "
                 f"{100 * train_test_gap:>+7.1f}pp "
                 f"{test_mcc:>7.4f} "
                 f"{100 * n_pred_up / n_test:>6.1f}%")
    logging.info(f"  {'Majority':<14s} {'—':>7s} {'57.5%':>7s} "
                 f"{'—':>8s} {'0.000':>7s} {'100.0%':>7s}")
    logging.info("-" * 70)

    # ---- Save predictions ----
    out = Path(predictions_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    logging.info(f"Saved predictions: {out} ({len(predictions)} rows)")

    # ---- Final summary ----
    logging.info("=" * 50)
    logging.info("XGBoost-v2 COMPLETE")
    logging.info(f"  Fitting rows:          {len(fit_df)}")
    logging.info(f"  Validation rows:       {len(val_df)}")
    logging.info(f"  Full train rows:       {len(train_df)}")
    logging.info(f"  Test rows:             {n_test}")
    logging.info(f"  Best config:           depth={best['max_depth']}, "
                 f"trees={best['trees_used']}, lr={best['learning_rate']}")
    logging.info(f"  Training accuracy:     {100 * train_accuracy:.1f}%")
    logging.info(f"  Test accuracy:         {100 * test_accuracy:.1f}%")
    logging.info(f"  Test MCC:              {test_mcc:.4f}")
    logging.info(f"  Train-test gap:        {100 * train_test_gap:+.1f}pp "
                 f"(v1: +18.6pp)")
    logging.info(f"  Majority baseline:     57.5%")
    logging.info(f"  Constant predictor:    {is_constant}")
    logging.info(f"  Output:                {predictions_path}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
