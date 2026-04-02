"""
COM3001 — XGBoost Directional Classifier
==========================================
Trains a gradient-boosted tree classifier (XGBoost) on the 14 engineered
features from the training set, generates predicted probabilities for the
held-out test set, and saves comparison-compatible predictions.

MODEL DESIGN:
  XGBoost learns an ensemble of shallow decision trees that partition
  the 14-dimensional feature space into regions with different predicted
  class probabilities. Unlike GeoBM (2 parameters, no features) and the
  GA (3 threshold rules, implicit feature selection), XGBoost can learn
  non-linear interactions across all 14 features simultaneously.

HYPERPARAMETERS:
  max_depth=3, n_estimators=100, learning_rate=0.1, subsample=0.8,
  colsample_bytree=0.8. These are conservative defaults chosen to limit
  overfitting on 3,252 training rows. All values are read from config.

ANTI-LEAKAGE DISCIPLINE:
  The model is fitted on training data only. No test data enters
  fitting, hyperparameter selection, or feature importance computation.
  Predicted probabilities are generated on the test set exactly once
  after training completes.

This module does NOT:
  - Use test data in training, tuning, or feature selection
  - Perform hyperparameter search (fixed defaults from config)
  - Scale or normalise features (XGBoost is invariant to monotonic
    transformations)
  - Simulate trading strategies or compute PnL
  - Perform walk-forward validation (evaluation-stage concern)

Usage:
    python src/models/xgboost_model/xgb_classifier.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
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


# Canonical feature list — must match build_features.py FEATURE_COLS
FEATURE_COLS = [
    "ret_lag_1", "ret_lag_2", "ret_lag_5",
    "vol_5", "vol_10", "vol_20",
    "mom_5", "mom_10",
    "close_sma5_ratio", "close_sma20_ratio", "sma5_sma20_ratio",
    "rsi_14",
    "volume_chg_1", "volume_sma20_ratio",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame, xgb_cfg: dict, seed: int
) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary classifier on the 14 engineered features.

    Hyperparameters are read from config and fixed before training.
    No hyperparameter search is performed — the values are conservative
    defaults justified in the component note.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with FEATURE_COLS and 'target' columns.
    xgb_cfg : dict
        XGBoost config block from data_config.yaml.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xgb.XGBClassifier
        Fitted classifier.
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values

    model = xgb.XGBClassifier(
        max_depth=xgb_cfg["max_depth"],
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        use_label_encoder=False,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def generate_predictions(
    model: xgb.XGBClassifier, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate directional predictions and probabilities for the test set.

    Uses predict_proba to get calibrated class probabilities, then
    thresholds at 0.5 for binary predictions.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Fitted classifier.
    test_df : pd.DataFrame
        Test data with FEATURE_COLS, 'date', and 'target' columns.

    Returns
    -------
    pd.DataFrame with columns: date, target, predicted, p_up, model
    """
    X_test = test_df[FEATURE_COLS].values

    # Predicted probability of class 1 (up)
    p_up = model.predict_proba(X_test)[:, 1]

    # Binary prediction at 0.5 threshold
    predicted = (p_up >= 0.5).astype(int)

    predictions = pd.DataFrame({
        "date": test_df["date"].values,
        "target": test_df["target"].values,
        "predicted": predicted,
        "p_up": p_up,
        "model": "xgboost",
    })

    return predictions


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def get_feature_importance(model: xgb.XGBClassifier) -> pd.DataFrame:
    """
    Extract built-in feature importance (gain-based) from the fitted model.

    Gain measures the average improvement in the loss function contributed
    by each feature across all splits in all trees where it appears.

    Returns
    -------
    pd.DataFrame with columns: feature, importance (sorted descending)
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run XGBoost directional classifier pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — XGBoost Directional Classifier")
    logging.info("=" * 60)

    config = load_config()
    xgb_cfg = config["xgboost"]
    seed = config["reproducibility"]["random_seed"]

    train_path = xgb_cfg["train_path"]
    test_path = xgb_cfg["test_path"]
    predictions_path = xgb_cfg["predictions_path"]

    logging.info(f"Train input:        {train_path}")
    logging.info(f"Test input:         {test_path}")
    logging.info(f"Predictions output: {predictions_path}")
    logging.info(f"Random seed:        {seed}")

    # ---- Load data ----
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    logging.info(f"Loaded train: {len(train_df)} rows "
                 f"({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logging.info(f"Loaded test:  {len(test_df)} rows "
                 f"({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # ---- Verify features ----
    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        raise ValueError(f"FATAL: Missing feature columns: {missing}")
    logging.info(f"Feature columns:    {len(FEATURE_COLS)} confirmed present")

    # ---- Log hyperparameters ----
    logging.info("-" * 50)
    logging.info("XGBOOST HYPERPARAMETERS (fixed, from config)")
    logging.info(f"  max_depth:         {xgb_cfg['max_depth']}")
    logging.info(f"  n_estimators:      {xgb_cfg['n_estimators']}")
    logging.info(f"  learning_rate:     {xgb_cfg['learning_rate']}")
    logging.info(f"  subsample:         {xgb_cfg['subsample']}")
    logging.info(f"  colsample_bytree:  {xgb_cfg['colsample_bytree']}")
    logging.info(f"  objective:         binary:logistic")
    logging.info(f"  Tuning method:     None (fixed defaults)")

    # ---- Train ----
    model = train_model(train_df, xgb_cfg, seed)
    logging.info("Model training complete.")

    # ---- Training accuracy ----
    train_preds = model.predict(train_df[FEATURE_COLS].values)
    train_accuracy = float((train_preds == train_df["target"].values).mean())
    logging.info(f"Training accuracy:  {train_accuracy:.4f} "
                 f"({100 * train_accuracy:.1f}%)")

    # ---- Generate test predictions ----
    predictions = generate_predictions(model, test_df)

    # ---- Evaluate ----
    n_test = len(predictions)
    correct = int((predictions["predicted"] == predictions["target"]).sum())
    test_accuracy = correct / n_test

    n_up_actual = int((predictions["target"] == 1).sum())
    n_down_actual = int((predictions["target"] == 0).sum())
    majority_baseline = max(n_up_actual, n_down_actual) / n_test

    n_pred_up = int((predictions["predicted"] == 1).sum())
    n_pred_down = int((predictions["predicted"] == 0).sum())
    is_constant = (n_pred_up == 0) or (n_pred_down == 0)

    train_test_gap = train_accuracy - test_accuracy

    logging.info("-" * 50)
    logging.info("TEST EVALUATION")
    logging.info(f"  Test accuracy:         {test_accuracy:.4f} "
                 f"({100 * test_accuracy:.1f}%)")
    logging.info(f"  Majority baseline:     {majority_baseline:.4f} "
                 f"({100 * majority_baseline:.1f}%)")
    logging.info(f"  GeoBM baseline:        0.5749 (57.5%)")
    logging.info(f"  GA baseline:           0.5669 (56.7%)")

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
        logging.info(
            f"  WARNING: Model is a constant '{constant_class}' predictor."
        )
    else:
        logging.info("  Model produces non-constant predictions.")

    logging.info("-" * 50)
    logging.info("ACTUAL CLASS BALANCE (test set)")
    logging.info(f"  Actual up (1):         {n_up_actual:>5d} "
                 f"({100 * n_up_actual / n_test:.1f}%)")
    logging.info(f"  Actual down (0):       {n_down_actual:>5d} "
                 f"({100 * n_down_actual / n_test:.1f}%)")

    # ---- Overfitting check ----
    logging.info("-" * 50)
    logging.info("OVERFITTING CHECK")
    logging.info(f"  Training accuracy:     {train_accuracy:.4f} "
                 f"({100 * train_accuracy:.1f}%)")
    logging.info(f"  Test accuracy:         {test_accuracy:.4f} "
                 f"({100 * test_accuracy:.1f}%)")
    logging.info(f"  Train-test gap:        {train_test_gap:+.4f} "
                 f"({100 * train_test_gap:+.1f}pp)")

    if train_test_gap > 0.10:
        logging.info(
            "  WARNING: Train-test gap exceeds 10pp — significant overfitting."
        )
    elif train_test_gap > 0.05:
        logging.info(
            "  WARNING: Train-test gap exceeds 5pp — moderate overfitting."
        )
    elif train_test_gap > 0.02:
        logging.info("  NOTE: Moderate train-test gap.")
    else:
        logging.info("  Train-test gap is small.")

    # ---- Feature importance ----
    importance_df = get_feature_importance(model)

    logging.info("-" * 50)
    logging.info("FEATURE IMPORTANCE (gain-based, top 14)")
    for _, row in importance_df.iterrows():
        bar = "#" * int(row["importance"] * 50)
        logging.info(f"  {row['feature']:25s} {row['importance']:.4f}  {bar}")

    # ---- Probability distribution ----
    p_up_values = predictions["p_up"]
    logging.info("-" * 50)
    logging.info("PREDICTED PROBABILITY DISTRIBUTION (test set)")
    logging.info(f"  min:    {p_up_values.min():.4f}")
    logging.info(f"  25%:    {p_up_values.quantile(0.25):.4f}")
    logging.info(f"  median: {p_up_values.median():.4f}")
    logging.info(f"  75%:    {p_up_values.quantile(0.75):.4f}")
    logging.info(f"  max:    {p_up_values.max():.4f}")

    # ---- Save predictions ----
    out = Path(predictions_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    logging.info(f"Saved predictions: {out} ({len(predictions)} rows)")

    # ---- Final summary ----
    logging.info("=" * 50)
    logging.info("XGBoost CLASSIFIER COMPLETE")
    logging.info(f"  Train rows:            {len(train_df)}")
    logging.info(f"  Test rows:             {n_test}")
    logging.info(f"  Features:              {len(FEATURE_COLS)}")
    logging.info(f"  Training accuracy:     {100 * train_accuracy:.1f}%")
    logging.info(f"  Test accuracy:         {100 * test_accuracy:.1f}%")
    logging.info(f"  Majority baseline:     {100 * majority_baseline:.1f}%")
    logging.info(f"  GeoBM baseline:        57.5%")
    logging.info(f"  GA baseline:           56.7%")
    logging.info(f"  Train-test gap:        {100 * train_test_gap:+.1f}pp")
    logging.info(f"  Constant predictor:    {is_constant}")
    logging.info(f"  Top feature:           {importance_df.iloc[0]['feature']} "
                 f"({importance_df.iloc[0]['importance']:.4f})")
    logging.info(f"  Output:                {predictions_path}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
