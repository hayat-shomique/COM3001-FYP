"""
COM3001 — Common Evaluation Harness
=====================================
Loads prediction files from all three paradigms (GeoBM, GA, XGBoost),
computes a standardised set of classification metrics for each, performs
pairwise statistical comparison, and produces a single summary table
suitable for direct inclusion in the dissertation.

EVALUATION DESIGN:
  All three models are evaluated on the identical 501-day held-out test
  set (2023-01-03 to 2024-12-30) under the same metrics. The majority-
  class baseline (57.5%) serves as the zero-skill reference. Every metric
  is computed from the same (target, predicted) pairs stored in the
  prediction CSVs — the harness introduces no new modelling decisions.

METRICS:
  - Accuracy (overall classification rate)
  - Balanced accuracy (mean of per-class recall)
  - Precision, recall, F1 for each class
  - Matthews Correlation Coefficient (MCC)
  - Confusion matrix counts (TP, TN, FP, FN)
  - Prediction class balance
  - McNemar's test for pairwise accuracy differences
  - Prediction diversity (agreement rates, unique p_up values)

REFERENCES:
  Sokolova & Lapalme (2009) for classification metric selection.
  Dietterich (1998, Neural Computation) for McNemar's test methodology.

This module does NOT:
  - Retrain or modify any model
  - Use any data beyond the prediction CSVs
  - Compute financial metrics (PnL, Sharpe, drawdown)
  - Produce visualisations (deferred to a separate plotting module)

Usage:
    python src/evaluation/evaluate.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute classification metrics from a prediction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'target' and 'predicted' columns (0/1 integers).

    Returns
    -------
    dict of metric name -> value
    """
    target = df["target"].values
    predicted = df["predicted"].values
    n = len(target)

    # Confusion matrix components
    tp = int(((predicted == 1) & (target == 1)).sum())
    tn = int(((predicted == 0) & (target == 0)).sum())
    fp = int(((predicted == 1) & (target == 0)).sum())
    fn = int(((predicted == 0) & (target == 1)).sum())

    # Accuracy
    accuracy = (tp + tn) / n

    # Per-class recall
    recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced accuracy
    balanced_accuracy = (recall_up + recall_down) / 2

    # Precision
    precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # F1
    f1_up = (2 * precision_up * recall_up / (precision_up + recall_up)
             if (precision_up + recall_up) > 0 else 0.0)
    f1_down = (2 * precision_down * recall_down / (precision_down + recall_down)
               if (precision_down + recall_down) > 0 else 0.0)

    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

    # Prediction class balance
    n_pred_up = int((predicted == 1).sum())
    n_pred_down = int((predicted == 0).sum())

    return {
        "n": n,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "f1_up": f1_up,
        "precision_down": precision_down,
        "recall_down": recall_down,
        "f1_down": f1_down,
        "pred_up": n_pred_up,
        "pred_down": n_pred_down,
        "pred_up_pct": 100 * n_pred_up / n,
        "mcc": mcc,
    }


def mcnemar_test(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """
    McNemar's test for paired binary predictions.

    Tests whether two classifiers disagree in a systematic direction.
    Under H0, the number of observations where A is correct and B is
    wrong should equal the number where B is correct and A is wrong.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Prediction DataFrames with 'target' and 'predicted' columns,
        aligned by row.

    Returns
    -------
    dict with b, c (off-diagonal counts), chi2 statistic, p-value
    """
    correct_a = (df_a["predicted"].values == df_a["target"].values)
    correct_b = (df_b["predicted"].values == df_b["target"].values)

    # b: A correct, B wrong
    b = int((correct_a & ~correct_b).sum())
    # c: A wrong, B correct
    c = int((~correct_a & correct_b).sum())

    # McNemar's chi-squared (with continuity correction)
    if (b + c) == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "chi2": chi2_stat,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_model_metrics(model_name: str, metrics: dict) -> None:
    """Log metrics for a single model."""
    logging.info(f"  {'Model:':<22s} {model_name}")
    logging.info(f"  {'Accuracy:':<22s} {metrics['accuracy']:.4f} "
                 f"({100 * metrics['accuracy']:.1f}%)")
    logging.info(f"  {'Balanced accuracy:':<22s} {metrics['balanced_accuracy']:.4f} "
                 f"({100 * metrics['balanced_accuracy']:.1f}%)")
    logging.info(f"  {'Precision (up):':<22s} {metrics['precision_up']:.4f}")
    logging.info(f"  {'Recall (up):':<22s} {metrics['recall_up']:.4f}")
    logging.info(f"  {'F1 (up):':<22s} {metrics['f1_up']:.4f}")
    logging.info(f"  {'Precision (down):':<22s} {metrics['precision_down']:.4f}")
    logging.info(f"  {'Recall (down):':<22s} {metrics['recall_down']:.4f}")
    logging.info(f"  {'F1 (down):':<22s} {metrics['f1_down']:.4f}")
    logging.info(f"  {'MCC:':<22s} {metrics['mcc']:.4f}")
    logging.info(f"  {'Confusion:':<22s} TP={metrics['tp']} TN={metrics['tn']} "
                 f"FP={metrics['fp']} FN={metrics['fn']}")
    logging.info(f"  {'Pred balance:':<22s} up={metrics['pred_up']} "
                 f"({metrics['pred_up_pct']:.1f}%) "
                 f"down={metrics['pred_down']}")


def log_comparison_table(all_metrics: dict, majority_baseline: float) -> None:
    """Log a compact comparison table across all models."""
    logging.info("-" * 80)
    logging.info("THREE-PARADIGM COMPARISON TABLE")
    logging.info(f"{'Model':<12s} {'Acc':>7s} {'vs Base':>8s} {'BalAcc':>7s} "
                 f"{'MCC':>7s} "
                 f"{'Prec↑':>7s} {'Rec↑':>7s} {'F1↑':>7s} "
                 f"{'Prec↓':>7s} {'Rec↓':>7s} {'F1↓':>7s} "
                 f"{'Pred↑%':>7s}")
    logging.info("-" * 88)

    for model_name, m in all_metrics.items():
        delta = m["accuracy"] - majority_baseline
        logging.info(
            f"{model_name:<12s} "
            f"{100 * m['accuracy']:>6.1f}% "
            f"{100 * delta:>+7.1f}pp "
            f"{100 * m['balanced_accuracy']:>6.1f}% "
            f"{m['mcc']:>7.4f} "
            f"{m['precision_up']:>7.4f} "
            f"{m['recall_up']:>7.4f} "
            f"{m['f1_up']:>7.4f} "
            f"{m['precision_down']:>7.4f} "
            f"{m['recall_down']:>7.4f} "
            f"{m['f1_down']:>7.4f} "
            f"{m['pred_up_pct']:>6.1f}%"
        )

    logging.info("-" * 88)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run common evaluation harness."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — Common Evaluation Harness")
    logging.info("=" * 60)

    config = load_config()
    eval_cfg = config["evaluation"]
    prediction_files = eval_cfg["prediction_files"]
    summary_path = eval_cfg["summary_path"]

    # ---- Load all prediction files ----
    predictions = {}
    for path in prediction_files:
        df = pd.read_csv(path, parse_dates=["date"])
        model_name = df["model"].iloc[0]
        predictions[model_name] = df
        logging.info(f"Loaded {model_name}: {path} ({len(df)} rows)")

    # ---- Verify alignment ----
    model_names = list(predictions.keys())
    ref = predictions[model_names[0]]

    for name in model_names[1:]:
        df = predictions[name]
        if not (df["date"].values == ref["date"].values).all():
            raise ValueError(f"Date mismatch: {model_names[0]} vs {name}")
        if not (df["target"].values == ref["target"].values).all():
            raise ValueError(f"Target mismatch: {model_names[0]} vs {name}")

    logging.info(f"Alignment verified: {len(ref)} rows, "
                 f"{len(model_names)} models, dates and targets match.")

    # ---- Compute majority-class baseline ----
    n_test = len(ref)
    n_up = int((ref["target"] == 1).sum())
    n_down = int((ref["target"] == 0).sum())
    majority_baseline = max(n_up, n_down) / n_test

    logging.info("-" * 50)
    logging.info("TEST SET SUMMARY")
    logging.info(f"  Test rows:           {n_test}")
    logging.info(f"  Actual up (1):       {n_up} ({100 * n_up / n_test:.1f}%)")
    logging.info(f"  Actual down (0):     {n_down} ({100 * n_down / n_test:.1f}%)")
    logging.info(f"  Majority baseline:   {majority_baseline:.4f} "
                 f"({100 * majority_baseline:.1f}%)")

    # ---- Compute metrics for each model ----
    all_metrics = {}
    for name in model_names:
        metrics = compute_metrics(predictions[name])
        all_metrics[name] = metrics

    # ---- Log per-model detail ----
    for name in model_names:
        logging.info("-" * 50)
        log_model_metrics(name, all_metrics[name])

    # ---- Compact comparison table ----
    log_comparison_table(all_metrics, majority_baseline)

    # ---- McNemar's pairwise tests ----
    logging.info("-" * 50)
    logging.info("McNEMAR'S PAIRWISE TESTS (continuity-corrected)")
    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            result = mcnemar_test(predictions[a], predictions[b])
            sig = "significant" if result["p_value"] < 0.05 else "not significant"
            logging.info(
                f"  {a} vs {b}: "
                f"b={result['b_a_correct_b_wrong']}, "
                f"c={result['c_a_wrong_b_correct']}, "
                f"chi2={result['chi2']:.3f}, "
                f"p={result['p_value']:.4f} ({sig} at alpha=0.05)"
            )
            pairs.append({
                "model_a": a, "model_b": b,
                "b": result["b_a_correct_b_wrong"],
                "c": result["c_a_wrong_b_correct"],
                "chi2": result["chi2"],
                "p_value": result["p_value"],
                "significant": result["p_value"] < 0.05,
            })

    # ---- Key findings ----
    logging.info("-" * 50)
    logging.info("KEY FINDINGS")

    best_model = max(all_metrics, key=lambda m: all_metrics[m]["accuracy"])
    best_acc = all_metrics[best_model]["accuracy"]
    logging.info(f"  Best test accuracy:  {best_model} "
                 f"({100 * best_acc:.1f}%)")

    if best_acc > majority_baseline:
        logging.info(f"  Baseline exceeded:   Yes, by "
                     f"{100 * (best_acc - majority_baseline):.1f}pp")
    else:
        logging.info(f"  Baseline exceeded:   No — no model beat "
                     f"{100 * majority_baseline:.1f}%")

    # Check which models produce non-trivial predictions
    for name, m in all_metrics.items():
        if m["pred_down"] == 0:
            logging.info(f"  {name}: constant 'up' predictor")
        elif m["pred_up"] == 0:
            logging.info(f"  {name}: constant 'down' predictor")

    # Balanced accuracy insight
    logging.info("-" * 50)
    logging.info("BALANCED ACCURACY INSIGHT")
    for name, m in all_metrics.items():
        logging.info(
            f"  {name:<12s}: accuracy={100 * m['accuracy']:.1f}%, "
            f"balanced={100 * m['balanced_accuracy']:.1f}%, "
            f"gap={100 * (m['accuracy'] - m['balanced_accuracy']):+.1f}pp"
        )
    logging.info("  (Positive gap = model benefits from class imbalance; "
                 "negative gap = model is hurt by it)")

    # ---- Prediction diversity ----
    logging.info("-" * 50)
    logging.info("PREDICTION DIVERSITY")

    # Pairwise agreement
    preds = {name: predictions[name]["predicted"].values for name in model_names}
    targets = ref["target"].values

    # All-agree check (works for any number of models)
    all_pred_arrays = np.column_stack([preds[n] for n in model_names])
    all_agree_mask = np.all(all_pred_arrays == all_pred_arrays[:, :1], axis=1)
    all_agree = int(all_agree_mask.sum())
    all_agree_correct = int((all_agree_mask & (preds[model_names[0]] == targets)).sum())
    logging.info(f"  All {len(model_names)} agree:      {all_agree}/{n_test} "
                 f"({100 * all_agree / n_test:.1f}%)")
    logging.info(f"    ...and correct:    {all_agree_correct}")
    logging.info(f"    ...and wrong:      {all_agree - all_agree_correct}")

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            na, nb = model_names[i], model_names[j]
            agree = int((preds[na] == preds[nb]).sum())
            logging.info(f"  {na} == {nb}: {agree}/{n_test} "
                         f"({100 * agree / n_test:.1f}%)")

    # Unique p_up values
    logging.info("  Unique p_up values:")
    for name in model_names:
        n_unique = predictions[name]["p_up"].nunique()
        logging.info(f"    {name:<12s}: {n_unique}")

    # When XGBoost disagrees with GeoBM, how does it do?
    if "xgboost" in preds and "geobm" in preds:
        disagree_mask = preds["xgboost"] != preds["geobm"]
        n_disagree = int(disagree_mask.sum())
        if n_disagree > 0:
            xgb_correct_on_disagree = int(
                (preds["xgboost"][disagree_mask] == targets[disagree_mask]).sum()
            )
            logging.info(f"  XGBoost disagrees with GeoBM on {n_disagree} days:")
            logging.info(f"    XGBoost correct:   {xgb_correct_on_disagree}/{n_disagree} "
                         f"({100 * xgb_correct_on_disagree / n_disagree:.1f}%)")
            logging.info(f"    GeoBM correct:     {n_disagree - xgb_correct_on_disagree}/"
                         f"{n_disagree} "
                         f"({100 * (n_disagree - xgb_correct_on_disagree) / n_disagree:.1f}%)")

    # ---- Save summary CSV ----
    summary_rows = []
    for name, m in all_metrics.items():
        row = {"model": name}
        row.update(m)
        row["vs_baseline"] = m["accuracy"] - majority_baseline
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out = Path(summary_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out, index=False)
    logging.info(f"Saved summary: {out}")

    # ---- Final summary ----
    logging.info("=" * 50)
    logging.info("EVALUATION COMPLETE")
    logging.info(f"  Models evaluated:    {len(model_names)}")
    logging.info(f"  Test rows:           {n_test}")
    logging.info(f"  Majority baseline:   {100 * majority_baseline:.1f}%")
    logging.info(f"  Best model:          {best_model} "
                 f"({100 * best_acc:.1f}%)")
    logging.info(f"  Baseline exceeded:   "
                 f"{'Yes' if best_acc > majority_baseline else 'No'}")
    logging.info(f"  Summary saved:       {summary_path}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
