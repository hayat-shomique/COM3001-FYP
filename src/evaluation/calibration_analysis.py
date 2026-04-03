"""
COM3001 — Calibration and Threshold Analysis
===============================================
Assesses the quality of XGBoost's predicted probabilities and tests
whether any decision threshold produces useful directional predictions.

THREE ANALYSES:
  1. Brier score — are probabilities better than always predicting
     the base rate?
  2. Calibration curve — does predicted confidence match actual accuracy?
  3. Threshold sweep — does any operating point produce MCC > 0?

Applies to both XGBoost-v1 (fixed defaults) and XGBoost-v2 (validated,
early-stopped) for comparison.

This module does NOT:
  - Retrain any model
  - Modify any prediction file
  - Compute financial metrics

Usage:
    python src/evaluation/calibration_analysis.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_score(target: np.ndarray, p_up: np.ndarray) -> float:
    """Brier score = mean((p - y)^2). Lower is better."""
    return float(np.mean((p_up - target) ** 2))


def compute_mcc(target: np.ndarray, predicted: np.ndarray) -> float:
    tp = int(((predicted == 1) & (target == 1)).sum())
    tn = int(((predicted == 0) & (target == 0)).sum())
    fp = int(((predicted == 1) & (target == 0)).sum())
    fn = int(((predicted == 0) & (target == 1)).sum())
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return num / den if den > 0 else 0.0


def threshold_metrics(
    target: np.ndarray, p_up: np.ndarray, threshold: float
) -> dict:
    """Compute classification metrics at a given probability threshold."""
    predicted = (p_up >= threshold).astype(int)
    n = len(target)
    tp = int(((predicted == 1) & (target == 1)).sum())
    tn = int(((predicted == 0) & (target == 0)).sum())
    fp = int(((predicted == 1) & (target == 0)).sum())
    fn = int(((predicted == 0) & (target == 1)).sum())

    accuracy = (tp + tn) / n
    rec_up = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rec_down = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (rec_up + rec_down) / 2
    mcc = compute_mcc(target, predicted)
    pred_up_pct = 100 * int((predicted == 1).sum()) / n

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "pred_up_pct": pred_up_pct,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 13, "figure.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    })


def fig_calibration_curve(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
):
    """Reliability diagram for v1 and v2."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    colours = {"XGBoost-v1": "#D65F5F", "XGBoost-v2": "#4878CF"}

    for label, (target, p_up) in models.items():
        try:
            fraction_pos, mean_predicted = calibration_curve(
                target, p_up, n_bins=10, strategy="uniform"
            )
            ax.plot(mean_predicted, fraction_pos, "o-",
                    color=colours.get(label, "gray"), label=label,
                    linewidth=1.5, markersize=5)
        except ValueError:
            logging.warning(f"Could not compute calibration curve for {label}")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual)")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = Path("results/figures/calibration_curve.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Saved figure: {path}")


def fig_threshold_sensitivity(
    sweeps: dict[str, list[dict]],
):
    """MCC vs threshold for v1 and v2."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colours = {"XGBoost-v1": "#D65F5F", "XGBoost-v2": "#4878CF"}

    for label, sweep in sweeps.items():
        thresholds = [s["threshold"] for s in sweep]
        mccs = [s["mcc"] for s in sweep]
        ax.plot(thresholds, mccs, "o-", color=colours.get(label, "gray"),
                label=label, linewidth=1.5, markersize=5)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1,
               label="MCC = 0 (random)")
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=0.8,
               label="Default threshold (0.5)")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Matthews Correlation Coefficient")
    ax.set_title("Threshold Sensitivity — MCC vs Decision Threshold")
    ax.legend(loc="best", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = Path("results/figures/threshold_sensitivity.pdf")
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Saved figure: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — Calibration and Threshold Analysis")
    logging.info("=" * 60)

    config = load_config()
    seed = config["reproducibility"]["random_seed"]
    setup_style()

    # ---- Load predictions ----
    v1 = pd.read_csv("results/predictions/xgb_predictions.csv")
    v2 = pd.read_csv("results/predictions/xgb_v2_predictions.csv")

    target = v1["target"].values
    p_up_v1 = v1["p_up"].values
    p_up_v2 = v2["p_up"].values
    n_test = len(target)
    base_rate = target.mean()

    logging.info(f"Loaded v1: {len(v1)} rows, p_up range [{p_up_v1.min():.4f}, {p_up_v1.max():.4f}]")
    logging.info(f"Loaded v2: {len(v2)} rows, p_up range [{p_up_v2.min():.4f}, {p_up_v2.max():.4f}]")
    logging.info(f"Test base rate: {base_rate:.4f} ({100*base_rate:.1f}% up)")

    # ==================================================================
    # Part 1 — Brier Scores
    # ==================================================================
    brier_v1 = brier_score(target, p_up_v1)
    brier_v2 = brier_score(target, p_up_v2)
    brier_naive = base_rate * (1 - base_rate)

    logging.info("-" * 50)
    logging.info("BRIER SCORES")
    logging.info(f"  XGBoost-v1:    {brier_v1:.4f}")
    logging.info(f"  XGBoost-v2:    {brier_v2:.4f}")
    logging.info(f"  Naive ({base_rate:.3f}):  {brier_naive:.4f}")

    for name, bs in [("v1", brier_v1), ("v2", brier_v2)]:
        if bs > brier_naive:
            logging.info(f"  {name} vs naive: WORSE (probabilities less informative "
                         f"than always predicting {base_rate:.3f})")
        elif bs < brier_naive:
            logging.info(f"  {name} vs naive: BETTER (probabilities contain "
                         f"some information)")
        else:
            logging.info(f"  {name} vs naive: EQUAL")

    # ==================================================================
    # Part 2 — Calibration Curves
    # ==================================================================
    logging.info("-" * 50)
    logging.info("CALIBRATION CURVES")
    fig_calibration_curve({
        "XGBoost-v1": (target, p_up_v1),
        "XGBoost-v2": (target, p_up_v2),
    })

    # ==================================================================
    # Part 3 — Threshold Sweep
    # ==================================================================
    thresholds = np.arange(0.30, 0.71, 0.05)
    sweeps = {}

    for name, p_up in [("XGBoost-v1", p_up_v1), ("XGBoost-v2", p_up_v2)]:
        sweep = [threshold_metrics(target, p_up, t) for t in thresholds]
        sweeps[name] = sweep

        logging.info("-" * 60)
        logging.info(f"THRESHOLD SWEEP — {name}")
        logging.info(f"  {'Threshold':>9s} {'Accuracy':>9s} {'BalAcc':>8s} "
                     f"{'MCC':>7s} {'Pred↑%':>7s}")
        logging.info(f"  {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*7}")

        for s in sweep:
            logging.info(
                f"  {s['threshold']:>9.2f} "
                f"{100*s['accuracy']:>8.1f}% "
                f"{100*s['balanced_accuracy']:>7.1f}% "
                f"{s['mcc']:>7.4f} "
                f"{s['pred_up_pct']:>6.1f}%"
            )

    # ---- Threshold figure ----
    fig_threshold_sensitivity(sweeps)

    # ---- Save sweep CSV ----
    csv_rows = []
    for name, sweep in sweeps.items():
        for s in sweep:
            row = {"model": name}
            row.update(s)
            csv_rows.append(row)

    csv_path = Path("results/tables/threshold_sweep.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logging.info(f"Saved table: {csv_path}")

    # ==================================================================
    # Interpretation
    # ==================================================================
    logging.info("-" * 50)
    logging.info("INTERPRETATION")

    for name, sweep in sweeps.items():
        best_mcc_row = max(sweep, key=lambda s: s["mcc"])
        logging.info(
            f"  Best MCC for {name}: {best_mcc_row['mcc']:.4f} "
            f"at threshold={best_mcc_row['threshold']:.2f} "
            f"(accuracy={100*best_mcc_row['accuracy']:.1f}%)"
        )

    any_positive = any(
        s["mcc"] > 0
        for sweep in sweeps.values()
        for s in sweep
    )
    if any_positive:
        logging.info("  Any threshold produces MCC > 0: YES")
        for name, sweep in sweeps.items():
            pos = [s for s in sweep if s["mcc"] > 0]
            if pos:
                best = max(pos, key=lambda s: s["mcc"])
                logging.info(
                    f"    {name}: best positive MCC = {best['mcc']:.4f} "
                    f"at threshold {best['threshold']:.2f}"
                )
    else:
        logging.info("  Any threshold produces MCC > 0: NO")
        logging.info("  The model has no useful signal at any operating point.")

    # ---- Summary ----
    logging.info("=" * 50)
    logging.info("CALIBRATION ANALYSIS COMPLETE")
    logging.info(f"  Brier v1: {brier_v1:.4f} ({'worse' if brier_v1 > brier_naive else 'better'} than naive)")
    logging.info(f"  Brier v2: {brier_v2:.4f} ({'worse' if brier_v2 > brier_naive else 'better'} than naive)")
    logging.info(f"  Figures:  calibration_curve.pdf, threshold_sensitivity.pdf")
    logging.info(f"  Table:    threshold_sweep.csv")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
