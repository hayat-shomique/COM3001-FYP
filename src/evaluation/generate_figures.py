"""
COM3001 — Dissertation Figure Generation
==========================================
Reads prediction files and evaluation summary, generates publication-
quality figures for direct inclusion in the dissertation.

Figures produced:
  1. Three-paradigm accuracy comparison (bar chart with baseline)
  2. Confusion matrices (side-by-side for all 3 models)
  3. Balanced accuracy vs accuracy comparison
  4. XGBoost predicted probability distribution
  5. XGBoost feature importance (horizontal bar chart)

All figures saved to results/figures/ as PDF for LaTeX compatibility.

Usage:
    python src/evaluation/generate_figures.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import yaml
import xgboost as xgb

matplotlib.use("Agg")  # Non-interactive backend for server/CI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
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

FIGURE_DIR = Path("results/figures")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_style():
    """Set consistent dissertation-quality plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ---------------------------------------------------------------------------
# Figure 1: Accuracy Comparison Bar Chart
# ---------------------------------------------------------------------------

def fig_accuracy_comparison(predictions: dict, majority_baseline: float):
    """Bar chart comparing test accuracy across all paradigms + baseline."""
    models = list(predictions.keys())
    accuracies = []
    for name in models:
        df = predictions[name]
        acc = (df["predicted"] == df["target"]).mean()
        accuracies.append(acc * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colours = ["#4878CF", "#6ACC65", "#D65F5F"]
    bars = ax.bar(models, accuracies, color=colours, width=0.5,
                  edgecolor="black", linewidth=0.5)

    # Baseline line
    ax.axhline(y=majority_baseline * 100, color="black", linestyle="--",
               linewidth=1.2, label=f"Majority baseline ({majority_baseline*100:.1f}%)")

    # Value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold",
                fontsize=11)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Three-Paradigm Directional Accuracy Comparison")
    ax.set_ylim(45, 65)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Rename x-ticks for clarity
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["GeoBM\n(stochastic)", "GA\n(evolutionary)",
                        "XGBoost\n(supervised)"])

    path = FIGURE_DIR / "accuracy_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Confusion Matrices
# ---------------------------------------------------------------------------

def fig_confusion_matrices(predictions: dict):
    """Side-by-side confusion matrices for all 3 models."""
    models = list(predictions.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    titles = {
        "geobm": "GeoBM (57.5%)",
        "ga": "GA (56.7%)",
        "xgboost": "XGBoost (53.1%)",
    }

    for i, name in enumerate(models):
        df = predictions[name]
        target = df["target"].values
        predicted = df["predicted"].values

        tp = ((predicted == 1) & (target == 1)).sum()
        tn = ((predicted == 0) & (target == 0)).sum()
        fp = ((predicted == 1) & (target == 0)).sum()
        fn = ((predicted == 0) & (target == 1)).sum()

        cm = np.array([[tn, fp], [fn, tp]])

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Down", "Up"], yticklabels=["Down", "Up"],
                    cbar=False, linewidths=0.5, linecolor="gray",
                    annot_kws={"size": 14, "weight": "bold"})
        axes[i].set_title(titles.get(name, name), fontweight="bold")
        axes[i].set_ylabel("Actual" if i == 0 else "")
        axes[i].set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices — Held-Out Test Set (501 days)",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()

    path = FIGURE_DIR / "confusion_matrices.pdf"
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Balanced Accuracy vs Accuracy
# ---------------------------------------------------------------------------

def fig_balanced_vs_accuracy(predictions: dict, majority_baseline: float):
    """Scatter/bar comparison of accuracy vs balanced accuracy."""
    models = list(predictions.keys())
    accs = []
    bal_accs = []

    for name in models:
        df = predictions[name]
        t, p = df["target"].values, df["predicted"].values
        tp = ((p == 1) & (t == 1)).sum()
        tn = ((p == 0) & (t == 0)).sum()
        fp = ((p == 1) & (t == 0)).sum()
        fn = ((p == 0) & (t == 1)).sum()
        acc = (tp + tn) / len(t)
        rec_up = tp / (tp + fn) if (tp + fn) > 0 else 0
        rec_down = tn / (tn + fp) if (tn + fp) > 0 else 0
        bal = (rec_up + rec_down) / 2
        accs.append(acc * 100)
        bal_accs.append(bal * 100)

    x = np.arange(len(models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy",
                   color="#4878CF", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, bal_accs, width, label="Balanced Accuracy",
                   color="#D65F5F", edgecolor="black", linewidth=0.5)

    ax.axhline(y=50, color="gray", linestyle=":", linewidth=1,
               label="Random (50%)")
    ax.axhline(y=majority_baseline * 100, color="black", linestyle="--",
               linewidth=1, label=f"Majority baseline ({majority_baseline*100:.1f}%)")

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, bal_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Metric (%)")
    ax.set_title("Accuracy vs Balanced Accuracy — Class Imbalance Exposure")
    ax.set_xticks(x)
    ax.set_xticklabels(["GeoBM", "GA", "XGBoost"])
    ax.set_ylim(40, 65)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = FIGURE_DIR / "balanced_vs_accuracy.pdf"
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: XGBoost Predicted Probability Distribution
# ---------------------------------------------------------------------------

def fig_xgb_probability_distribution(xgb_preds: pd.DataFrame):
    """Histogram of XGBoost P(up) on test set, split by actual class."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    up_mask = xgb_preds["target"] == 1
    down_mask = xgb_preds["target"] == 0

    ax.hist(xgb_preds.loc[up_mask, "p_up"], bins=30, alpha=0.6,
            label=f"Actual Up (n={up_mask.sum()})", color="#4878CF",
            edgecolor="black", linewidth=0.3)
    ax.hist(xgb_preds.loc[down_mask, "p_up"], bins=30, alpha=0.6,
            label=f"Actual Down (n={down_mask.sum()})", color="#D65F5F",
            edgecolor="black", linewidth=0.3)

    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.2,
               label="Decision threshold (0.5)")

    ax.set_xlabel("Predicted P(Up)")
    ax.set_ylabel("Count")
    ax.set_title("XGBoost Predicted Probability Distribution by Actual Class")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = FIGURE_DIR / "xgb_probability_distribution.pdf"
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: XGBoost Feature Importance
# ---------------------------------------------------------------------------

def fig_xgb_feature_importance(config: dict):
    """Horizontal bar chart of XGBoost gain-based feature importance."""
    # Retrain model to get importances (deterministic with seed 42)
    seed = config["reproducibility"]["random_seed"]
    xgb_cfg = config["xgboost"]

    train_df = pd.read_csv(xgb_cfg["train_path"])
    model = xgb.XGBClassifier(
        max_depth=xgb_cfg["max_depth"],
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=seed, use_label_encoder=False, verbosity=0,
    )
    model.fit(train_df[FEATURE_COLS].values, train_df["target"].values)

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color="#4878CF", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Gain-Based Importance")
    ax.set_title("XGBoost Feature Importance (14 Engineered Features)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the near-uniform spread
    spread = importances.max() - importances.min()
    ax.text(0.95, 0.05,
            f"Spread: {spread:.4f}\n(near-uniform)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    path = FIGURE_DIR / "xgb_feature_importance.pdf"
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: Capacity–Accuracy Inversion
# ---------------------------------------------------------------------------

def fig_capacity_inversion():
    """Simple plot showing the capacity–accuracy inversion pattern."""
    models = ["GeoBM\n(2 params)", "GA\n(9 genes)", "XGBoost\n(100 trees)"]
    accuracies = [57.5, 56.7, 53.1]
    capacity = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(capacity, accuracies, "o-", color="#D65F5F", linewidth=2,
            markersize=10, markerfacecolor="white", markeredgewidth=2)

    ax.axhline(y=57.5, color="black", linestyle="--", linewidth=1,
               label="Majority baseline (57.5%)")

    for i, (cap, acc) in enumerate(zip(capacity, accuracies)):
        ax.annotate(f"{acc:.1f}%", (cap, acc), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontweight="bold", fontsize=11)

    ax.set_xticks(capacity)
    ax.set_xticklabels(models)
    ax.set_xlabel("Model Capacity")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Capacity–Accuracy Inversion")
    ax.set_ylim(49, 62)
    ax.legend(loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = FIGURE_DIR / "capacity_inversion.pdf"
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"  Saved: {path}")


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
    logging.info("COM3001 — Dissertation Figure Generation")
    logging.info("=" * 60)

    config = load_config()
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions
    eval_cfg = config["evaluation"]
    predictions = {}
    for path in eval_cfg["prediction_files"]:
        df = pd.read_csv(path, parse_dates=["date"])
        model_name = df["model"].iloc[0]
        predictions[model_name] = df

    n_up = (predictions["geobm"]["target"] == 1).sum()
    majority_baseline = n_up / len(predictions["geobm"])

    logging.info(f"Loaded {len(predictions)} model predictions.")
    logging.info(f"Generating figures to {FIGURE_DIR}/")

    fig_accuracy_comparison(predictions, majority_baseline)
    fig_confusion_matrices(predictions)
    fig_balanced_vs_accuracy(predictions, majority_baseline)
    fig_xgb_probability_distribution(predictions["xgboost"])
    fig_xgb_feature_importance(config)
    fig_capacity_inversion()

    logging.info("=" * 50)
    logging.info("ALL FIGURES GENERATED")
    logging.info(f"  Output directory: {FIGURE_DIR}")
    n_files = len(list(FIGURE_DIR.glob("*.pdf")))
    logging.info(f"  PDF files:        {n_files}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
