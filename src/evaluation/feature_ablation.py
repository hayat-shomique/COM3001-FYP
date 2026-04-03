"""
COM3001 — Feature Ablation Study
==================================
Removes one feature category at a time and retrains XGBoost-v2 to
measure each category's marginal contribution to directional prediction.

EXPERIMENT DESIGN:
  7 experiments: full model (14 features) + 6 single-category removals.
  All experiments use the same XGBoost-v2 best configuration (max_depth=2,
  n_estimators=17, learning_rate=0.1, subsample=0.8, colsample=0.8).
  No hyperparameter re-tuning — the ablation tests feature contribution,
  not hyperparameter sensitivity.

FEATURE CATEGORIES (6 categories, 14 features):
  lagged_returns:       ret_lag_1, ret_lag_2, ret_lag_5
  realised_volatility:  vol_5, vol_10, vol_20
  momentum:             mom_5, mom_10
  ma_ratios:       close_sma5_ratio, close_sma20_ratio, sma5_sma20_ratio
  rsi:             rsi_14
  volume:          volume_chg_1, volume_sma20_ratio

This module does NOT:
  - Re-run grid search or re-tune hyperparameters per ablation
  - Remove individual features (category-level ablation is chosen)
  - Modify any model module or prediction file
  - Use test data in training

Usage:
    python src/evaluation/feature_ablation.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


ALL_FEATURES = [
    "ret_lag_1", "ret_lag_2", "ret_lag_5",
    "vol_5", "vol_10", "vol_20",
    "mom_5", "mom_10",
    "close_sma5_ratio", "close_sma20_ratio", "sma5_sma20_ratio",
    "rsi_14",
    "volume_chg_1", "volume_sma20_ratio",
]

CATEGORIES = {
    "lagged_returns":      ["ret_lag_1", "ret_lag_2", "ret_lag_5"],
    "realised_volatility": ["vol_5", "vol_10", "vol_20"],
    "momentum":            ["mom_5", "mom_10"],
    "ma_ratios":           ["close_sma5_ratio", "close_sma20_ratio", "sma5_sma20_ratio"],
    "rsi":                 ["rsi_14"],
    "volume":              ["volume_chg_1", "volume_sma20_ratio"],
}

# Guard: assert categories cover exactly ALL_FEATURES — fail loudly on drift
_cat_features = sorted(f for fs in CATEGORIES.values() for f in fs)
assert _cat_features == sorted(ALL_FEATURES), (
    f"FATAL: CATEGORIES and ALL_FEATURES have diverged.\n"
    f"  In CATEGORIES but not ALL_FEATURES: {set(_cat_features) - set(ALL_FEATURES)}\n"
    f"  In ALL_FEATURES but not CATEGORIES: {set(ALL_FEATURES) - set(_cat_features)}"
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute accuracy, balanced accuracy, MCC, and class balance."""
    n = len(target)
    tp = int(((predicted == 1) & (target == 1)).sum())
    tn = int(((predicted == 0) & (target == 0)).sum())
    fp = int(((predicted == 1) & (target == 0)).sum())
    fn = int(((predicted == 0) & (target == 1)).sum())

    accuracy = (tp + tn) / n

    rec_up = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rec_down = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (rec_up + rec_down) / 2

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

    n_pred_up = int((predicted == 1).sum())

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "pred_up_pct": 100 * n_pred_up / n,
    }


# ---------------------------------------------------------------------------
# Single Ablation Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    seed: int,
    v2_best: dict | None = None,
) -> dict:
    """Train XGBoost with v2 best config on given features and evaluate on test."""
    # Default to v2 best config; overridable for testing
    cfg = v2_best or {"max_depth": 2, "n_estimators": 17, "learning_rate": 0.1}
    model = xgb.XGBClassifier(
        max_depth=cfg["max_depth"],
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        use_label_encoder=False,
        verbosity=0,
    )

    X_train = train_df[features].values
    y_train = train_df["target"].values
    X_test = test_df[features].values
    y_test = test_df["target"].values

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return compute_metrics(y_test, preds)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_ablation_figure(results: list[dict], majority_baseline: float):
    """Horizontal bar chart of ablation test accuracies."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 13, "figure.facecolor": "white",
        "savefig.bbox": "tight",
    })

    labels = [r["experiment"] for r in results]
    accs = [100 * r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ["#4878CF" if i == 0 else "#6ACC65" for i in range(len(labels))]
    bars = ax.barh(labels, accs, color=colours, edgecolor="black", linewidth=0.4)

    ax.axvline(x=majority_baseline * 100, color="black", linestyle="--",
               linewidth=1.2, label=f"Majority baseline ({majority_baseline*100:.1f}%)")
    ax.axvline(x=accs[0], color="#4878CF", linestyle=":", linewidth=1,
               label=f"Full model ({accs[0]:.1f}%)")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", ha="left", va="center", fontsize=10)

    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("Feature Category Ablation — XGBoost-v2")
    ax.set_xlim(45, 62)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    path = Path("results/figures/feature_ablation.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    logging.info("COM3001 — Feature Ablation Study")
    logging.info("=" * 60)

    config = load_config()
    v2_cfg = config["xgboost_v2"]
    seed = config["reproducibility"]["random_seed"]

    train_df = pd.read_csv(v2_cfg["train_path"], parse_dates=["date"])
    test_df = pd.read_csv(v2_cfg["test_path"], parse_dates=["date"])

    logging.info(f"Loaded train: {len(train_df)} rows")
    logging.info(f"Loaded test:  {len(test_df)} rows")
    logging.info(f"Seed:         {seed}")

    # V2 best config — from the grid search in xgb_classifier_v2.py
    v2_best = {"max_depth": 2, "n_estimators": 17, "learning_rate": 0.1}
    logging.info(f"XGBoost config: depth={v2_best['max_depth']}, "
                 f"trees={v2_best['n_estimators']}, "
                 f"lr={v2_best['learning_rate']} (v2 best)")

    n_up = int((test_df["target"] == 1).sum())
    majority_baseline = n_up / len(test_df)

    # ---- Run experiments ----
    results = []

    # Full model
    logging.info("-" * 50)
    logging.info("Running 7 ablation experiments...")

    full_metrics = run_experiment(train_df, test_df, ALL_FEATURES, seed, v2_best)
    full_metrics["experiment"] = "Full (14 features)"
    full_metrics["n_features"] = 14
    full_metrics["removed"] = "—"
    results.append(full_metrics)

    full_acc = full_metrics["accuracy"]

    # Category removals
    for cat_name, cat_features in CATEGORIES.items():
        remaining = [f for f in ALL_FEATURES if f not in cat_features]
        metrics = run_experiment(train_df, test_df, remaining, seed, v2_best)
        metrics["experiment"] = f"−{cat_name}"
        metrics["n_features"] = len(remaining)
        metrics["removed"] = cat_name
        metrics["delta_vs_full"] = metrics["accuracy"] - full_acc
        results.append(metrics)

    # ---- Log results table ----
    logging.info("-" * 80)
    logging.info("ABLATION RESULTS TABLE")
    logging.info(
        f"  {'Experiment':<22s} {'Feat':>4s} {'Acc':>7s} {'vs Full':>8s} "
        f"{'BalAcc':>7s} {'MCC':>7s} {'Pred↑%':>7s}"
    )
    logging.info(f"  {'-'*22} {'-'*4} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for r in results:
        delta = r.get("delta_vs_full", 0)
        delta_str = "—" if r["removed"] == "—" else f"{100*delta:+.1f}pp"
        logging.info(
            f"  {r['experiment']:<22s} {r['n_features']:>4d} "
            f"{100*r['accuracy']:>6.1f}% {delta_str:>8s} "
            f"{100*r['balanced_accuracy']:>6.1f}% "
            f"{r['mcc']:>7.4f} "
            f"{r['pred_up_pct']:>6.1f}%"
        )

    # ---- Interpretation ----
    logging.info("-" * 50)
    logging.info("INTERPRETATION")

    ablations = [r for r in results if r["removed"] != "—"]

    biggest_drop = min(ablations, key=lambda r: r["delta_vs_full"])
    logging.info(
        f"  Biggest accuracy DROP:    −{biggest_drop['removed']} "
        f"({100*biggest_drop['delta_vs_full']:+.1f}pp)"
    )
    logging.info(
        f"    → This category contributes most to the model's predictions."
    )

    improvements = [r for r in ablations if r["delta_vs_full"] > 0]
    if improvements:
        for imp in improvements:
            logging.info(
                f"  Accuracy INCREASED by removing: {imp['removed']} "
                f"({100*imp['delta_vs_full']:+.1f}pp)"
            )
            logging.info(
                f"    → This category adds noise — its features hurt predictions."
            )
    else:
        logging.info("  No category removal increased accuracy.")

    above_baseline = [r for r in ablations
                      if r["accuracy"] > majority_baseline]
    if above_baseline:
        for ab in above_baseline:
            logging.info(
                f"  ABOVE BASELINE after removing {ab['removed']}: "
                f"{100*ab['accuracy']:.1f}% > {100*majority_baseline:.1f}%"
            )
    else:
        logging.info(
            f"  No single-category removal pushed accuracy above the "
            f"{100*majority_baseline:.1f}% baseline."
        )

    # GA comparison
    logging.info("-" * 50)
    logging.info("CROSS-PARADIGM COMPARISON WITH GA")
    logging.info(
        "  The GA converged to ma_ratios (close_sma20_ratio, close_sma5_ratio)."
    )
    ma_result = next(r for r in ablations if r["removed"] == "ma_ratios")
    logging.info(
        f"  Removing ma_ratios: {100*ma_result['delta_vs_full']:+.1f}pp "
        f"(accuracy {100*ma_result['accuracy']:.1f}%)"
    )
    if ma_result["delta_vs_full"] < 0:
        logging.info(
            "  The ablation CONFIRMS the GA's implicit feature selection: "
            "removing ma_ratios hurts XGBoost accuracy."
        )
    elif ma_result["delta_vs_full"] > 0:
        logging.info(
            "  The ablation CONTRADICTS the GA: removing ma_ratios "
            "improves XGBoost accuracy — the GA's selected features add noise."
        )
    else:
        logging.info(
            "  Removing ma_ratios has no effect — the GA's selection is neutral."
        )

    # ---- Save CSV ----
    csv_rows = []
    for r in results:
        csv_rows.append({
            "experiment": r["experiment"],
            "n_features": r["n_features"],
            "removed_category": r["removed"],
            "accuracy": r["accuracy"],
            "delta_vs_full": r.get("delta_vs_full", 0.0),
            "balanced_accuracy": r["balanced_accuracy"],
            "mcc": r["mcc"],
            "pred_up_pct": r["pred_up_pct"],
        })

    csv_path = Path("results/tables/feature_ablation.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logging.info(f"Saved table: {csv_path}")

    # ---- Generate figure ----
    generate_ablation_figure(results, majority_baseline)

    # ---- Summary ----
    logging.info("=" * 50)
    logging.info("ABLATION STUDY COMPLETE")
    logging.info(f"  Experiments:         {len(results)}")
    logging.info(f"  Full model accuracy: {100*full_acc:.1f}%")
    logging.info(f"  Majority baseline:   {100*majority_baseline:.1f}%")
    logging.info(f"  Table saved:         {csv_path}")
    logging.info(f"  Figure saved:        results/figures/feature_ablation.pdf")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
