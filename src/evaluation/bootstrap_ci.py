"""
COM3001 — Bootstrap Confidence Intervals
==========================================
Computes 95% bootstrap confidence intervals for accuracy and MCC
across all models using 10,000 resamples of the 501-day test set.

METHODOLOGY:
  Percentile bootstrap (Efron & Tibshirani, 1993). For each of 10,000
  resamples, draw 501 indices with replacement, compute accuracy and
  MCC on the resampled (target, predicted) pairs, and take the 2.5th
  and 97.5th percentiles as the 95% CI bounds.

  GeoBM and majority-baseline MCC are structurally zero — constant
  predictors produce MCC = 0 on every resample because at least one
  confusion matrix cell is always zero. These are noted as structural
  properties, not computed as empirical findings.

This module does NOT:
  - Retrain any model
  - Modify any prediction file
  - Use parametric distributional assumptions

Usage:
    python src/evaluation/bootstrap_ci.py

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

def compute_mcc(target: np.ndarray, predicted: np.ndarray) -> float:
    tp = int(((predicted == 1) & (target == 1)).sum())
    tn = int(((predicted == 0) & (target == 0)).sum())
    fp = int(((predicted == 1) & (target == 0)).sum())
    fn = int(((predicted == 0) & (target == 1)).sum())
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    target: np.ndarray,
    predicted: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    compute_mcc_ci: bool = True,
) -> dict:
    """
    Compute 95% percentile bootstrap CIs for accuracy and MCC.

    Parameters
    ----------
    target, predicted : arrays of 0/1
    n_resamples : number of bootstrap resamples
    seed : random seed for reproducibility
    compute_mcc_ci : if False, skip MCC (for constant predictors where
                     MCC is structurally zero)

    Returns
    -------
    dict with point estimates, CI bounds, and CI widths
    """
    rng = np.random.default_rng(seed)
    n = len(target)

    acc_point = float((predicted == target).mean())
    mcc_point = compute_mcc(target, predicted)

    acc_boots = np.empty(n_resamples)
    mcc_boots = np.empty(n_resamples) if compute_mcc_ci else None

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        t_boot = target[idx]
        p_boot = predicted[idx]
        acc_boots[i] = (p_boot == t_boot).mean()
        if compute_mcc_ci:
            mcc_boots[i] = compute_mcc(t_boot, p_boot)

    acc_lo, acc_hi = np.percentile(acc_boots, [2.5, 97.5])

    result = {
        "accuracy": acc_point,
        "acc_ci_lo": float(acc_lo),
        "acc_ci_hi": float(acc_hi),
        "acc_ci_width": float(acc_hi - acc_lo),
        "mcc": mcc_point,
    }

    if compute_mcc_ci:
        mcc_lo, mcc_hi = np.percentile(mcc_boots, [2.5, 97.5])
        result["mcc_ci_lo"] = float(mcc_lo)
        result["mcc_ci_hi"] = float(mcc_hi)
    else:
        # Structurally zero — constant predictor
        result["mcc_ci_lo"] = 0.0
        result["mcc_ci_hi"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_ci_figure(results: dict):
    """Horizontal error bar chart: accuracy point estimates with 95% CIs."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 13, "figure.facecolor": "white",
        "savefig.bbox": "tight",
    })

    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    lo = [results[n]["acc_ci_lo"] * 100 for n in names]
    hi = [results[n]["acc_ci_hi"] * 100 for n in names]
    errors_lo = [a - l for a, l in zip(accs, lo)]
    errors_hi = [h - a for a, h in zip(accs, hi)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = range(len(names))

    ax.barh(y_pos, accs, xerr=[errors_lo, errors_hi],
            color=["#888888", "#4878CF", "#6ACC65", "#D65F5F", "#E5A03A"],
            edgecolor="black", linewidth=0.4, capsize=4, height=0.5)

    ax.axvline(x=57.5, color="black", linestyle="--", linewidth=1.2,
               label="Majority baseline (57.5%)")

    for i, (a, l, h) in enumerate(zip(accs, lo, hi)):
        ax.text(h + 0.3, i, f"{a:.1f}% [{l:.1f}, {h:.1f}]",
                va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("95% Bootstrap Confidence Intervals (10,000 resamples)")
    ax.set_xlim(45, 68)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    path = Path("results/figures/bootstrap_ci.pdf")
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
    logging.info("COM3001 — Bootstrap Confidence Intervals")
    logging.info("=" * 60)

    config = load_config()
    seed = config["reproducibility"]["random_seed"]
    n_resamples = 10_000

    logging.info(f"Resamples:   {n_resamples:,}")
    logging.info(f"Seed:        {seed}")
    logging.info(f"CI method:   Percentile bootstrap (Efron & Tibshirani, 1993)")

    # ---- Load predictions ----
    pred_files = {
        "GeoBM":       "results/predictions/geobm_predictions.csv",
        "GA":          "results/predictions/ga_predictions.csv",
        "XGBoost-v1":  "results/predictions/xgb_predictions.csv",
        "XGBoost-v2":  "results/predictions/xgb_v2_predictions.csv",
    }

    predictions = {}
    for name, path in pred_files.items():
        df = pd.read_csv(path)
        predictions[name] = df
        logging.info(f"Loaded {name}: {path} ({len(df)} rows)")

    target = predictions["GeoBM"]["target"].values
    n_test = len(target)
    base_rate = target.mean()

    # ---- Majority baseline ----
    majority_predicted = np.ones(n_test, dtype=int)
    majority_ci = bootstrap_ci(
        target, majority_predicted, n_resamples, seed, compute_mcc_ci=False
    )

    # ---- Model CIs ----
    # GeoBM is a constant "up" predictor — MCC structurally zero
    is_constant = {
        "GeoBM": True,
        "GA": False,
        "XGBoost-v1": False,
        "XGBoost-v2": False,
    }

    results = {"Majority": majority_ci}

    for name, df in predictions.items():
        predicted = df["predicted"].values
        ci = bootstrap_ci(
            target, predicted, n_resamples, seed,
            compute_mcc_ci=not is_constant[name]
        )
        results[name] = ci

    # ---- Log CI table ----
    logging.info("-" * 90)
    logging.info("BOOTSTRAP CONFIDENCE INTERVALS (95%, 10,000 resamples)")
    logging.info(
        f"  {'Model':<14s} {'Acc':>6s} {'95% CI Acc':>16s} {'Width':>6s} "
        f"{'MCC':>7s} {'95% CI MCC':>18s} {'Incl 57.5%':>11s}"
    )
    logging.info(
        f"  {'-'*14} {'-'*6} {'-'*16} {'-'*6} "
        f"{'-'*7} {'-'*18} {'-'*11}"
    )

    for name, r in results.items():
        acc_str = f"{100*r['accuracy']:.1f}%"
        ci_acc = f"[{100*r['acc_ci_lo']:.1f}, {100*r['acc_ci_hi']:.1f}]"
        width = f"{100*r['acc_ci_width']:.1f}pp"

        if is_constant.get(name, name == "Majority"):
            mcc_str = "0.000*"
            ci_mcc = "[0.000, 0.000]*"
        else:
            mcc_str = f"{r['mcc']:.3f}"
            ci_mcc = f"[{r['mcc_ci_lo']:.3f}, {r['mcc_ci_hi']:.3f}]"

        includes_baseline = (r["acc_ci_lo"] <= 0.575 <= r["acc_ci_hi"])
        incl_str = "Yes" if includes_baseline else "No"
        if name == "Majority":
            incl_str = "—"

        logging.info(
            f"  {name:<14s} {acc_str:>6s} {ci_acc:>16s} {width:>6s} "
            f"{mcc_str:>7s} {ci_mcc:>18s} {incl_str:>11s}"
        )

    logging.info("  * Structurally zero — constant predictor, MCC = 0 on every resample")

    # ---- Interpretation ----
    logging.info("-" * 50)
    logging.info("INTERPRETATION")

    # GeoBM vs GA overlap
    geobm_ci = results["GeoBM"]
    ga_ci = results["GA"]
    overlap_geobm_ga = (geobm_ci["acc_ci_lo"] <= ga_ci["acc_ci_hi"] and
                        ga_ci["acc_ci_lo"] <= geobm_ci["acc_ci_hi"])
    logging.info(f"  GeoBM and GA accuracy CIs overlap: {overlap_geobm_ga}")
    if overlap_geobm_ga:
        logging.info("    → Consistent with McNemar's non-significance (p=0.4795)")

    # Any model CI excludes baseline?
    for name in ["GeoBM", "GA", "XGBoost-v1", "XGBoost-v2"]:
        r = results[name]
        if r["acc_ci_hi"] < 0.575:
            logging.info(
                f"  {name} CI entirely below 57.5%: "
                f"statistically confirmed worse than baseline"
            )
        elif r["acc_ci_lo"] > 0.575:
            logging.info(
                f"  {name} CI entirely above 57.5%: "
                f"statistically confirmed better than baseline"
            )
        else:
            logging.info(
                f"  {name} CI includes 57.5%: "
                f"cannot reject equivalence with baseline"
            )

    # Any MCC CI excludes zero?
    for name in ["GA", "XGBoost-v1", "XGBoost-v2"]:
        r = results[name]
        if r["mcc_ci_hi"] < 0:
            logging.info(
                f"  {name} MCC CI entirely below 0: "
                f"statistically confirmed worse than random"
            )
        elif r["mcc_ci_lo"] > 0:
            logging.info(
                f"  {name} MCC CI entirely above 0: "
                f"statistically confirmed better than random"
            )
        else:
            logging.info(
                f"  {name} MCC CI includes 0: "
                f"cannot reject random-level performance"
            )

    # CI widths
    widths = [100 * results[n]["acc_ci_width"] for n in results]
    logging.info(f"  Accuracy CI widths: {min(widths):.1f}pp to {max(widths):.1f}pp")
    logging.info(f"  (On 501 observations, ±4pp is expected)")

    # ---- Save CSV ----
    csv_rows = []
    for name, r in results.items():
        csv_rows.append({
            "model": name,
            "accuracy": r["accuracy"],
            "acc_ci_lo": r["acc_ci_lo"],
            "acc_ci_hi": r["acc_ci_hi"],
            "acc_ci_width": r["acc_ci_width"],
            "mcc": r["mcc"],
            "mcc_ci_lo": r["mcc_ci_lo"],
            "mcc_ci_hi": r["mcc_ci_hi"],
            "includes_baseline": r["acc_ci_lo"] <= 0.575 <= r["acc_ci_hi"],
        })

    csv_path = Path("results/tables/bootstrap_ci.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logging.info(f"Saved table: {csv_path}")

    # ---- Figure ----
    generate_ci_figure(results)

    # ---- Summary ----
    logging.info("=" * 50)
    logging.info("BOOTSTRAP CI COMPLETE")
    logging.info(f"  Models:      {len(results)}")
    logging.info(f"  Resamples:   {n_resamples:,}")
    logging.info(f"  Table:       {csv_path}")
    logging.info(f"  Figure:      results/figures/bootstrap_ci.pdf")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
