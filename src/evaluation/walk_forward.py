"""
COM3001 — Walk-Forward Temporal Robustness Evaluation
=======================================================
Retrains all three paradigms (GeoBM, GA, XGBoost-v2) from scratch on
three rolling temporal windows and evaluates on each window's held-out
test period. This tests whether the null result is specific to the
primary train/test split or holds across market regimes.

WINDOWS:
  1. Train 2010–2018, Test 2019–2020 (pre-COVID → includes COVID crash)
  2. Train 2012–2020, Test 2021–2022 (includes COVID → post-COVID recovery)
  3. Train 2014–2022, Test 2023–2024 (≈ existing holdout, sanity check)

MODELS RETRAINED PER WINDOW:
  GeoBM:      Analytic P(up) from training-period log returns
  GA:         DEAP evolution (3 rules, pop 100, gen 50, seed 42)
  XGBoost-v2: Fixed config (depth=2, trees=17, lr=0.1)

This is beyond-taught-material work (López de Prado, 2018; Pardo, 2008).

Usage:
    python src/evaluation/walk_forward.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from scipy.stats import norm
from deap import base, creator, tools, algorithms

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


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

WINDOWS = [
    {"name": "W1", "train_start": 2010, "train_end": 2018,
     "test_start": 2019, "test_end": 2020,
     "label": "Train 2010–2018, Test 2019–2020"},
    {"name": "W2", "train_start": 2012, "train_end": 2020,
     "test_start": 2021, "test_end": 2022,
     "label": "Train 2012–2020, Test 2021–2022"},
    {"name": "W3", "train_start": 2014, "train_end": 2022,
     "test_start": 2023, "test_end": 2024,
     "label": "Train 2014–2022, Test 2023–2024"},
]


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


def evaluate_predictions(target: np.ndarray, predicted: np.ndarray) -> dict:
    n = len(target)
    acc = float((predicted == target).mean())
    mcc = compute_mcc(target, predicted)
    pred_up_pct = 100 * float((predicted == 1).sum()) / n
    return {"accuracy": acc, "mcc": mcc, "pred_up_pct": pred_up_pct}


# ---------------------------------------------------------------------------
# GeoBM (inline — 10 lines of core logic)
# ---------------------------------------------------------------------------

def run_geobm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Analytic GeoBM directional prediction."""
    log_returns = np.log(1 + train_df["return"].values)
    mu_minus_half_sig2 = np.mean(log_returns)
    sigma = np.std(log_returns, ddof=1)
    mu = mu_minus_half_sig2 + 0.5 * sigma**2
    p_up = norm.cdf(mu_minus_half_sig2 / sigma)

    predicted_class = 1 if p_up > 0.5 else 0
    predicted = np.full(len(test_df), predicted_class)

    metrics = evaluate_predictions(test_df["target"].values, predicted)
    metrics.update({"mu": mu, "sigma": sigma, "p_up": p_up})
    return metrics


# ---------------------------------------------------------------------------
# GA (replicated from ga_strategy.py with same config)
# ---------------------------------------------------------------------------

def compute_feature_ranges(train_df: pd.DataFrame) -> dict:
    return {c: (float(train_df[c].min()), float(train_df[c].max()))
            for c in FEATURE_COLS}


def decode_individual(individual, feature_ranges, n_rules):
    n_features = len(FEATURE_COLS)
    rules = []
    for i in range(n_rules):
        feat_pct = np.clip(individual[i * 3], 0.0, 1.0)
        thresh_pct = np.clip(individual[i * 3 + 1], 0.0, 1.0)
        dir_pct = np.clip(individual[i * 3 + 2], 0.0, 1.0)
        feat_idx = int(round(feat_pct * (n_features - 1)))
        feat_name = FEATURE_COLS[feat_idx]
        fmin, fmax = feature_ranges[feat_name]
        threshold = fmin + thresh_pct * (fmax - fmin)
        direction = ">" if dir_pct > 0.5 else "<"
        rules.append({"feature": feat_name, "threshold": threshold,
                       "direction": direction})
    return rules


def predict_with_rules(rules, df, n_rules):
    votes = np.zeros(len(df))
    for rule in rules:
        vals = df[rule["feature"]].values
        if rule["direction"] == ">":
            votes += (vals > rule["threshold"]).astype(float)
        else:
            votes += (vals < rule["threshold"]).astype(float)
    return (votes > n_rules / 2).astype(int)


def run_ga(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> dict:
    """Evolve GA from scratch on this window's training data."""
    n_rules = 3
    feature_ranges = compute_feature_ranges(train_df)
    target_train = train_df["target"].values

    def evaluate_ind(individual):
        rules = decode_individual(individual, feature_ranges, n_rules)
        preds = predict_with_rules(rules, train_df, n_rules)
        return (float((preds == target_train).mean()),)

    # Clean DEAP slate
    for name in ("FitnessMax", "Individual"):
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("attr_gene", random.random)
    tb.register("individual", tools.initRepeat,
                creator.Individual, tb.attr_gene, n=n_rules * 3)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", evaluate_ind)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    tb.register("select", tools.selTournament, tournsize=3)

    # Reset seed per window so each evolution is an independent,
    # reproducible experiment starting from the same random state.
    # Different training data produces different evolutionary trajectories
    # despite identical initial populations.
    random.seed(seed)
    np.random.seed(seed)

    pop = tb.population(n=100)
    hof = tools.HallOfFame(1)
    pop, _ = algorithms.eaSimple(
        pop, tb, cxpb=0.7, mutpb=0.2, ngen=50,
        halloffame=hof, verbose=False,
    )

    best = hof[0]
    best_fitness = best.fitness.values[0]
    best_rules = decode_individual(best, feature_ranges, n_rules)

    test_preds = predict_with_rules(best_rules, test_df, n_rules)
    metrics = evaluate_predictions(test_df["target"].values, test_preds)
    metrics["best_fitness"] = best_fitness
    metrics["rules"] = best_rules
    return metrics


# ---------------------------------------------------------------------------
# XGBoost-v2 (fixed config, no re-tuning)
# ---------------------------------------------------------------------------

def run_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame,
                seed: int) -> dict:
    """Train XGBoost-v2 fixed config on this window."""
    model = xgb.XGBClassifier(
        max_depth=2, n_estimators=17, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        random_state=seed, use_label_encoder=False, verbosity=0,
    )
    model.fit(train_df[FEATURE_COLS].values, train_df["target"].values)
    preds = model.predict(test_df[FEATURE_COLS].values)
    return evaluate_predictions(test_df["target"].values, preds)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(all_results: list[dict]):
    """Grouped bar chart: 3 windows × 3 models + per-window baseline."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 13, "figure.facecolor": "white",
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = ["GeoBM", "GA", "XGBoost-v2"]
    colours = ["#4878CF", "#6ACC65", "#D65F5F"]
    bar_width = 0.22
    x = np.arange(len(all_results))

    for j, model in enumerate(model_names):
        accs = [r["models"][model]["accuracy"] * 100 for r in all_results]
        offset = (j - 1) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, label=model,
                      color=colours[j], edgecolor="black", linewidth=0.4)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=8)

    # Per-window baseline lines
    for i, r in enumerate(all_results):
        bl = r["majority_baseline"] * 100
        ax.plot([i - 0.4, i + 0.4], [bl, bl], "k--", linewidth=1.2)
        ax.text(i + 0.42, bl, f"{bl:.1f}%", va="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in all_results], fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Walk-Forward Temporal Robustness — Three Windows")
    ax.set_ylim(45, 65)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = Path("results/figures/walk_forward_stability.pdf")
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
    logging.info("COM3001 — Walk-Forward Temporal Robustness Evaluation")
    logging.info("=" * 60)

    config = load_config()
    seed = config["reproducibility"]["random_seed"]
    featured_path = config["features"]["output_path"]

    logging.info(f"Dataset:     {featured_path}")
    logging.info(f"Seed:        {seed}")
    logging.info(f"Windows:     {len(WINDOWS)}")

    # ---- Load full featured dataset ----
    full_df = pd.read_csv(featured_path, parse_dates=["date"])
    full_df["year"] = full_df["date"].dt.year
    logging.info(f"Loaded: {len(full_df)} rows "
                 f"({full_df['date'].min().date()} to {full_df['date'].max().date()})")

    # ---- Process each window ----
    all_results = []

    for w in WINDOWS:
        logging.info("=" * 60)
        logging.info(f"WINDOW {w['name']}: {w['label']}")
        logging.info("=" * 60)

        # Filter
        train_df = full_df[
            (full_df["year"] >= w["train_start"]) &
            (full_df["year"] <= w["train_end"])
        ].reset_index(drop=True)

        test_df = full_df[
            (full_df["year"] >= w["test_start"]) &
            (full_df["year"] <= w["test_end"])
        ].reset_index(drop=True)

        n_train = len(train_df)
        n_test = len(test_df)
        train_up = (train_df["target"] == 1).mean()
        test_up = (test_df["target"] == 1).mean()
        majority_bl = max(test_up, 1 - test_up)

        logging.info(f"  Train: {n_train} rows "
                     f"({train_df['date'].min().date()} to {train_df['date'].max().date()}) "
                     f"| {100*train_up:.1f}% up")
        logging.info(f"  Test:  {n_test} rows "
                     f"({test_df['date'].min().date()} to {test_df['date'].max().date()}) "
                     f"| {100*test_up:.1f}% up")
        logging.info(f"  Majority baseline: {100*majority_bl:.1f}%")

        window_result = {
            "name": w["name"], "label": w["label"],
            "n_train": n_train, "n_test": n_test,
            "train_up_pct": 100 * train_up,
            "test_up_pct": 100 * test_up,
            "majority_baseline": majority_bl,
            "models": {},
        }

        # ---- GeoBM ----
        geobm = run_geobm(train_df, test_df)
        window_result["models"]["GeoBM"] = geobm
        logging.info(f"  GeoBM:      μ={geobm['mu']:.6f}, σ={geobm['sigma']:.6f}, "
                     f"P(up)={geobm['p_up']:.4f} → "
                     f"acc={100*geobm['accuracy']:.1f}%")

        # ---- GA ----
        ga = run_ga(train_df, test_df, seed)
        window_result["models"]["GA"] = ga
        rules_str = " | ".join(
            f"{r['feature']} {r['direction']} {r['threshold']:.4f}"
            for r in ga["rules"]
        )
        logging.info(f"  GA:         best_fit={100*ga['best_fitness']:.1f}%, "
                     f"test_acc={100*ga['accuracy']:.1f}%")
        logging.info(f"              rules: {rules_str}")

        # ---- XGBoost-v2 ----
        xgb_res = run_xgboost(train_df, test_df, seed)
        window_result["models"]["XGBoost-v2"] = xgb_res
        logging.info(f"  XGBoost-v2: acc={100*xgb_res['accuracy']:.1f}%")

        # ---- Window comparison table ----
        logging.info("-" * 60)
        logging.info(f"  {'Model':<14s} {'Acc':>7s} {'vs Base':>8s} "
                     f"{'MCC':>7s} {'Pred↑%':>7s}")
        logging.info(f"  {'-'*14} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

        for mname in ["GeoBM", "GA", "XGBoost-v2"]:
            m = window_result["models"][mname]
            delta = m["accuracy"] - majority_bl
            logging.info(
                f"  {mname:<14s} {100*m['accuracy']:>6.1f}% "
                f"{100*delta:>+7.1f}pp "
                f"{m['mcc']:>7.4f} "
                f"{m['pred_up_pct']:>6.1f}%"
            )

        all_results.append(window_result)

    # ==================================================================
    # Stability Summary
    # ==================================================================
    logging.info("=" * 60)
    logging.info("STABILITY SUMMARY")
    logging.info("=" * 60)

    # Distinguish trivial (≤1.0pp) from substantive (>1.0pp) exceedances.
    # 1.0pp is well within the ±4.4pp bootstrap CI width (note 14).
    substantive_exceedance = False
    trivial_exceedances = []
    for r in all_results:
        for mname, m in r["models"].items():
            delta = m["accuracy"] - r["majority_baseline"]
            if delta > 0:
                if delta > 0.01:  # >1.0pp
                    substantive_exceedance = True
                    logging.info(
                        f"  {mname} SUBSTANTIVELY beats baseline in {r['name']}: "
                        f"{100*m['accuracy']:.1f}% > {100*r['majority_baseline']:.1f}% "
                        f"(+{100*delta:.1f}pp)"
                    )
                else:
                    trivial_exceedances.append(
                        f"{mname} in {r['name']} (+{100*delta:.1f}pp)"
                    )

    if trivial_exceedances:
        logging.info(f"  Trivial exceedances (≤1.0pp, within bootstrap noise):")
        for te in trivial_exceedances:
            logging.info(f"    {te}")

    if not substantive_exceedance:
        logging.info("  Null result holds: Yes — substantively.")
        logging.info("  No model beats the majority baseline by more than 1.0pp "
                     "in any window.")

    # Majority baselines across windows
    logging.info("-" * 50)
    logging.info("MAJORITY BASELINES ACROSS WINDOWS")
    for r in all_results:
        logging.info(f"  {r['name']} ({r['label']}): "
                     f"{100*r['majority_baseline']:.1f}% "
                     f"(test up-rate: {r['test_up_pct']:.1f}%)")

    # Cross-paradigm convergence check
    logging.info("-" * 50)
    logging.info("CROSS-PARADIGM CONVERGENCE (Pred Up% per window)")
    logging.info(f"  {'Window':<5s} {'GeoBM':>8s} {'GA':>8s} {'XGB-v2':>8s}")
    for r in all_results:
        logging.info(
            f"  {r['name']:<5s} "
            f"{r['models']['GeoBM']['pred_up_pct']:>7.1f}% "
            f"{r['models']['GA']['pred_up_pct']:>7.1f}% "
            f"{r['models']['XGBoost-v2']['pred_up_pct']:>7.1f}%"
        )

    # Window 3 vs main evaluation comparison
    logging.info("-" * 50)
    logging.info("WINDOW 3 vs MAIN EVALUATION (sanity check)")
    w3 = all_results[2]
    logging.info(f"  {'':14s} {'W3':>8s} {'Main':>8s}")
    logging.info(f"  {'GeoBM acc':<14s} "
                 f"{100*w3['models']['GeoBM']['accuracy']:>7.1f}% {'57.5%':>8s}")
    logging.info(f"  {'GA acc':<14s} "
                 f"{100*w3['models']['GA']['accuracy']:>7.1f}% {'56.7%':>8s}")
    logging.info(f"  {'XGB-v2 acc':<14s} "
                 f"{100*w3['models']['XGBoost-v2']['accuracy']:>7.1f}% {'55.7%':>8s}")
    logging.info("  (Minor differences expected due to different training periods)")

    # ---- Save CSV ----
    csv_rows = []
    for r in all_results:
        for mname, m in r["models"].items():
            csv_rows.append({
                "window": r["name"],
                "window_label": r["label"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "majority_baseline": r["majority_baseline"],
                "model": mname,
                "accuracy": m["accuracy"],
                "vs_baseline": m["accuracy"] - r["majority_baseline"],
                "mcc": m["mcc"],
                "pred_up_pct": m["pred_up_pct"],
            })

    csv_path = Path("results/tables/walk_forward_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logging.info(f"Saved table: {csv_path}")

    # ---- Figure ----
    generate_figure(all_results)

    # ---- Final ----
    logging.info("=" * 50)
    logging.info("WALK-FORWARD EVALUATION COMPLETE")
    logging.info(f"  Windows:         {len(all_results)}")
    logging.info(f"  Models per window: 3 (GeoBM, GA, XGBoost-v2)")
    logging.info(f"  Null result holds: "
                 f"{'Yes — substantively across all windows' if not substantive_exceedance else 'No — substantive exceedance detected'}")
    logging.info(f"  Table:           {csv_path}")
    logging.info(f"  Figure:          results/figures/walk_forward_stability.pdf")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
