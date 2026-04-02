"""
COM3001 — Genetic Algorithm Directional Baseline (DEAP)
========================================================
Evolves a small set of interpretable threshold rules over the engineered
feature space using DEAP, then evaluates the best evolved individual on
the held-out test set for next-day directional prediction.

INDIVIDUAL REPRESENTATION:
  Each individual encodes N threshold rules (default 3). Each rule is
  a triple of genes in [0, 1]:
    (feature_percentile, threshold_percentile, direction_percentile)
  During evaluation, these decode into:
    "if feature_name {>|<} threshold_value then vote UP"
  The final prediction is the majority vote across all rules.

SEARCH DESIGN:
  Population:    100 individuals
  Generations:   50
  Selection:     Tournament (size 3)
  Crossover:     Two-point, probability 0.7
  Mutation:      Gaussian (sigma=0.2, per-gene probability 0.2)
  Fitness:       Training-set classification accuracy
  Elitism:       Hall of fame preserves best individual across generations

ANTI-LEAKAGE DISCIPLINE:
  The evolutionary search operates on training data only.
  Feature value ranges for threshold decoding are computed from training
  data only. The test set is used exactly once, after evolution completes,
  to evaluate the best individual. No test data enters fitness evaluation,
  feature selection, or threshold interpretation.

This module does NOT:
  - Use test data in fitness evaluation or threshold search
  - Perform rolling re-evolution or adaptive parameter updates
  - Simulate trading strategies or compute PnL
  - Scale or normalise features
  - Use symbolic regression, grammar-based, or tree-based representations

Usage:
    python src/models/genetic/ga_strategy.py

Author: Shomique Hayat (sh02588@surrey.ac.uk)
Module: COM3001 Final Year Project
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from deap import base, creator, tools, algorithms


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
# Feature Range Computation (training data only)
# ---------------------------------------------------------------------------

def compute_feature_ranges(train_df: pd.DataFrame) -> dict:
    """
    Compute min/max for each feature from training data only.

    These ranges decode threshold percentiles into actual feature values.
    Using training-only ranges prevents any test information from
    influencing threshold interpretation.
    """
    ranges = {}
    for col in FEATURE_COLS:
        ranges[col] = (float(train_df[col].min()), float(train_df[col].max()))
    return ranges


# ---------------------------------------------------------------------------
# Individual Decoding and Prediction
# ---------------------------------------------------------------------------

def decode_individual(individual: list, feature_ranges: dict, n_rules: int) -> list:
    """
    Decode a DEAP individual into human-readable threshold rules.

    Each rule occupies 3 genes (all floats, clipped to [0, 1]):
      gene[i*3+0]: feature percentile -> index into FEATURE_COLS
      gene[i*3+1]: threshold percentile -> mapped to [feat_min, feat_max]
      gene[i*3+2]: direction percentile -> >0.5 means ">" comparison

    Clipping ensures that crossover/mutation artefacts outside [0, 1]
    produce valid phenotypes. This is standard practice in real-coded
    evolutionary computation.
    """
    n_features = len(FEATURE_COLS)
    rules = []

    for i in range(n_rules):
        feat_pct = np.clip(individual[i * 3], 0.0, 1.0)
        thresh_pct = np.clip(individual[i * 3 + 1], 0.0, 1.0)
        dir_pct = np.clip(individual[i * 3 + 2], 0.0, 1.0)

        feat_idx = int(round(feat_pct * (n_features - 1)))
        feat_name = FEATURE_COLS[feat_idx]
        feat_min, feat_max = feature_ranges[feat_name]
        threshold = feat_min + thresh_pct * (feat_max - feat_min)
        direction = ">" if dir_pct > 0.5 else "<"

        rules.append({
            "feature": feat_name,
            "feat_idx": feat_idx,
            "threshold": threshold,
            "direction": direction,
        })

    return rules


def predict_with_rules(
    rules: list, df: pd.DataFrame, n_rules: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate binary predictions using majority vote of threshold rules.

    Each rule votes UP (1) or DOWN (0) for each row. The final prediction
    is UP if strictly more than half of the rules vote UP.

    Returns
    -------
    (predictions, vote_fractions) : tuple
        predictions: int array of 0/1
        vote_fractions: float array in [0, 1], fraction of rules voting UP
    """
    votes = np.zeros(len(df))

    for rule in rules:
        values = df[rule["feature"]].values
        if rule["direction"] == ">":
            votes += (values > rule["threshold"]).astype(float)
        else:
            votes += (values < rule["threshold"]).astype(float)

    vote_fractions = votes / n_rules
    predictions = (votes > n_rules / 2).astype(int)

    return predictions, vote_fractions


def predict_with_individual(
    individual: list, df: pd.DataFrame, feature_ranges: dict, n_rules: int
) -> np.ndarray:
    """Decode individual into rules and return predictions."""
    rules = decode_individual(individual, feature_ranges, n_rules)
    predictions, _ = predict_with_rules(rules, df, n_rules)
    return predictions


# ---------------------------------------------------------------------------
# Fitness Function
# ---------------------------------------------------------------------------

def make_fitness_function(
    train_df: pd.DataFrame, feature_ranges: dict, n_rules: int
):
    """
    Create a fitness closure over training data.

    Fitness = classification accuracy on the training set.

    Plain accuracy is chosen over balanced accuracy for consistency:
    the test-set evaluation metric is also plain accuracy, and the
    majority-class baseline (57.5%) is defined in accuracy terms.
    Using a different metric for fitness would create a train/eval
    mismatch.
    """
    target = train_df["target"].values

    def evaluate(individual):
        predictions = predict_with_individual(
            individual, train_df, feature_ranges, n_rules
        )
        accuracy = float((predictions == target).mean())
        return (accuracy,)

    return evaluate


# ---------------------------------------------------------------------------
# DEAP Setup and Evolution
# ---------------------------------------------------------------------------

def setup_deap(
    train_df: pd.DataFrame, feature_ranges: dict, ga_cfg: dict
) -> base.Toolbox:
    """Configure DEAP creator, toolbox, and genetic operators."""
    n_rules = ga_cfg["n_rules"]
    n_genes = n_rules * 3

    # Clean slate for creator (safe for re-import)
    for name in ("FitnessMax", "Individual"):
        if hasattr(creator, name):
            delattr(creator, name)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Gene initialisation: uniform [0, 1]
    toolbox.register("attr_gene", random.random)
    toolbox.register(
        "individual", tools.initRepeat,
        creator.Individual, toolbox.attr_gene, n=n_genes
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness
    toolbox.register(
        "evaluate",
        make_fitness_function(train_df, feature_ranges, n_rules)
    )

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_evolution(
    toolbox: base.Toolbox, ga_cfg: dict, seed: int
) -> tuple:
    """
    Execute the evolutionary search.

    Seeds both Python's random and numpy's random for full reproducibility.
    Returns the best individual (from hall of fame) and the logbook.
    """
    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=ga_cfg["population_size"])
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("mean", np.mean)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=ga_cfg["crossover_prob"],
        mutpb=ga_cfg["mutation_prob"],
        ngen=ga_cfg["generations"],
        stats=stats, halloffame=hof, verbose=False,
    )

    return hof[0], logbook


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run GA directional baseline pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("COM3001 — Genetic Algorithm Directional Baseline (DEAP)")
    logging.info("=" * 60)

    config = load_config()
    ga_cfg = config["genetic"]
    seed = config["reproducibility"]["random_seed"]

    train_path = ga_cfg["train_path"]
    test_path = ga_cfg["test_path"]
    predictions_path = ga_cfg["predictions_path"]
    n_rules = ga_cfg["n_rules"]

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

    # ---- Compute feature ranges from training data only ----
    feature_ranges = compute_feature_ranges(train_df)

    # ---- Log configuration ----
    logging.info("-" * 50)
    logging.info("GA CONFIGURATION")
    logging.info(f"  Feature candidates:   {len(FEATURE_COLS)}")
    logging.info(f"  Rules per individual: {n_rules}")
    logging.info(f"  Genes per rule:       3 (feature_pct, threshold_pct, direction_pct)")
    logging.info(f"  Total genes:          {n_rules * 3}")
    logging.info(f"  Population size:      {ga_cfg['population_size']}")
    logging.info(f"  Generations:          {ga_cfg['generations']}")
    logging.info(f"  Crossover prob:       {ga_cfg['crossover_prob']}")
    logging.info(f"  Mutation prob:        {ga_cfg['mutation_prob']}")
    logging.info(f"  Selection:            Tournament (size 3)")
    logging.info(f"  Fitness function:     Training-set classification accuracy")
    logging.info(f"  Prediction rule:      Majority vote of {n_rules} threshold rules")

    # ---- Setup and run evolution ----
    toolbox = setup_deap(train_df, feature_ranges, ga_cfg)
    best_individual, logbook = run_evolution(toolbox, ga_cfg, seed)

    # ---- Decode and log best individual ----
    best_rules = decode_individual(best_individual, feature_ranges, n_rules)
    best_train_fitness = best_individual.fitness.values[0]

    # Evolution progress at milestones
    logging.info("-" * 50)
    logging.info("EVOLUTION PROGRESS")
    milestones = {0, 10, 20, 30, 40, ga_cfg["generations"]}
    for record in logbook:
        gen = record["gen"]
        if gen in milestones:
            logging.info(
                f"  Gen {gen:>3d}: max={record['max']:.4f}  "
                f"mean={record['mean']:.4f}"
            )

    # Best individual
    logging.info("-" * 50)
    logging.info("BEST EVOLVED INDIVIDUAL")
    logging.info(f"  Training accuracy: {best_train_fitness:.4f} "
                 f"({100 * best_train_fitness:.1f}%)")
    for i, rule in enumerate(best_rules):
        logging.info(
            f"  Rule {i + 1}: if {rule['feature']} {rule['direction']} "
            f"{rule['threshold']:.6f} then vote UP"
        )
    logging.info(
        f"  Decision: majority vote ({n_rules} rules, "
        f"need >{n_rules // 2} votes for UP)"
    )

    # ---- Generate test predictions ----
    test_predictions, vote_fractions = predict_with_rules(
        best_rules, test_df, n_rules
    )

    predictions = pd.DataFrame({
        "date": test_df["date"].values,
        "target": test_df["target"].values,
        "predicted": test_predictions,
        "p_up": vote_fractions,
        "model": "ga",
    })

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

    train_test_gap = best_train_fitness - test_accuracy

    logging.info("-" * 50)
    logging.info("TEST EVALUATION")
    logging.info(f"  Test accuracy:         {test_accuracy:.4f} "
                 f"({100 * test_accuracy:.1f}%)")
    logging.info(f"  Majority baseline:     {majority_baseline:.4f} "
                 f"({100 * majority_baseline:.1f}%)")
    logging.info(f"  GeoBM baseline:        0.5749 (57.5%)")

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

    logging.info("-" * 50)
    logging.info("OVERFITTING CHECK")
    logging.info(f"  Best training accuracy: {best_train_fitness:.4f} "
                 f"({100 * best_train_fitness:.1f}%)")
    logging.info(f"  Test accuracy:          {test_accuracy:.4f} "
                 f"({100 * test_accuracy:.1f}%)")
    logging.info(f"  Train-test gap:         {train_test_gap:+.4f} "
                 f"({100 * train_test_gap:+.1f}pp)")

    if abs(train_test_gap) > 0.05:
        logging.info(
            "  WARNING: Train-test gap exceeds 5pp — potential overfitting."
        )
    elif abs(train_test_gap) > 0.02:
        logging.info("  NOTE: Moderate train-test gap.")
    else:
        logging.info("  Train-test gap is small.")

    # ---- Save predictions ----
    out = Path(predictions_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    logging.info(f"Saved predictions: {out} ({len(predictions)} rows)")

    # ---- Final summary ----
    logging.info("=" * 50)
    logging.info("GA BASELINE COMPLETE")
    logging.info(f"  Train rows:            {len(train_df)}")
    logging.info(f"  Test rows:             {n_test}")
    logging.info(f"  Rules evolved:         {n_rules}")
    logging.info(f"  Best training acc:     {100 * best_train_fitness:.1f}%")
    logging.info(f"  Test accuracy:         {100 * test_accuracy:.1f}%")
    logging.info(f"  Majority baseline:     {100 * majority_baseline:.1f}%")
    logging.info(f"  GeoBM baseline:        57.5%")
    logging.info(f"  Train-test gap:        {100 * train_test_gap:+.1f}pp")
    logging.info(f"  Constant predictor:    {is_constant}")
    logging.info(f"  Output:                {predictions_path}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
