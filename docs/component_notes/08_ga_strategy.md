# Component Note: ga_strategy.py

**File:** `src/models/genetic/ga_strategy.py`
**Date:** 2026-04-02

---

## Purpose

Evolves a small set of interpretable threshold rules over the 14 engineered features using DEAP, evaluates the best evolved individual on the held-out test set for next-day directional prediction, and saves comparison-compatible predictions to `results/predictions/ga_predictions.csv`.

This module is the second of three modelling paradigms in the project's comparative framework. Its role is to test whether a bounded evolutionary search over the engineered feature space can discover directional prediction logic that exceeds the GeoBM / majority-class baseline (57.5%) under the same held-out temporal evaluation boundary. The GA operationalises the hypothesis that simple, interpretable threshold rules over technical features contain directional information beyond what the unconditional return distribution provides.

---

## Why the GA is a separate modelling paradigm

The three-model comparison evaluates fundamentally different approaches to the same prediction task. GeoBM derives predictions from the unconditional return distribution (two parameters, no features). XGBoost learns non-linear decision boundaries from the full feature matrix. The GA occupies a distinct position: it evolves interpretable, human-readable threshold rules over the feature space using a stochastic search procedure with no gradient information.

This separation reflects a genuine difference in modelling philosophy, not an arbitrary division. The GA does not learn a function approximation — it searches a structured hypothesis space of majority-vote threshold rules. The representation, search mechanism, and output are all qualitatively different from the other two paradigms, which is what makes the comparison informative.

---

## Why GA is appropriate in this project

Genetic algorithms are well-established in the evolutionary computation literature as a method for exploring structured hypothesis spaces where the fitness landscape is non-differentiable and the solution representation is discrete or mixed (Holland, 1975; Goldberg, 1989; Eiben & Smith, 2015). In the context of financial direction prediction, GAs have been applied to evolve trading rules over technical indicators (Allen & Karjalainen, 1999, *Journal of Financial Economics*; Chen, 2002), making them a natural fit for this project's feature-based directional prediction task.

The GA is appropriate for three specific reasons:

1. **It tests the feature space directly.** GeoBM ignores the 14 engineered features entirely. The GA searches over them. If the GA cannot exceed the GeoBM baseline, the implication is that the threshold-rule hypothesis family does not capture useful directional structure in the features — a finding that narrows the space of viable modelling approaches before XGBoost is attempted.

2. **It produces interpretable output.** The best evolved individual is a small set of human-readable threshold rules (e.g., "if close_sma20_ratio < 1.036 then vote UP"). This interpretability is valuable for the dissertation: the examiner can inspect exactly what the evolutionary search discovered, not just its accuracy.

3. **It demonstrates evolutionary computation as a CS paradigm.** The COM3001 rubric rewards "knowledge beyond taught material." Evolutionary search is a distinct computational paradigm from the supervised learning and stochastic modelling approaches represented by the other two models. Including it broadens the methodological range of the comparison.

---

## The role of GA in the three-model comparison

| Model | Assumptions | Inputs used | Prediction mechanism |
|---|---|---|---|
| GeoBM | Returns are IID log-normal | Raw returns only (2 parameters) | Analytic probability |
| GA | Threshold rules over features can predict direction | Engineered feature space | Evolved directional rules |
| XGBoost | Non-linear feature interactions predict direction | 14 engineered features | Learned decision trees |

The GA was designed to occupy the middle position in the three-paradigm comparison by expressive capacity — it uses engineered features, unlike GeoBM, but with a restricted hypothesis family of threshold-based majority-vote rules, unlike XGBoost, which can learn arbitrary tree-based decision boundaries. Whether this additional capacity translates into better predictions is an empirical question. On this dataset, it did not: the GA achieved 56.7% against GeoBM's 57.5%. If XGBoost exceeds both, the margin quantifies the value of non-linear modelling capacity over interpretable threshold rules.

---

## Chromosome / individual representation

Each individual encodes 3 threshold rules. Each rule occupies 3 genes (9 genes total), all floats in approximately [0, 1]:

| Gene | Role | Decoding |
|---|---|---|
| gene[i*3 + 0] | Feature selector | Clipped to [0, 1], scaled to feature index [0, 13] |
| gene[i*3 + 1] | Threshold percentile | Clipped to [0, 1], mapped to [feature_min, feature_max] from training data |
| gene[i*3 + 2] | Comparison direction | >0.5 means ">", ≤0.5 means "<" |

Each rule decodes into: "if feature_name {>|<} threshold_value then vote UP, else vote DOWN." The final prediction is the majority vote across the 3 rules: predict UP if at least 2 of 3 rules vote UP.

**Why 3 rules.** Three rules is the smallest odd number that supports a non-degenerate majority vote. With 1 rule, the individual reduces to a single threshold comparison with no voting capacity. With 5 or more rules, the chromosome grows and the search space expands without proportionate benefit for a proof-of-concept baseline. Three rules provide the simplest non-trivial voting scheme while keeping the individual fully interpretable — every evolved solution can be printed as 3 human-readable lines.

**Why percentile encoding.** The 14 features have different scales (RSI ranges 0–100, returns range ±0.1, MA ratios cluster near 1.0). Encoding thresholds as percentiles in [0, 1], mapped to each feature's training-set [min, max], normalises the search space so that crossover and mutation operate uniformly across all genes regardless of the underlying feature scale. Genes may drift outside [0, 1] during mutation; clipping during decoding ensures valid phenotypes. This is standard practice in real-coded evolutionary computation (Herrera, Lozano & Verdegay, 1998).

---

## Feature subset policy

All 14 engineered features are available as candidates. The GA implicitly selects which features to use via the feature-selector gene in each rule. With 3 rules, the individual uses at most 3 features simultaneously — the effective feature subset is bounded by the chromosome size, not by a pre-filter.

Pre-filtering to a smaller candidate set was considered and rejected. Any manual feature selection before evolution would impose human bias on the search and reduce the GA's opportunity to discover unexpected feature combinations. The GA's search mechanism already performs implicit feature selection: features that do not contribute to training accuracy will not survive tournament selection. With only 14 candidates and 3 rules, the combinatorial space (14^3 = 2,744 feature triples) is small enough that 100 × 50 = 5,000 evaluations can explore it adequately.

---

## Fitness function

Training-set classification accuracy: the proportion of training rows where the individual's majority-vote prediction matches the actual target.

Plain accuracy was chosen over balanced accuracy for consistency. The test-set evaluation metric is also plain accuracy, and the majority-class baseline (57.5%) is defined in accuracy terms. Using a different metric for fitness would create a train/eval mismatch that complicates interpretation. If the GA converges to a majority-class-mimicking predictor under plain accuracy, that is itself an informative finding — it means the search could not improve on the class prior.

---

## Crossover, mutation, and selection

| Operator | Choice | Justification |
|---|---|---|
| Selection | Tournament (size 3) | Standard selection pressure for small populations. Larger tournaments risk premature convergence; smaller tournaments slow convergence. |
| Crossover | Two-point, p=0.7 | Two-point crossover preserves rule-level structure (3 genes per rule) better than uniform crossover. The 0.7 probability is a standard default (De Jong, 2006). |
| Mutation | Gaussian (mu=0, sigma=0.2, per-gene p=0.2) | Gaussian mutation provides local search around promising regions. Sigma=0.2 relative to the [0, 1] gene range balances exploration and exploitation. |
| Elitism | Hall of fame (size 1) | The best individual ever found is preserved. eaSimple does not re-insert elites into the population, but the hall of fame ensures the best solution is never lost. |

Population size (100) and generation count (50) are proportionate for the 9-dimensional search space. Larger budgets were not justified for a proof-of-concept baseline — the evolution converged by generation 20 in the real run.

---

## Why DEAP was used

DEAP (Distributed Evolutionary Algorithms in Python) is the standard Python library for evolutionary computation research. It provides a flexible toolbox architecture where representation, operators, and fitness are composed declaratively rather than subclassed. This matches the project's need for a transparent, auditable evolutionary search with minimal framework overhead.

Alternatives considered: PyGAD was considered but offers less operator flexibility. A from-scratch implementation was considered but would add 200+ lines of boilerplate for selection, crossover, and mutation logic that DEAP already provides correctly — with no methodological benefit for the dissertation.

---

## How the best individual generates test predictions

After evolution completes, the best individual from the hall of fame is decoded into 3 threshold rules. Each rule is applied to every row in the test set independently: the rule compares the row's feature value against its threshold and votes UP or DOWN. The final prediction for each row is the majority vote across the 3 rules.

The vote fraction (proportion of rules voting UP, e.g. 2/3 = 0.667) is recorded in the `p_up` column of the output CSV for comparison-harness compatibility with the GeoBM and XGBoost prediction schemas.

---

## Rejected alternatives

### Symbolic regression / tree-based GP

Genetic programming with tree-based individuals could evolve arbitrary mathematical expressions over the features. This was rejected because: (1) the hypothesis space is unbounded, making evolved solutions difficult to interpret and defend in a dissertation; (2) bloat control (tree depth limits, parsimony pressure) introduces additional hyperparameters that are difficult to justify for a proof-of-concept; (3) the threshold-rule representation is sufficient to test the core hypothesis (do the features contain directional structure?).

### Balanced accuracy as fitness

Balanced accuracy (mean of per-class recall) was considered to avoid convergence to a majority-class predictor. It was rejected for consistency: the evaluation metric is plain accuracy, and the baselines (57.5%) are defined in accuracy terms. Using balanced accuracy for fitness but plain accuracy for evaluation would create an interpretation mismatch.

### Chronological validation split within training

Carving out a validation set from the training data (e.g., last 2 years of training for validation, rest for fitness) was considered for hyperparameter tuning or early stopping. It was rejected because: (1) the GA's hyperparameters (population size, generations, operator rates) were set before the experiment, not tuned on any data; (2) the search space is small enough (9 genes) that overfitting to training data is bounded by the representation's capacity; (3) adding a validation split would reduce the training set available for fitness evaluation without addressing the fundamental single-holdout limitation, which must be acknowledged regardless.

---

## Challenging aspects and how they were resolved

### 1. The single-holdout evaluation limitation

The GA evolves rules using the full training set and is evaluated once on the held-out test set. There is no nested cross-validation or validation set for model selection. This means the test accuracy is a single-sample estimate of out-of-sample performance, and there is no way to separate the effect of the evolutionary search procedure (which features and thresholds were selected) from the effect of the particular test period.

This is acknowledged as a methodological limitation, not hidden. The GA's hyperparameters were fixed before the experiment (not tuned on test data), and the representation's bounded capacity (9 genes, 3 rules) limits the degree to which it can overfit. The train-test gap (-0.8pp in the real run) is small, suggesting that overfitting is not the dominant concern for this baseline. Walk-forward validation would strengthen the evaluation but is an evaluation-module concern, not a baseline-module concern.

The best evolved individual achieved 55.9% training accuracy against a training-set majority class of 55.0% — a margin of only 0.9 percentage points after 50 generations. This confirms that the threshold-rule representation is capacity-limited: even with full access to training labels, the GA cannot find rules that substantially exceed class-frequency prediction.

### 2. Near-constant prediction bias

The best evolved individual predicts UP on 96.4% of test days (483/501). While not technically a constant predictor, it is heavily biased toward UP. This bias is a consequence of the training-period class balance (55.0% up) and the positive-drift environment: rules that predict UP on most days achieve ~55% training accuracy, and the evolutionary search finds threshold conditions that only occasionally flip to DOWN.

This is surfaced in the diagnostics rather than hidden. The near-constant prediction pattern means the GA has not discovered strong directional structure — it has found a slightly refined version of the majority-class heuristic that applies feature-based exceptions on a small number of days.

All three evolved rules converged to moving-average ratio features with "<" comparisons, independently discovering mean-reversion logic — predict up when the price is below its trend. This is economically interpretable and connects to the established technical analysis literature on mean-reversion signals (Brock, Lakonishok & LeBaron, 1992). The GA was given no prior bias toward these features; their selection emerged from the evolutionary search.

Rule 1 (close_sma20_ratio < 0.9566) is strictly contained within Rule 2 (close_sma20_ratio < 1.0363) — every row where Rule 1 votes UP, Rule 2 also votes UP. Rule 1 never changes the majority-vote outcome. The effective model is therefore a 2-condition threshold rule: predict UP when close_sma20_ratio < 1.036 AND close_sma5_ratio < 1.024. This redundancy reveals that the evolutionary search has no parsimony pressure — eaSimple does not penalise redundant rules.

### 3. Feature range extrapolation

Threshold percentiles are decoded using training-set feature ranges. If test-set feature values fall outside the training range, the thresholds may not cover the same relative position in the feature distribution. This is a known limitation of any threshold-based method applied out-of-sample, but it is minor for this dataset: the test period (2023–2024) does not contain extreme outliers relative to the training period (2010–2022), which includes the COVID crash.

---

## Scope boundaries

**Rolling or adaptive re-evolution** would re-run the GA as new data arrives, updating rules to track regime changes. This would require using test-period data in the evolutionary search, violating the held-out evaluation boundary. Excluded by design.

**Trading strategy overlay** (position sizing, stop losses, transaction costs) is an evaluation-stage concern. This module produces directional predictions in a generic format; strategy-level evaluation operates on those predictions downstream.

**Feature scaling or normalisation** is not applied. The percentile-based threshold encoding handles cross-feature scale differences internally. External scaling would add a transformation step that is unnecessary for threshold comparisons.

**Hyperparameter tuning** (searching over population sizes, generation counts, operator rates) is not performed. The GA's hyperparameters were set at standard defaults before the experiment. Tuning them on the test set would invalidate the evaluation; tuning them on a training subset would reduce data available for fitness evaluation without materially improving a proof-of-concept baseline.

---

## Rubric mapping (COM3001 assessor language)

| Assessor criterion | How this component satisfies it |
|---|---|
| **Requirements analysed** | The comparative framework requires a feature-based model that tests whether the engineered features contain directional structure beyond the GeoBM baseline. The GA provides this as an interpretable evolutionary baseline. |
| **Design issues discussed** | Chromosome representation justified (3 rules, 9 genes, majority vote). Feature subset policy justified (all 14 available, implicitly selected). Fitness function justified (plain accuracy for train/eval consistency). Operator choices justified with literature. Rejected alternatives documented. |
| **Challenging aspects** | Single-holdout evaluation limitation acknowledged honestly. Near-constant prediction bias surfaced and interpreted. Feature range extrapolation discussed. Train-test gap logged explicitly. |
| **Replication detail** | All hyperparameters in config. Seed 42 applied to both random and numpy. Feature ranges computed from training only. Best individual logged in human-readable form. Another researcher can reproduce by running `python src/models/genetic/ga_strategy.py` from the repo root. |
| **Advanced knowledge demonstrated** | Evolutionary computation as a modelling paradigm distinct from gradient-based learning (Holland, 1975; Goldberg, 1989). Real-coded GA with percentile encoding for mixed-scale features. Implicit feature selection through chromosome design. DEAP toolbox architecture. Honest assessment of single-holdout evaluation limitations. |

---

## How this feeds the dissertation

**Methodology chapter, "Genetic Algorithm Baseline" subsection (~400–500 words).**

> *Ready-to-lift wording (methodology):* "The second paradigm in the comparative framework is a genetic algorithm that evolves interpretable threshold rules over the 14 engineered features. Each individual in the population encodes three threshold rules, each specifying a feature, a comparison direction, and a threshold value. The final directional prediction for each day is the majority vote across the three rules. The chromosome uses a real-coded percentile representation: feature selectors and threshold values are encoded as floats in [0, 1] and decoded relative to the training-set feature ranges, ensuring that crossover and mutation operate uniformly across features of different scales (Herrera, Lozano & Verdegay, 1998). The evolutionary search used DEAP (Fortin et al., 2012) with a population of 100 individuals, tournament selection (size 3), two-point crossover (p = 0.7), and Gaussian mutation (sigma = 0.2, per-gene probability = 0.2) over 50 generations. The fitness function was training-set classification accuracy, chosen for consistency with the test-set evaluation metric. The random seed (42) was set for both Python's random module and numpy to ensure full reproducibility. All feature value ranges were computed from training data only; no test information entered the evolutionary search at any point."

**Evaluation chapter, "GA Baseline Results" paragraph (~200–300 words).**

> *Ready-to-lift wording (evaluation):* "The best evolved GA individual achieved 56.7% accuracy on the 501-day test set, below the majority-class baseline of 57.5% and the equivalent GeoBM baseline. The evolved rules — three threshold conditions on close_sma20_ratio and close_sma5_ratio — produced a heavily up-biased prediction pattern (96.4% up predictions), indicating that the evolutionary search converged to a near-majority-class heuristic with minor feature-based exceptions. The train-test accuracy gap was -0.8 percentage points (55.9% training vs 56.7% test), indicating no evidence of overfitting — the model is underfitting rather than overfitting, unable to find threshold rules that reliably discriminate up-days from down-days even on the training data. This result is informative for the comparison framework: the GA's failure to exceed the majority-class baseline suggests that simple majority-vote threshold rules over the engineered feature space do not capture directional structure that the unconditional return distribution misses. The GA's `p_up` column represents the fraction of rules voting UP (a vote score), not a calibrated probability like GeoBM's analytic P(up). The question that passes to XGBoost is whether non-linear interaction patterns across the full feature set can succeed where interpretable threshold rules could not."

---

## Evidence produced

This component evolved threshold rules from 3,252 training observations and evaluated the best individual on 501 test observations. The exact outputs:

| Metric | Value |
|---|---|
| Training rows | 3,252 |
| Test rows | 501 |
| Feature candidates | 14 (all engineered features) |
| Chromosome | 3 rules, 9 genes, majority-vote prediction |
| Population size | 100 |
| Generations | 50 |
| Best training accuracy | 55.9% (1,818 / 3,252) |
| Best evolved rules | Rule 1: close_sma20_ratio < 0.9566 → UP |
| | Rule 2: close_sma20_ratio < 1.0363 → UP |
| | Rule 3: close_sma5_ratio < 1.0241 → UP |
| Test accuracy | 56.7% (284 / 501) |
| Majority-class baseline | 57.5% (288 / 501) |
| GeoBM baseline | 57.5% (288 / 501) |
| Exceeds baseline | No |
| Predicted up (test) | 483 (96.4%) |
| Predicted down (test) | 18 (3.6%) |
| Constant predictor | No (but heavily up-biased) |
| Train-test gap | -0.8pp (no overfitting) |
| Random seed | 42 |
| Prediction method | Majority vote of 3 threshold rules |

Output saved to `results/predictions/ga_predictions.csv` with columns: date, target, predicted, p_up, model.
