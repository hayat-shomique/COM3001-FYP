# Component Note: temporal_split.py

**File:** `src/splitting/temporal_split.py`
**Date:** 2026-04-01

---

## Purpose

Reads the feature-bearing dataset (`data/processed/spy_featured.csv`), partitions it into training and test sets at a strict chronological boundary (train ≤ 2022-12-31, test ≥ first trading day in 2023), verifies zero date overlap between the partitions, logs class distributions and date boundaries, and saves the results to `data/processed/spy_train.csv` and `data/processed/spy_test.csv`.

This module operationalises the controlled evaluation boundary of the experiment. In any supervised prediction task, the validity of performance claims depends entirely on the separation between the data used to estimate model parameters and the data used to assess those parameters. This module enforces that separation. Everything on or before the boundary is parameter-estimation history: models train on it, fit to it, and learn from it. Everything after the boundary is held-out evaluation data: models are assessed against it but are never exposed to it during training. The split is therefore the single architectural decision that determines whether downstream accuracy figures reflect genuine out-of-sample predictive ability or memorisation of training observations. Every performance claim in the evaluation chapter depends on this boundary being enforced without exception.

---

## Why temporal splitting is a separate module

The upstream pipeline follows strict separation of concerns: ingestion preserves raw CRSP data, preprocessing standardises CRSP conventions, target construction encodes the supervised learning task, and feature engineering computes the predictor variables. The temporal split is the fifth distinct concern — it partitions the complete, feature-bearing dataset into non-overlapping chronological segments for model training and out-of-sample evaluation.

This separation matters for two reasons. First, the split boundary is an evaluation design decision, not a data property. Which date divides training from testing depends on the experimental protocol: how many out-of-sample years are needed, whether the test period should span identifiable market regimes, and how much training history each model requires for stable parameter estimation. Embedding this decision in the feature engineering module would conflate feature computation with evaluation design, making it harder to audit either independently. Second, the split must be verifiable for temporal leakage as an isolated concern. A single-responsibility module with one input, one boundary, and two outputs makes it straightforward to prove that no test-period observation contaminates the training set — and that proof is logged on every run.

---

## Why chronological splitting is required for time-series prediction

Financial time series exhibit serial dependence — today's returns, volatility, and market regime are statistically correlated with yesterday's and last week's. Any splitting method that ignores the temporal ordering of observations risks producing optimistic performance estimates that do not reflect real-world predictive ability. The specific mechanisms by which non-chronological splits invalidate evaluation in this pipeline are:

**1. Causal ordering violation.** The features in this pipeline are backward-looking by construction: every rolling window and lag computation uses only data at or before time *t*, enforced by the anti-leakage discipline in `build_features.py`. The features themselves do not leak future information. The problem that a chronological split prevents is different and more fundamental: if post-boundary observations were permitted in the training set, the model would be trained on periods it is subsequently asked to predict. It would not be learning from the future in a feature-engineering sense — it would be evaluated on data it had already seen during parameter estimation. The resulting accuracy figures would reflect memorisation, not generalisation, producing optimistically biased estimates with no bearing on real out-of-sample performance. A strict chronological boundary prevents this by ensuring that every observation in the test set is strictly posterior to every observation in the training set.

**2. Regime contamination.** Market regimes (sustained bull or bear trends, periods of elevated or suppressed volatility) persist for months or years. A non-chronological split would scatter observations from the same regime across both partitions, allowing the model to memorise regime-specific statistical signatures during training and then appear to generalise when the test set contains observations drawn from the same regime. The model would be interpolating within a familiar distribution, not predicting into an unseen period. A chronological split ensures that the test period constitutes a genuinely novel regime sequence that the model has never encountered.

**3. Autocorrelation leakage.** Daily equity returns exhibit short-term autocorrelation: adjacent trading days are not statistically independent. If day *t* is in the training set and day *t+1* is in the test set under a random split, the features at *t+1* (which include lagged returns from *t*) provide the model with information about a day whose neighbours it trained on, enabling interpolation rather than prediction. A chronological split with a clean temporal boundary — where the last training day (2022-12-30) is separated from the first test day (2023-01-03) by a market closure — eliminates this leakage path entirely.

These three mechanisms correspond to the methodological standard established by López de Prado (2018, *Advances in Financial Machine Learning*): "In finance, observations are not IID. Any evaluation methodology that treats them as such — including random train/test splits and standard k-fold cross-validation — will produce optimistically biased performance estimates." The chronological split is not a refinement; it is the minimum requirement for a valid backtesting evaluation on serially dependent data.

---

## Rejected alternatives

### Random split

A random 80/20 split (`sklearn.model_selection.train_test_split` with `shuffle=True`) is the default in introductory machine learning practice. It is invalid for time-series prediction because it destroys causal ordering: test observations from 2023 would be interspersed with training observations from 2024, and the model would be assessed on data it trained on within the same market regime. For IID tabular data where observations are exchangeable, random splitting is methodologically sound. For serially dependent financial data, it produces accuracy estimates that are systematically higher than what the model would achieve in forward deployment. Every published backtesting methodology warns against this (Pardo, 2008; Bailey et al., 2014; López de Prado, 2018).

### Stratified split

Stratified splitting preserves the class ratio (up/down) in both partitions. While this addresses class imbalance, it requires shuffling observations across time to balance the classes — reintroducing the causal ordering violation that a chronological split is designed to prevent. In this dataset, the class imbalance is mild: 55.0% up-days in training, 57.5% in testing. Stratification would provide negligible distributional benefit while invalidating the temporal integrity of the evaluation. If the imbalance were severe, the correct remedy would be class weighting or oversampling within the training set after the temporal split, not stratified splitting across time.

### Standard k-fold cross-validation

Standard k-fold CV randomly partitions the dataset into *k* folds and rotates the test fold. Like random splitting, this destroys temporal ordering. The time-series-aware alternative — walk-forward validation or purged k-fold CV (López de Prado, 2018) — preserves causal ordering and is methodologically sound, but it is an evaluation-stage concern, not a data-splitting concern. This module performs the single chronological holdout split that establishes the primary out-of-sample evaluation boundary. Walk-forward validation, if implemented, would operate on sub-partitions of the training set within the evaluation module.

---

## Scope boundaries

This section documents what this module deliberately does not do, and why each exclusion is a bounded design choice rather than an omission.

**Scaling and normalisation** are model-dependent transformations. XGBoost is invariant to monotonic feature transformations and does not require scaling. The GA's fitness function operates on simulated trading returns, not raw feature magnitudes. GeoBM estimates drift and volatility from the raw return series, not from engineered features. Scaling belongs in the model training stage, fitted on the training set only, with learned parameters carried forward to the test set to prevent information leakage through the scaler's statistics. This module passes all 22 columns through unmodified.

**Walk-forward and expanding-window validation** strengthen evaluation robustness by testing the model across multiple non-overlapping temporal windows. They are evaluation-stage concerns that operate on sub-partitions of the training set. This module's responsibility ends at the primary train/test boundary; walk-forward logic, if implemented, sits in the evaluation module.

**Embargo and purge gaps** are not applied. López de Prado's (2018) purging framework is designed for event-labelling schemes with variable-length outcome windows — for example, triple-barrier labels where a single label may span days or weeks and whose outcome window could straddle the train/test boundary. In this pipeline, the target is a 1-day-ahead binary label: the outcome of the prediction made at close on day *t* is fully determined by the return on day *t+1*. There are no overlapping label windows and no multi-day outcome horizons, so there is no structural cross-boundary leakage that purging would address. Applying purging here would remove observations from the boundary region without methodological justification — a form of overclaiming methodological rigour that does not match the label structure of this experiment.

**Validation sets** for hyperparameter tuning are deferred to the model training stage. A train/validation/test three-way split would prevent test-set contamination during tuning. The training set (3,252 observations spanning 13 years) provides sufficient depth for a further chronological validation partition downstream. This module enforces the primary boundary; how the training set is subdivided for tuning is a model-stage decision.

---

## Baseline logic

The test-period class distribution — 288 up-days (57.5%) and 213 down-days (42.5%) — establishes the naive majority-class baseline against which all three models must be assessed. A classifier that predicts "up" on every test day achieves 57.5% accuracy by construction, without learning any pattern from the feature set and without estimating any parameters. This is the zero-skill threshold: any model that fails to exceed 57.5% on the test set demonstrates no predictive value beyond the directional bias already present in the evaluation period.

This baseline is specific to the test period. The training period has a different class balance (55.0% up), so the majority-class rate is not a property of the asset in general — it is a property of the particular 501-day evaluation window defined by the split boundary. This distinction matters: a model that achieves 58% accuracy on a 57.5% baseline has demonstrated marginal predictive value, not strong predictive ability. The evaluation chapter must interpret model accuracy relative to this specific threshold, not relative to a naive 50% coin-flip assumption.

---

## Rubric mapping (COM3001 assessor language)

| Assessor criterion | How this component satisfies it |
|---|---|
| **Requirements analysed** | The comparative framework requires that all three paradigms are evaluated on identical held-out data that none of them trained on. A strict chronological split is the minimum architectural requirement for valid backtesting of serially dependent time-series models. |
| **Design issues discussed** | Chronological splitting justified against three rejected alternatives (random, stratified, k-fold). Three temporal leakage mechanisms explained with precise language (causal ordering violation, regime contamination, autocorrelation leakage). Embargo/purge explicitly scoped out against the project's 1-day label structure rather than applied mechanically. Separation from feature engineering justified as distinct evaluation-design concern. |
| **Challenging aspects** | The split operation itself is computationally trivial. The challenge lies in justifying why the trivial operation is the correct one: demonstrating that simpler alternatives (random split, stratified split) are invalid for serially dependent data, explaining the specific leakage mechanisms they introduce, and scoping out more complex alternatives (purged k-fold) with precise reference to the project's label structure rather than applying them uncritically. |
| **Replication detail** | Split boundary read from `config/data_config.yaml`. Zero-overlap verification logged on every run. Class balance reported for both partitions. Exact row counts, date ranges, and class distributions recorded. Another researcher can reproduce this step by running `python src/splitting/temporal_split.py` from the repo root. |
| **Advanced knowledge demonstrated** | Temporal leakage prevention in financial backtesting (López de Prado, 2018; Pardo, 2008). Distinction between IID evaluation methodology and time-series evaluation methodology. Embargo/purge gaps scoped against the project's label structure rather than imported wholesale. Majority-class baseline derived from the test-set class distribution to anchor all downstream model evaluation. |

---

## How this feeds the dissertation

**Methodology chapter, "Train/Test Split" subsection (~400–500 words).** The chronological split rationale, the three leakage mechanisms, the rejected alternatives, and the embargo scoping translate directly into methodology prose. The evidence table becomes a figure or inline summary.

> *Ready-to-lift wording (methodology):* "The feature-bearing dataset (3,753 observations, February 2010 to December 2024) was partitioned into training and test sets using a strict chronological split. All observations on or before 31 December 2022 were assigned to the training set (3,252 rows); all observations from the first trading day of 2023 onwards were assigned to the test set (501 rows). No randomisation or shuffling was applied. Zero-overlap verification confirmed that the last training date (30 December 2022) strictly precedes the first test date (3 January 2023), with no date appearing in both partitions. This design prevents three forms of temporal leakage that would produce optimistically biased performance estimates: causal ordering violation, where the model trains on observations it is subsequently asked to predict, measuring memorisation rather than generalisation; regime contamination, where observations from the same persistent market regime appear in both partitions, enabling interpolation within a familiar distribution; and autocorrelation leakage, where adjacent days' serial dependence provides the model with information about training-set neighbours of test observations. Random splitting, stratified splitting, and standard k-fold cross-validation were each rejected because they violate the temporal ordering assumption that is fundamental to valid time-series evaluation (López de Prado, 2018; Pardo, 2008). No embargo or purge gap was applied: the target is a 1-day-ahead binary label with no overlapping outcome windows, so there is no structural cross-boundary leakage that purging would address (López de Prado, 2018, ch. 7)."

**Evaluation chapter, "Baseline and Interpretation Framework" paragraph (~200–300 words).** The majority-class baseline and its interpretation anchor the opening of the evaluation results.

> *Ready-to-lift wording (evaluation):* "The test-period class distribution — 288 up-days (57.5%) and 213 down-days (42.5%) — establishes the naive majority-class baseline against which all three models are assessed. A classifier that predicts 'up' on every test day achieves 57.5% accuracy by construction, without learning any pattern from the feature set and without estimating any parameters. This is the zero-skill threshold: any model that fails to exceed it demonstrates no predictive value beyond the directional bias already present in the evaluation period. This baseline is specific to the 501-day test window (January 2023 to December 2024) and differs from the training-period majority-class rate (55.0%), so it cannot be assumed as a fixed property of the asset. Model accuracy must therefore be interpreted relative to the test-specific 57.5% threshold, not relative to a naive 50% coin-flip assumption. A model achieving, for example, 58% accuracy on a 57.5% baseline has demonstrated marginal predictive value — a finding that is informative for the comparative framework but should not be overclaimed as strong predictive ability."

---

## Evidence produced

This component partitioned 3,753 rows from `data/processed/spy_featured.csv` into two non-overlapping chronological partitions at the 2022-12-31 boundary. No rows were gained or lost (3,252 + 501 = 3,753). The exact outputs:

| Metric | Value |
|---|---|
| Input rows | 3,753 |
| Train rows | 3,252 |
| Test rows | 501 |
| Rows lost | 0 |
| Train date range | 2010-02-02 to 2022-12-30 |
| Test date range | 2023-01-03 to 2024-12-30 |
| Train class balance | 1,788 up (55.0%) / 1,464 down (45.0%) |
| Test class balance | 288 up (57.5%) / 213 down (42.5%) |
| Majority-class baseline (test) | 57.5% (predict "up" on every day) |
| Zero overlap | Confirmed — last train date (2022-12-30) strictly precedes first test date (2023-01-03) |

Output saved to `data/processed/spy_train.csv` and `data/processed/spy_test.csv`.
