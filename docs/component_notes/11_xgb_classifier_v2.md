# Component Note: xgb_classifier_v2.py

**File:** `src/models/xgboost_model/xgb_classifier_v2.py`
**Date:** 2026-04-03

---

## Purpose

Addresses the 18.6pp overfitting gap from XGBoost-v1 by applying chronological validation within the training set, early stopping on validation logloss, and grid search over a small hyperparameter space. This is not a replacement for v1 — it is a disciplined refinement that answers the examiner's most important question: "Your model overfit — did you try to fix it?"

The answer: yes. The overfitting gap collapsed from 18.6pp (v1) to 0.6pp (v2). But the null result survives: v2 achieves 55.7% test accuracy, still below the 57.5% majority-class baseline. The failure to beat the baseline is confirmed under methodological refinement — it is a property of the feature set and the prediction task, not of the v1 hyperparameters.

---

## Why v2 exists

XGBoost-v1 achieved 71.7% training accuracy but only 53.1% test accuracy — an 18.6pp gap indicating significant overfitting. The conservative fixed hyperparameters (depth 3, subsample 0.8, colsample 0.8) were insufficient to prevent the model from memorising training-period noise. An examiner would reasonably ask: "Did you try to fix this?" Without v2, the answer would be "no," which undermines the credibility of the null result — the poor test accuracy could be attributed to misconfiguration rather than to genuine unpredictability.

V2 applies the standard ML engineering response to overfitting: chronological validation, early stopping, and hyperparameter selection. If v2 still fails to beat the baseline, the null result is confirmed under refinement. If v2 succeeds, the margin quantifies what proper tuning contributes.

---

## Chronological validation design

The training set (3,252 rows, 2010-02-02 to 2022-12-30) is split chronologically into:

| Partition | Rows | Date range | Purpose |
|---|---|---|---|
| Fitting set | 2,749 | 2010-02-02 to 2020-12-31 | Model fitting during grid search |
| Validation set | 503 | 2021-01-04 to 2022-12-30 | Early stopping and configuration selection |

The validation set is always posterior to the fitting set — no shuffling, no randomisation. The validation boundary (2020-12-31) is read from config, making the split reproducible and adjustable.

This design preserves temporal ordering at both levels: the validation set follows the fitting set, and both precede the test set. The model is never exposed to data from a period it will later be evaluated on.

---

## Grid search design

The grid searches over 18 combinations of three hyperparameters:

| Parameter | Values | Rationale |
|---|---|---|
| `max_depth` | {2, 3, 4} | Controls per-tree complexity. Depth 2 is the most constrained (4 leaves); depth 4 allows richer interactions (16 leaves). |
| `n_estimators` | {50, 100, 200} | Maximum number of trees before early stopping. Early stopping typically halts well before this limit. |
| `learning_rate` | {0.05, 0.1} | Controls each tree's contribution. Lower rates require more trees but reduce overfitting. |

Fixed across all combinations: `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=10`. These regularisation settings are inherited from v1.

For each combination, the model is fitted on the fitting set (2,749 rows) with `eval_set=[(X_val, y_val)]` and `early_stopping_rounds=10`. Training halts when validation logloss does not improve for 10 consecutive rounds. The number of trees actually used (`best_iteration + 1`) is recorded — this is the model self-regulating its own complexity.

The combination with the highest validation accuracy is selected. The final model is then retrained on the **full** training set (3,252 rows) using the selected hyperparameters and the early-stopped tree count.

---

## Grid search results

All 18 combinations, sorted by validation accuracy:

| Depth | n_est | LR | Trees used | Val acc | Val MCC |
|---|---|---|---|---|---|
| 2 | 50 | 0.10 | 17 | 52.7% | 0.1002 |
| 2 | 100 | 0.10 | 17 | 52.7% | 0.1002 |
| 2 | 200 | 0.10 | 17 | 52.7% | 0.1002 |
| 2 | 50 | 0.05 | 22 | 51.9% | 0.0803 |
| 2 | 100 | 0.05 | 22 | 51.9% | 0.0803 |
| 2 | 200 | 0.05 | 22 | 51.9% | 0.0803 |
| 4 | 50 | 0.05 | 10 | 51.9% | 0.0971 |
| 3 | 50 | 0.10 | 3 | 51.1% | 0.0382 |
| 3 | 50 | 0.05 | 3 | 50.7% | 0.0000 |

Three observations from the grid:

1. **Early stopping is aggressive.** The best configuration used only 17 of 50 allowed trees. Most configurations were stopped within 3–22 rounds. The model self-regulates heavily because validation loss deteriorates quickly after a few trees — the features contain very little learnable signal.

The validation MCC values are uniformly low (0.00 to 0.10), confirming that even the best-performing configuration on validation data has negligible discriminative ability. The "best" configuration is best only in relative terms — it is still close to random in absolute terms.

2. **Depth 2 wins.** The shallowest trees outperform depths 3 and 4. Deeper trees overfit faster and are stopped earlier. This confirms that the optimal model complexity for this feature set is very low.

3. **n_estimators is irrelevant.** Due to early stopping, configurations with 50, 100, and 200 maximum trees produce identical results at the same depth and learning rate. The model never reaches the upper bound.

---

## Selected configuration

- `max_depth=2`, `n_estimators=17` (early-stopped from 50), `learning_rate=0.1`
- Validation accuracy: 52.7%, validation MCC: 0.1002

---

## v1-vs-v2 comparison

| Metric | XGBoost-v1 | XGBoost-v2 |
|---|---|---|
| Hyperparameters | Fixed defaults | Grid-searched, early-stopped |
| max_depth | 3 | 2 |
| Trees | 100 (all used) | 17 (early-stopped) |
| learning_rate | 0.1 | 0.1 |
| Training accuracy | 71.7% | 56.3% |
| Test accuracy | 53.1% | 55.7% |
| Train-test gap | +18.6pp | +0.6pp |
| Test MCC | -0.037 | -0.090 |
| Predicted up % | 80.4% | 97.4% |
| Exceeds baseline | No | No |

**The overfitting gap collapsed from 18.6pp to 0.6pp.** V2's training accuracy (56.3%) is much closer to its test accuracy (55.7%), confirming that early stopping and shallower trees eliminated the noise-fitting that dominated v1.

**Test accuracy improved by 2.6pp** (53.1% → 55.7%), but still falls 1.8pp below the 57.5% majority baseline. The null result survives refinement.

**V2 converged toward a near-constant "up" predictor** (97.4% up), similar to GeoBM and the GA. As the model's capacity is constrained to match the signal level, it defaults to the prior distribution — predicting "up" on almost every day. This is the same behaviour observed across all three paradigms, reinforcing the conclusion that the features do not contain reliable directional signal.

**MCC worsened** (-0.037 → -0.090). This is because v2 predicts "down" on only 13 days (vs v1's 98), and those 13 deviations from "predict up" are mostly wrong. The few non-majority predictions actively hurt MCC.

The convergence of v2 toward near-constant "up" prediction (97.4%) mirrors the behaviour of GeoBM (100%) and the GA (96.4%). This cross-paradigm convergence is the project's most revealing pattern: as each model's effective capacity is reduced to match the available signal — whether by mathematical structure (GeoBM), representational constraint (GA), or regularisation (XGBoost-v2) — all three default to the same strategy. The majority class is the attractor in a low-signal environment.

---

## Challenging aspects

### 1. The refinement confirms the null result — this is the finding

V2's purpose was to test whether v1's poor accuracy was caused by overfitting or by feature-set limitations. The answer is clear: the overfitting gap collapsed (18.6pp → 0.6pp), but accuracy did not reach the baseline (55.7% < 57.5%). The features genuinely do not contain enough directional signal to beat naive class-frequency prediction, regardless of how carefully the model is tuned.

### 2. Early stopping as self-regulating complexity

The model used only 17 of 50 allowed trees. This means validation logloss started deteriorating after just 7 effective trees (17 minus the 10-round patience). The feature set's information content is exhausted very quickly — additional trees add noise rather than signal.

### 3. Retraining on full training set after selection

After selecting the best configuration on validation data, the final model is retrained on all 3,252 training rows. This is standard practice: the validation set was used for selection, not for final parameter estimation. The selected hyperparameters (depth 2, 17 trees, lr 0.1) are carried forward, but the tree splits are re-learned from the full training history.

---

## Scope boundaries

**This module does not replace v1.** Both variants are retained for comparison. V1 shows what fixed defaults produce. V2 shows what disciplined tuning produces. The comparison between them is the deliverable.

**Bayesian optimisation** was considered as an alternative to grid search. It was rejected because: (1) the 18-combination grid is exhaustive in under 1 second — there is no computational benefit to a more sophisticated search; (2) Bayesian optimisation adds library dependencies and implementation complexity for no marginal gain on an 18-point grid.

**Walk-forward validation** would re-run the entire grid search at multiple time points. This is a legitimate evaluation-stage extension but is disproportionate for this proof-of-concept.

---

## Rubric mapping

| Criterion | How satisfied |
|---|---|
| **Design issues discussed** | Chronological validation split justified. Grid search bounded and justified. Early stopping as self-regulating regularisation. Retraining on full data after selection. |
| **Challenging aspects** | Overfitting diagnosed in v1, corrected in v2, null result confirmed under refinement. Early stopping behaviour interpreted (17/50 trees → signal exhaustion). MCC degradation explained. |
| **Advanced knowledge** | Chronological validation for time-series tuning. Early stopping with eval_set. v1-vs-v2 comparison as methodological tightening, not model rescue. Understanding that reduced overfitting ≠ improved accuracy when signal is absent. |
| **Replication detail** | Grid search ranges, validation boundary, and early stopping rounds in config. Seed 42. All 18 results logged. Another researcher can reproduce by running `python src/models/xgboost_model/xgb_classifier_v2.py`. |

---

## How this feeds the dissertation

**Methodology chapter, "XGBoost Refinement" subsection (~300–400 words).**

> *Ready-to-lift wording (methodology):* "To address the 18.6 percentage point overfitting gap observed in XGBoost-v1, a validated variant (v2) was implemented with chronological hyperparameter selection and early stopping. The training set (3,252 rows) was split chronologically into a fitting portion (2,749 rows, 2010–2020) and a validation portion (503 rows, 2021–2022). A grid search over 18 hyperparameter combinations — max_depth in {2, 3, 4}, n_estimators in {50, 100, 200}, learning_rate in {0.05, 0.1} — was conducted with early stopping on validation logloss (patience = 10 rounds). The configuration with the highest validation accuracy (max_depth=2, 17 trees early-stopped from 50, learning_rate=0.1) was selected and retrained on the full training set before evaluation on the held-out test set. This design ensures that no test data entered the tuning process, and that the validation set is always temporally posterior to the fitting set."

**Evaluation chapter, "v1-vs-v2 Comparison" paragraph (~300 words).**

**Relationship to v1.** The v1-vs-v2 comparison is not about which model is "better." Both fail to beat the baseline. The comparison demonstrates that the null result is robust to the most common methodological objection (insufficient tuning) and reveals the ceiling imposed by the feature set's information content. This is the kind of controlled refinement that distinguishes a first-class evaluation from a first-pass observation.

> *Ready-to-lift wording (evaluation):* "XGBoost-v2 reduced the overfitting gap from 18.6pp (v1: 71.7% training, 53.1% test) to 0.6pp (v2: 56.3% training, 55.7% test), confirming that v1's poor test accuracy was partly attributable to overfitting. However, v2 still did not exceed the majority-class baseline of 57.5%, falling short by 1.8pp. This is the strongest evidence in the comparative framework for the null result: even with disciplined hyperparameter selection, chronological validation, and early stopping, the supervised learner could not extract directional signal from the 14 engineered features sufficient to outperform naive class-frequency prediction. The early stopping behaviour is itself diagnostic: the best configuration used only 17 of 50 allowed trees, indicating that the feature set's information content is exhausted after very few boosting rounds. As model capacity was constrained to match the signal level, v2 converged toward a near-constant 'up' predictor (97.4% up), the same behaviour observed in GeoBM and the GA — further evidence that the features do not contain reliable directional structure for this asset at daily frequency."

---

## Evidence produced

| Metric | XGBoost-v1 | XGBoost-v2 |
|---|---|---|
| Hyperparameters | Fixed defaults | Grid-searched, early-stopped |
| max_depth | 3 | 2 |
| Trees used | 100 | 17 |
| learning_rate | 0.1 | 0.1 |
| Fitting rows | 3,252 (all) | 2,749 (grid search) → 3,252 (retrain) |
| Validation rows | None | 503 |
| Training accuracy | 71.7% | 56.3% |
| Test accuracy | 53.1% | 55.7% |
| Train-test gap | +18.6pp | +0.6pp |
| Test MCC | -0.037 | -0.090 |
| Predicted up (test) | 403 (80.4%) | 488 (97.4%) |
| Exceeds baseline | No | No |
| Grid combinations | 0 | 18 |
| Best val accuracy | N/A | 52.7% |

Output saved to `results/predictions/xgb_v2_predictions.csv` with columns: date, target, predicted, p_up, model.
