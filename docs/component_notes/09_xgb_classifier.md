# Component Note: xgb_classifier.py

**File:** `src/models/xgboost_model/xgb_classifier.py`
**Date:** 2026-04-02

---

## Purpose

Trains an XGBoost gradient-boosted tree classifier on the 14 engineered features from the training set, generates predicted probabilities for every day in the held-out test set, and saves comparison-compatible predictions to `results/predictions/xgb_predictions.csv`.

This module is the third and final modelling paradigm in the project's comparative framework. It is the decisive test: XGBoost has non-linear modelling capacity, access to the full 14-feature engineered space, and learned feature interactions — capabilities that neither GeoBM (2 parameters, no features) nor the GA (3 threshold rules, implicit feature selection) possess. If XGBoost cannot exceed the 57.5% majority-class baseline, the conclusion is that daily SPY direction is not predictable from these features by any of the three paradigms under this evaluation protocol — itself evidence for weak-form market efficiency at daily frequency and horizon. This interpretation carries the standard scope caveat: the evidence applies to this specific asset (SPY), feature set (14 technical indicators), evaluation period (2023–2024), and prediction horizon (next-day binary direction). Generalisation beyond this scope requires additional experiments.

---

## Why XGBoost is the third paradigm

The three-model comparison evaluates fundamentally different computational approaches to the same prediction task:

| Model | Approach | Inputs | Capacity |
|---|---|---|---|
| GeoBM | Stochastic process | Raw returns (2 parameters) | Minimal — unconditional distribution only |
| GA | Evolutionary search | Engineered feature space (3 rules) | Low — bounded threshold-rule family |
| XGBoost | Supervised ensemble learning | 14 engineered features (all) | High — non-linear interactions across all features |

XGBoost completes the comparison by providing the highest-capacity paradigm. If it fails, the failure cannot be attributed to insufficient model capacity — it must be attributed to insufficient signal in the features or to fundamental unpredictability of the target.

This separation reflects genuinely different modelling philosophies: GeoBM assumes returns are IID; the GA evolves interpretable rules with no gradient information; XGBoost learns a function approximation via sequential gradient descent on an ensemble of decision trees (Chen & Guestrin, 2016, *KDD*). Each represents a distinct branch of computational intelligence.

---

## Why XGBoost is appropriate

XGBoost is the dominant supervised learning method for structured tabular classification tasks (Chen & Guestrin, 2016). It has won the majority of tabular data competitions on Kaggle and is widely used in financial ML applications (López de Prado, 2018). Its appropriateness for this project rests on three properties:

1. **It is the strongest conventional baseline for tabular data.** If XGBoost cannot extract directional signal from the 14 features, it is unlikely that any standard tabular classifier could. This makes XGBoost the right upper bound on what supervised learning can achieve here.

2. **It does not require feature scaling.** XGBoost splits are based on rank ordering, making the algorithm invariant to monotonic feature transformations. This eliminates the need for a separate scaling step and the associated risk of leaking test-set statistics through the scaler.

3. **It provides built-in feature importance.** Gain-based importance scores reveal which features contributed most to the learned decision boundaries, connecting the model's predictions back to the feature engineering stage and enabling cross-paradigm comparison with the GA's evolved feature selections.

---

## Hyperparameter choices and justification

All hyperparameters are read from `config/data_config.yaml` and fixed before training. No hyperparameter search is performed.

| Parameter | Value | Justification |
|---|---|---|
| `max_depth` | 3 | Shallow trees limit per-tree complexity, reducing overfitting risk. Depth 3 allows 2^3 = 8 leaf nodes per tree — sufficient for capturing basic interactions without memorising noise. This is the standard conservative default (Chen & Guestrin, 2016). |
| `n_estimators` | 100 | The number of boosting rounds. 100 is the standard default and provides sufficient ensemble capacity for 3,252 training rows. More trees risk overfitting; fewer trees risk underfitting. |
| `learning_rate` | 0.1 | Controls the contribution of each tree. Lower values require more trees but reduce overfitting. 0.1 is the standard default that balances convergence speed with regularisation. |
| `subsample` | 0.8 | Row sampling per tree. Using 80% of training rows per tree introduces stochastic regularisation that reduces overfitting. |
| `colsample_bytree` | 0.8 | Feature sampling per tree. Using 80% of features per tree forces the ensemble to be robust to feature subsets and reduces the risk of over-relying on a single feature. |
| `reg_alpha` | 0 (default) | L1 regularisation on leaf weights. The default of zero means no L1 penalty is applied. Combined with `reg_lambda`, this controls the complexity penalty on individual leaves. |
| `reg_lambda` | 1 (default) | L2 regularisation on leaf weights. The default of 1 applies a moderate L2 penalty. This is the primary leaf-level regularisation mechanism and was left at its default because explicit tuning was not performed. |

**Why fixed defaults rather than tuning.** Hyperparameter tuning (e.g., grid search with chronological cross-validation) would strengthen the model's test performance and methodological defensibility. However, for this proof-of-concept baseline, fixed conservative defaults are sufficient to answer the comparison question: can XGBoost, even with untuned parameters, beat the majority-class baseline? If it cannot with sensible defaults, the conclusion about feature-set limitations is stronger, not weaker. The overfitting risk is managed through subsample, colsample_bytree, and shallow trees rather than through a tuning procedure. The default regularisation parameters (reg_alpha=0, reg_lambda=1) provide moderate L2 leaf-weight penalisation without L1 sparsity. These were not tuned — their defaults are part of the fixed-configuration design.

---

## How predictions are generated

After training, `predict_proba` generates the predicted probability P(up) for each test row. The binary prediction is:

    predicted = 1 (up) if P(up) >= 0.5, else 0 (down)

Unlike GeoBM (where P(up) is a single constant, 0.5166) and the GA (where P(up) is a vote fraction with only 3 discrete values), XGBoost produces a continuous probability distribution across test rows. In the real run, P(up) ranged from 0.2633 to 0.8048 with median 0.5634, confirming that the model produces genuinely varying predictions — it is not a constant-class predictor.

---

## Feature importance

XGBoost's built-in gain-based feature importance measures the average improvement in the loss function contributed by each feature across all splits in all trees. The real-run results show a remarkably flat importance distribution:

| Rank | Feature | Importance |
|---|---|---|
| 1 | volume_sma20_ratio | 0.0781 |
| 2 | mom_5 | 0.0756 |
| 3 | sma5_sma20_ratio | 0.0737 |
| 4 | mom_10 | 0.0737 |
| 5 | close_sma20_ratio | 0.0736 |
| 6 | rsi_14 | 0.0730 |
| 7 | volume_chg_1 | 0.0718 |
| 8 | close_sma5_ratio | 0.0706 |
| 9 | vol_20 | 0.0706 |
| 10 | vol_5 | 0.0705 |
| 11 | ret_lag_1 | 0.0704 |
| 12 | ret_lag_2 | 0.0701 |
| 13 | vol_10 | 0.0656 |
| 14 | ret_lag_5 | 0.0626 |

The importance values range from 0.0626 to 0.0781 — a spread of only 0.0155 across 14 features. No single feature dominates the model. This near-uniform distribution is itself diagnostic: the model could not find any feature or feature subset with strong discriminative power, so it distributed its splitting budget approximately evenly. This is consistent with the hypothesis that no feature contains reliable directional signal.

The GA converged to MA-ratio features (close_sma20_ratio, close_sma5_ratio). XGBoost's top features include volume_sma20_ratio, momentum, and the same MA ratios — partial overlap, but no strong convergence. The cross-paradigm feature selection is weakly consistent but not conclusive.

Caveat: colsample_bytree effect on importance. The colsample_bytree=0.8 setting means each tree sees only 80% of features, which mechanically flattens gain-based importance — features excluded from a tree cannot accumulate gain in that tree, and the missing gain is redistributed across whichever features happen to be sampled. The near-uniform importance distribution is therefore partially an artefact of the column-sampling regularisation, not solely evidence of uniform signal strength. A model with colsample_bytree=1.0 would produce a less flattened distribution, though likely still without a dominant feature given the low overall signal. This caveat does not change the conclusion — no feature contains strong directional signal — but it qualifies the degree to which the flat distribution can be attributed to the features versus the training procedure.

---

## Rejected alternatives

### Hyperparameter tuning with chronological validation

A chronological validation split (e.g., last 2 years of training for validation, earlier years for fitting) could support grid search or Bayesian optimisation of hyperparameters. This was considered and deferred. The fixed-default approach answers a cleaner comparison question: "can XGBoost with sensible defaults beat the baseline?" If yes, tuning would quantify how much more can be gained. Since the answer is no, tuning is unlikely to reverse the conclusion — the 18.6pp overfitting gap suggests the model is already fitting noise, and tuning would primarily reduce that gap rather than improve test accuracy above the baseline.

### Early stopping on a validation set

Early stopping would halt training when validation loss stops improving, reducing overfitting. This was considered and deferred for the same reason: the fixed-default approach provides a cleaner first-pass comparison. Early stopping is a legitimate refinement for a second iteration but is not needed to answer the primary comparison question.

### Feature selection before training

Pre-selecting a subset of the 14 features (e.g., via mutual information or univariate tests) could reduce dimensionality and overfitting. This was rejected because: (1) XGBoost performs implicit feature selection via its splitting mechanism; (2) colsample_bytree=0.8 already introduces feature-level regularisation; (3) manual feature pre-selection would impose human bias on the search, inconsistent with the GA comparison where implicit feature selection was the design choice.

### Class-weight balancing or oversampling

XGBoost supports `scale_pos_weight` for class-weight adjustment, and preprocessing techniques such as SMOTE could be applied to oversample the minority class. These were considered and rejected. The class imbalance (55.0% up in training, 57.5% up in test) is mild — it does not approach the extreme ratios (e.g., 95/5) where balancing techniques provide substantial benefit. More importantly, applying class weights would optimise for balanced accuracy rather than plain accuracy, creating an inconsistency with the evaluation metric and the majority-class baseline definition. The GA uses plain accuracy as its fitness function for the same reason. If class-weight balancing were applied and XGBoost then exceeded the baseline in balanced-accuracy terms, the comparison with GeoBM and the GA (which were not rebalanced) would be confounded. The unbalanced configuration keeps the comparison clean.

---

## Challenging aspects and how they were resolved

### 1. Overfitting — the central diagnostic

XGBoost achieved 71.7% training accuracy but only 53.1% test accuracy — a gap of 18.6 percentage points. This is the most important result in the module and the most important diagnostic in the entire three-model comparison.

The overfitting is genuine, not an artefact. The model learned patterns in the training data (2010–2022) that do not generalise to the test period (2023–2024). With 100 trees of depth 3, the model has sufficient capacity to memorise statistical regularities in 3,252 training rows — particularly given that daily returns have low signal-to-noise ratios and the features are themselves noisy summaries of price and volume dynamics.

This result is surfaced rather than hidden. It is not a failure of the implementation — it is a finding about the feature set and the prediction task. The conservative hyperparameters (depth 3, subsample 0.8, colsample 0.8) were specifically chosen to limit overfitting, yet the gap is still 18.6pp. This suggests that the overfitting is driven by the low signal-to-noise ratio of the features rather than by model misconfiguration.

### 2. Below-baseline test accuracy

XGBoost's 53.1% test accuracy is below the 57.5% majority baseline, below GeoBM (57.5%), and below the GA (56.7%). The model is the worst performer of the three paradigms on the test set, despite being the highest-capacity model.

This is counterintuitive but interpretable: XGBoost's greater capacity allows it to fit more noise in the training data, which actively hurts test performance. The simpler models (GeoBM and GA) are constrained enough that they default to near-majority-class prediction, which happens to be a reasonable strategy in a period with 57.5% up-days. XGBoost's learned decision boundaries deviate from the majority class on ~20% of test days — and those deviations reduce rather than improve accuracy.

### 3. Near-uniform feature importance

No feature stands out. The importance spread (0.0626 to 0.0781) is narrow. This means the model could not identify any strong directional signal in any individual feature. If any feature contained reliable predictive information, XGBoost would have concentrated its splits on that feature, producing a peaked importance distribution. The flat distribution is a negative finding about the feature set's directional content.

---

## Scope boundaries

**Hyperparameter tuning** is deferred. The fixed-default approach provides a clean first-pass comparison. A tuned variant (with chronological validation) is a legitimate second-pass refinement documented as future work.

**Early stopping** is deferred for the same reason. It would reduce the training-test gap but does not change the fundamental conclusion about feature-set limitations.

**SHAP values** were considered for richer feature importance analysis but deferred. Built-in gain importance is sufficient for the first-pass comparison. SHAP analysis would be valuable in a second iteration to understand per-prediction feature contributions.

**Walk-forward validation** is an evaluation-module concern. This module trains once and evaluates once under the project's single-holdout protocol.

---

## Rubric mapping (COM3001 assessor language)

| Assessor criterion | How this component satisfies it |
|---|---|
| **Requirements analysed** | The comparative framework requires the highest-capacity paradigm to test whether supervised learning can extract signal the other two missed. XGBoost fills this role. |
| **Design issues discussed** | Hyperparameter choices justified with literature. Overfitting diagnosed and interpreted. Feature importance analysed. Fixed defaults vs tuning trade-off documented. Three rejected alternatives with rationale. |
| **Challenging aspects** | The 18.6pp overfitting gap is the central finding. Below-baseline test accuracy is interpreted structurally (capacity enables noise-fitting). Near-uniform feature importance is interpreted as a negative finding about directional content. |
| **Replication detail** | All hyperparameters in config. Seed 42 for reproducibility. Feature columns verified before training. Full diagnostics logged. Another researcher can reproduce by running `python src/models/xgboost_model/xgb_classifier.py` from the repo root. |
| **Advanced knowledge demonstrated** | Gradient boosting as sequential function approximation (Chen & Guestrin, 2016). Overfitting diagnosis in ensemble methods. Feature importance analysis. Understanding that below-baseline performance with a strong learner is evidence about the features, not just the model. |

---

## How this feeds the dissertation

**Methodology chapter, "XGBoost Classifier" subsection (~400–500 words).**

> *Ready-to-lift wording (methodology):* "The third paradigm in the comparative framework is an XGBoost gradient-boosted tree classifier trained on all 14 engineered features. XGBoost constructs an ensemble of shallow decision trees by sequential gradient descent on the binary cross-entropy loss, where each tree corrects the residual errors of the previous ensemble (Chen & Guestrin, 2016). The model was configured with conservative hyperparameters: maximum tree depth of 3, 100 boosting rounds, learning rate of 0.1, row subsampling at 80%, and feature subsampling at 80% per tree. These values were fixed before training and read from the project configuration file — no hyperparameter search was performed. The conservative settings were chosen to limit overfitting on 3,252 training rows while providing sufficient ensemble capacity for the 14-dimensional feature space. The random seed (42) was applied for full reproducibility. The model was trained on the training set only; the test set was used exactly once after training to generate predicted probabilities via `predict_proba`, which were thresholded at 0.5 for binary directional predictions."

**Evaluation chapter, "XGBoost Results" paragraph (~300–400 words).**

> *Ready-to-lift wording (evaluation):* "XGBoost achieved 71.7% accuracy on the training set but only 53.1% on the 501-day test set — a gap of 18.6 percentage points indicating significant overfitting. The test accuracy falls below the majority-class baseline (57.5%), below the GeoBM baseline (57.5%), and below the GA baseline (56.7%), making XGBoost the worst-performing model in the held-out comparison despite having the greatest modelling capacity. This result is the central finding of the three-paradigm comparison. The model learned statistical patterns in the 2010–2022 training period that did not generalise to the 2023–2024 test period, despite conservative hyperparameters (depth 3, subsample 0.8, colsample 0.8) specifically chosen to limit overfitting. The feature importance distribution was near-uniform: no single feature contributed disproportionately to the model's decision boundaries, with importance values ranging from 0.0626 (ret_lag_5) to 0.0781 (volume_sma20_ratio). This flat distribution indicates that the model could not identify any strong directional signal in the engineered feature set. Taken together, the three-paradigm results tell a coherent story: the stochastic baseline collapsed to a constant predictor (GeoBM, 57.5%); the evolutionary search found only near-majority-class threshold rules (GA, 56.7%); and the supervised learner overfit to training noise without extracting generalisable signal (XGBoost, 53.1%). None of the three paradigms exceeded the naive majority-class baseline of 57.5%, providing evidence that daily SPY direction is not reliably predictable from these 14 technical features under this evaluation protocol. This null result is itself informative: it is consistent with the weak-form efficient market hypothesis, which predicts that historical price and volume information — the basis of all 14 features — should not systematically predict future returns (Fama, 1970)."

---

## Evidence produced

This component trained an XGBoost classifier on 3,252 training observations with 14 features and evaluated on 501 test observations. The exact outputs:

| Metric | Value |
|---|---|
| Training rows | 3,252 |
| Test rows | 501 |
| Features | 14 (all engineered features) |
| Hyperparameters | max_depth=3, n_estimators=100, lr=0.1, subsample=0.8, colsample=0.8 |
| Tuning method | None (fixed defaults from config) |
| Training accuracy | 71.7% |
| Test accuracy | 53.1% (266 / 501) |
| Majority-class baseline | 57.5% (288 / 501) |
| GeoBM baseline | 57.5% (288 / 501) |
| GA baseline | 56.7% (284 / 501) |
| Exceeds baseline | No — worst of all three models |
| Train-test gap | +18.6pp (significant overfitting) |
| Predicted up (test) | 403 (80.4%) |
| Predicted down (test) | 98 (19.6%) |
| Constant predictor | No — produces genuinely varying predictions |
| P(up) range | 0.2633 to 0.8048 (median 0.5634) |
| Top feature | volume_sma20_ratio (0.0781) |
| Feature importance spread | 0.0626 to 0.0781 (near-uniform) |
| Random seed | 42 |

Output saved to `results/predictions/xgb_predictions.csv` with columns: date, target, predicted, p_up, model.

### Three-paradigm comparison summary

| Model | Test accuracy | vs baseline | Prediction pattern | Key diagnostic |
|---|---|---|---|---|
| GeoBM | 57.5% | Equals baseline | Constant "up" (100%) | P(up) = 0.5166, positive drift |
| GA | 56.7% | -0.8pp | Near-constant "up" (96.4%) | 0.9pp above train majority |
| XGBoost | 53.1% | -4.4pp | Up-biased (80.4%) | 18.6pp overfitting gap |
| Majority class | 57.5% | — | Constant "up" (100%) | Zero-skill threshold |

No model exceeded the naive majority-class baseline on the held-out test set.
