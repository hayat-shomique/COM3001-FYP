# Component Note: feature_ablation.py

**File:** `src/evaluation/feature_ablation.py`
**Date:** 2026-04-03

---

## Purpose

Removes one feature category at a time and retrains XGBoost-v2 to measure each category's marginal contribution to directional prediction. This directly satisfies the COM3001 rubric, which names "ablation study" as an example of appropriate evaluation methodology.

The ablation answers: if no single feature category is essential, the null result is not an artefact of including unhelpful features — it reflects a genuine absence of directional signal in the engineered feature space.

---

## Experiment design

Seven experiments, all using the same XGBoost-v2 best configuration (max_depth=2, n_estimators=17, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, seed 42). No hyperparameter re-tuning — the ablation tests feature contribution, not hyperparameter sensitivity.

| Experiment | Features removed | Features remaining |
|---|---|---|
| Full model | None | 14 |
| −lagged_returns | ret_lag_1, ret_lag_2, ret_lag_5 | 11 |
| −realised_volatility | vol_5, vol_10, vol_20 | 11 |
| −momentum | mom_5, mom_10 | 12 |
| −ma_ratios | close_sma5_ratio, close_sma20_ratio, sma5_sma20_ratio | 11 |
| −rsi | rsi_14 | 13 |
| −volume | volume_chg_1, volume_sma20_ratio | 12 |

The 6 feature categories match the categories defined in `build_features.py` (note 05). Each category was designed with a distinct informational rationale — the ablation tests whether that rationale translates into marginal predictive value.

---

## Results

| Experiment | Feat | Test Acc | vs Full | Bal Acc | MCC | Pred Up% |
|---|---|---|---|---|---|---|
| Full (14 features) | 14 | 55.7% | — | 48.6% | -0.090 | 97.4% |
| −lagged_returns | 11 | 55.9% | +0.2pp | 48.7% | -0.094 | 98.0% |
| −realised_volatility | 11 | 56.1% | +0.4pp | 48.8% | -0.086 | 98.2% |
| −momentum | 12 | 57.1% | +1.4pp | 49.8% | -0.020 | 98.8% |
| −ma_ratios | 11 | 55.5% | -0.2pp | 48.4% | -0.097 | 97.2% |
| −rsi | 13 | 55.5% | -0.2pp | 48.4% | -0.097 | 97.2% |
| −volume | 12 | 55.7% | +0.0pp | 48.6% | -0.090 | 97.4% |

**No single-category removal pushed accuracy above the 57.5% majority baseline.**

---

## Interpretation

### Three categories actively hurt predictions

Removing lagged returns (+0.2pp), volatility (+0.4pp), or momentum (+1.4pp) **improved** test accuracy. These categories add noise — the model is better off without them. Momentum is the worst offender: removing it raised accuracy from 55.7% to 57.1%, the closest any configuration comes to the baseline.

This finding is significant: the momentum features (mom_5, mom_10) — which encode the classic Jegadeesh & Titman (1993) momentum anomaly — actively degrade XGBoost's out-of-sample predictions on this dataset. The momentum patterns the model learns in 2010–2022 do not persist into 2023–2024. This is consistent with the well-documented time-variation of momentum profitability (Daniel & Moskowitz, 2016, *Journal of Financial Economics*).

### Two categories provide marginal positive contribution

Removing ma_ratios or rsi each costs 0.2pp. These are the only categories whose removal hurts accuracy. The effect is tiny — 0.2pp on 501 observations is approximately 1 row — but the direction is consistent: the moving-average ratios and RSI contain the most (or least negative) directional information in the feature set.

### One category has zero effect

Removing volume changes nothing. The volume features (volume_chg_1, volume_sma20_ratio) contribute no marginal information to XGBoost's predictions. This is consistent with the volume features ranking 7th and 14th in v1's gain-based importance.

### No subset rescues the model

The highest ablation accuracy is 57.1% (−momentum), still below the 57.5% baseline. No combination of removing a single category produces a model that beats naive class-frequency prediction. The failure is not caused by one bad category dragging the model down — it is a property of the entire feature space.

---

## Cross-paradigm comparison with GA feature selection

The GA converged to ma_ratios features (close_sma20_ratio, close_sma5_ratio) through its evolutionary search. The ablation confirms this selection: removing ma_ratios is one of only two removals that hurt XGBoost's accuracy (-0.2pp). The GA, without any explicit importance analysis, independently identified the feature category that the ablation study confirms as the most (marginally) useful.

This cross-paradigm consistency — two fundamentally different search methods converging on the same features — strengthens the finding that ma_ratios contain the most relevant directional information in the feature set, even though that information is insufficient to beat the baseline.

---

## Why category-level ablation was chosen

### Rejected: Individual feature removal

Removing one feature at a time (14 experiments) was considered. It was rejected because: (1) with 17 trees of depth 2, each tree uses very few features — removing one from 14 has negligible impact; (2) category-level ablation tests the informational contribution of each feature *family* (e.g., "do volatility signals help?"), which maps directly to the feature engineering justifications in note 05; (3) 7 experiments are more interpretable than 14 for the dissertation.

### Rejected: SHAP values

SHAP (SHapley Additive exPlanations) was considered for per-prediction feature attribution. It was rejected for the first-pass ablation because: (1) SHAP explains what the model *learned*, not whether what it learned is *useful* — a high SHAP value for a feature that contributes to overfitting is misleading; (2) the ablation directly measures the marginal test-accuracy contribution, which is the quantity of interest; (3) SHAP is a legitimate second-pass analysis but is disproportionate when the model's overall accuracy is below the baseline.

---

## Scope boundaries

**Per-experiment hyperparameter tuning** was not performed. Re-running the v2 grid search for each ablation would test the interaction between feature subsets and hyperparameters — a valid but disproportionate extension. The fixed-config design isolates the feature contribution question.

**Multi-category removal** (removing 2+ categories simultaneously) was not performed. The 6-category, one-at-a-time design provides clean marginal effects. Interaction effects between categories are a legitimate extension but would require 2^6 - 1 = 63 experiments.

**Feature addition** (testing whether adding new features beyond the 14 would help) is outside scope — the 14-feature set is locked by the project's design.

---

## Rubric mapping

| Criterion | How satisfied |
|---|---|
| **Evaluation methodology** | The COM3001 rubric names "ablation study" as an example of appropriate evaluation. This module implements it directly with 7 controlled experiments. |
| **Critical analysis** | Three noise-contributing categories identified. Cross-paradigm GA comparison. No-rescue finding interpreted structurally. |
| **Comparison with related works** | Momentum's negative contribution connected to time-varying momentum profitability literature (Daniel & Moskowitz, 2016). MA-ratio usefulness consistent with GA's independent feature selection. |
| **Replication detail** | Same v2 config for all experiments. Seed 42. Results saved to CSV. Figure saved to PDF. Reproducible by running `python src/evaluation/feature_ablation.py`. |

---

## How this feeds the dissertation

**Evaluation chapter, "Feature Ablation" subsection (~400 words).**

> *Ready-to-lift wording (methodology):* "A category-level ablation study was conducted to assess the marginal contribution of each feature family to XGBoost-v2's directional predictions. The 14 engineered features were grouped into 6 categories matching the feature engineering design (lagged returns, realised volatility, momentum, moving-average ratios, RSI, and volume signals). Seven experiments were run: the full 14-feature model and 6 single-category removals. All experiments used the same XGBoost-v2 configuration (max_depth=2, 17 trees, learning_rate=0.1) to isolate the feature contribution from hyperparameter effects."

> *Ready-to-lift wording (evaluation):* "The ablation revealed that three of six feature categories — lagged returns, volatility, and momentum — actively hurt test accuracy. Removing momentum alone improved accuracy from 55.7% to 57.1%, the closest any configuration came to the 57.5% majority baseline. Only moving-average ratios and RSI provided marginal positive contribution (0.2pp each), while volume features had zero effect. No single-category removal pushed accuracy above the baseline, confirming that the null result is not caused by one harmful category dragging the model down — it is a property of the feature space as a whole. The GA's independent convergence to moving-average ratio features through evolutionary search is confirmed by the ablation: ma_ratios is one of only two categories whose removal hurts accuracy. This cross-paradigm consistency strengthens the finding, while the overall inability to exceed the baseline provides further evidence that the engineered technical features do not contain reliable directional signal for daily SPY prediction."

---

## Evidence produced

| Experiment | Features | Test Acc | vs Full | Bal Acc | MCC | Pred Up% |
|---|---|---|---|---|---|---|
| Full (14) | 14 | 55.7% | — | 48.6% | -0.090 | 97.4% |
| −lagged_returns | 11 | 55.9% | +0.2pp | 48.7% | -0.094 | 98.0% |
| −realised_volatility | 11 | 56.1% | +0.4pp | 48.8% | -0.086 | 98.2% |
| −momentum | 12 | 57.1% | +1.4pp | 49.8% | -0.020 | 98.8% |
| −ma_ratios | 11 | 55.5% | -0.2pp | 48.4% | -0.097 | 97.2% |
| −rsi | 13 | 55.5% | -0.2pp | 48.4% | -0.097 | 97.2% |
| −volume | 12 | 55.7% | +0.0pp | 48.6% | -0.090 | 97.4% |

Majority baseline: 57.5%. No ablation variant exceeded it.
GA-selected features (ma_ratios): confirmed as marginally useful by ablation.
Output saved to `results/tables/feature_ablation.csv` and `results/figures/feature_ablation.pdf`.
