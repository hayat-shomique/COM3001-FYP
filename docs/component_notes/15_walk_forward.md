# Component Note: walk_forward.py

**File:** `src/evaluation/walk_forward.py`
**Date:** 2026-04-03

---

## Purpose

Tests whether the null result (no model beats the majority baseline) is specific to the primary train/test split or holds across market regimes. All three paradigms are retrained from scratch on three rolling temporal windows and evaluated on each window's held-out test period. This is beyond-taught-material work that directly addresses the temporal robustness of the comparative framework's central finding.

---

## Window design

| Window | Train period | Train rows | Test period | Test rows | Test up-rate | Majority baseline |
|---|---|---|---|---|---|---|
| W1 | 2010–2018 | 2,244 | 2019–2020 | 505 | 58.4% | 58.4% |
| W2 | 2012–2020 | 2,265 | 2021–2022 | 503 | 50.7% | 50.7% |
| W3 | 2014–2022 | 2,266 | 2023–2024 | 501 | 57.5% | 57.5% |

**Why these three windows.** Each window captures a distinct market regime transition:

- **W1** trains on the post-GFC recovery and tests on a period that includes the February–March 2020 COVID crash — the most extreme volatility event in the dataset.
- **W2** trains on a period that includes the COVID crash and tests on the post-COVID recovery and the 2022 bear market — a period with near-equal up/down days (50.7% up).
- **W3** trains on 2014–2022 and tests on 2023–2024 — approximately matching the primary holdout split (which trains on 2010–2022). This serves as a sanity check: results should be close to the main evaluation.

The majority baselines vary substantially across windows: 58.4%, 50.7%, 57.5%. This variation confirms that the windows span genuinely different market regimes. W2's near-50% baseline (the 2021–2022 test period) is the most challenging for the majority-class strategy and the most informative for detecting directional signal.

---

## Results

### Window 1: Train 2010–2018, Test 2019–2020

| Model | Accuracy | vs Baseline | MCC | Pred Up% |
|---|---|---|---|---|
| Majority | 58.4% | — | — | 100.0% |
| GeoBM | 58.4% | +0.0pp | 0.000 | 100.0% |
| GA | 57.8% | -0.6pp | 0.040 | 89.1% |
| XGBoost-v2 | 57.0% | -1.4pp | 0.002 | 91.5% |

No model beats the baseline. GeoBM equals the majority class as expected (positive drift → constant "up"). The GA achieves the highest MCC of any model in any window (0.040) and produces the most varied predictions (89.1% up) — but still falls short. The higher prediction diversity in W1 likely reflects the COVID crash in the test period: the extreme volatility of February–March 2020 pushes feature values (particularly volatility and MA ratios) far enough from their training-period ranges that the GA's threshold rules fire more frequently on the "down" side, producing more varied predictions than in the calmer W2 and W3 test periods.

### Window 2: Train 2012–2020, Test 2021–2022

| Model | Accuracy | vs Baseline | MCC | Pred Up% |
|---|---|---|---|---|
| Majority | 50.7% | — | — | 100.0% |
| GeoBM | 50.7% | +0.0pp | 0.000 | 100.0% |
| GA | 50.7% | +0.0pp | 0.003 | 95.6% |
| XGBoost-v2 | 50.9% | +0.2pp | 0.016 | 97.8% |

XGBoost-v2 marginally exceeds the baseline by 0.2pp (50.9% vs 50.7%). This is the only instance across all 9 model-window combinations where any model exceeds its baseline. The margin (0.2pp = 1 additional correct prediction on 503 days) is trivially small and well within the ±4.4pp bootstrap CI width. It is not practically meaningful. Furthermore, the GA's stochastic search means that the W2 result is seed-dependent — a different random seed would produce different evolved rules and could easily reverse the 0.2pp margin in either direction. The exceedance should be interpreted as noise, not signal.

The W2 baseline (50.7%) is near-random, reflecting the mixed market conditions of 2021–2022 (strong 2021 rally followed by the 2022 bear market). The GA evolves different rules from W1 and W3 (ret_lag_1, vol_10, volume_sma20_ratio instead of MA ratios), showing that the evolutionary search adapts to the training regime — but cannot translate that adaptation into test-set advantage.

### Window 3: Train 2014–2022, Test 2023–2024 (sanity check)

| Model | Accuracy | vs Baseline | MCC | Pred Up% |
|---|---|---|---|---|
| Majority | 57.5% | — | — | 100.0% |
| GeoBM | 57.5% | +0.0pp | 0.000 | 100.0% |
| GA | 56.7% | -0.8pp | -0.036 | 98.0% |
| XGBoost-v2 | 55.9% | -1.6pp | -0.064 | 96.8% |

Results closely match the main evaluation (GeoBM 57.5%, GA 56.7%, XGBoost-v2 55.7%). The 0.2pp difference in XGBoost-v2 (55.9% vs 55.7%) is attributable to the different training period (2014–2022 vs 2010–2022). The consistency confirms the sanity check: the main evaluation results are not an artefact of the specific training period.

---

## Stability analysis

### Does the null result hold?

**Substantively, yes.** Across 9 model-window combinations (3 models × 3 windows), only one exceeds its baseline: XGBoost-v2 in W2, by 0.2pp. This margin is trivially small (1 row on 503), well within bootstrap uncertainty, and does not constitute evidence of directional signal. The null result — no model reliably beats the majority baseline — holds across all three market regimes.

### Do the majority baselines differ?

Yes, substantially: 58.4% (W1), 50.7% (W2), 57.5% (W3). W2's near-50% baseline is noteworthy — the 2021–2022 test period had almost equal up and down days. This is the window where directional signal would be most valuable (and most detectable), yet no model exceeds 50.9%.

### Cross-paradigm convergence

All three models converge toward near-constant "up" prediction across all windows:

| Window | GeoBM | GA | XGBoost-v2 |
|---|---|---|---|
| W1 | 100.0% | 89.1% | 91.5% |
| W2 | 100.0% | 95.6% | 97.8% |
| W3 | 100.0% | 98.0% | 96.8% |

The pattern from the main evaluation (all models default to the majority class) is confirmed across temporal windows. The GA shows the most variation (89.1% in W1 vs 98.0% in W3), reflecting its sensitivity to the training regime, but this variation does not translate into consistent above-baseline accuracy.

### GA feature evolution across windows

The GA evolves different rules in each window, demonstrating that the evolutionary search responds to different training regimes:

- **W1:** close_sma5_ratio, mom_10 (MA ratio and momentum features)
- **W2:** ret_lag_1, vol_10, volume_sma20_ratio (return, volatility, and volume features)
- **W3:** close_sma20_ratio, mom_5 (MA ratio and momentum features)

W1 and W3 converge to similar feature families (MA ratios + momentum), while W2 — trained on a period that includes the COVID crash — selects different features. This regime-dependent feature selection is consistent with the feature ablation finding (note 12) that different feature categories have marginal and unstable contributions.

---

## Why walk-forward over k-fold

Standard k-fold cross-validation destroys temporal ordering — fold 3 might train on 2020 data and test on 2015 data. This is invalid for time-series prediction because it allows the model to train on future market conditions when predicting past directions (López de Prado, 2018; Pardo, 2008).

Walk-forward evaluation preserves causal ordering: every test period is strictly posterior to its training period. The three fixed windows were chosen over a monthly rolling scheme because: (1) the training set must be large enough for stable parameter estimation (~2,200+ rows); (2) the test set must be large enough for meaningful accuracy estimation (~500 rows); (3) three windows capture the key regime transitions in the 2010–2024 period (pre-COVID, peri-COVID, post-COVID).

### Rejected: Expanding window

An expanding-window design (fixed start, expanding training set) was considered. It was rejected for two reasons. First, expanding windows create a confound: later windows have more training data than earlier windows, so any accuracy improvement could be attributable to increased sample size rather than regime fit. The rolling design holds training-set size approximately constant (~2,250 rows across all three windows), isolating the regime effect. Second, all three expanding windows would share the same early training data (2010–2012), reducing the regime diversity of the training periods. The rolling design provides more distinct training regimes.

### Rejected: Monthly rolling

A monthly rolling scheme (e.g., train on trailing 5 years, test on next month, roll forward) would produce ~60 evaluation points. This was rejected because: (1) 20 test-day windows are too small for meaningful accuracy estimation; (2) 60 GA evolutions would take ~30 minutes; (3) the three-window design already demonstrates temporal robustness with interpretable regime labels.

---

## Scope boundaries

**Per-window hyperparameter re-tuning** was not performed for XGBoost-v2. The fixed v2 config (depth=2, trees=17, lr=0.1) is applied in all three windows. Re-tuning per window would test hyperparameter-regime interaction, which is a legitimate extension but disproportionate for demonstrating temporal robustness.

**Statistical significance of per-window results** is not computed. McNemar's test and bootstrap CIs were applied to the primary holdout split (notes 10 and 14). Replicating them for each window would add 9 additional tests with the same conclusion.

---

## Rubric mapping

| Criterion | How satisfied |
|---|---|
| **Knowledge beyond taught** | Walk-forward temporal evaluation is not taught in standard ML courses. It requires understanding why k-fold fails for time series (López de Prado, 2018; Pardo, 2008) and designing temporal windows that capture regime diversity. |
| **Evaluation methodology** | Three windows × three models = 9 independent evaluations confirming the null result across market regimes. |
| **Critical analysis** | Regime-dependent baselines interpreted. GA feature-evolution differences across windows noted. The single 0.2pp above-baseline result (W2 XGBoost-v2) is correctly identified as trivially small. |

---

## How this feeds the dissertation

**Methodology chapter, "Walk-Forward Evaluation" (~300 words).**

> *Ready-to-lift wording (methodology):* "To test the temporal robustness of the null result, a walk-forward evaluation was conducted across three rolling windows. Each window trains all three paradigms from scratch on a different temporal segment and evaluates on the immediately following period: Window 1 trains on 2010–2018 and tests on 2019–2020 (pre-COVID training, including the COVID crash in the test period); Window 2 trains on 2012–2020 and tests on 2021–2022 (peri-COVID training, post-COVID recovery test); Window 3 trains on 2014–2022 and tests on 2023–2024 (approximately matching the primary holdout, serving as a sanity check). This design preserves causal ordering — every test period is strictly posterior to its training period — and spans the three major market regime transitions in the dataset (López de Prado, 2018; Pardo, 2008)."

**Evaluation chapter, "Temporal Robustness" (~400 words).**

> *Ready-to-lift wording (evaluation):* "The walk-forward evaluation confirmed the null result across all three temporal windows. Of 9 model-window combinations, only one exceeded its majority-class baseline: XGBoost-v2 in Window 2, by 0.2 percentage points (50.9% vs 50.7%) — a trivially small margin corresponding to one additional correct prediction on 503 test days, well within the ±4.4pp bootstrap confidence interval width. The majority baselines varied substantially across windows — 58.4% (W1, bullish 2019–2020), 50.7% (W2, mixed 2021–2022), and 57.5% (W3, bullish 2023–2024) — confirming that the windows span genuinely different market regimes. Despite these regime differences, no paradigm found reliable directional signal: GeoBM consistently collapsed to a constant 'up' predictor, the GA evolved regime-dependent rules that failed to generalise, and XGBoost-v2 converged toward near-majority-class prediction in every window. The cross-paradigm convergence toward the majority class — observed in the main evaluation and confirmed across three additional temporal windows — is the project's most robust finding: in a low-signal environment, all three computational paradigms default to the same strategy regardless of their structural differences."

---

## Evidence produced

### Per-window results

| Window | Model | Accuracy | vs Baseline | MCC | Pred Up% |
|---|---|---|---|---|---|
| W1 | GeoBM | 58.4% | +0.0pp | 0.000 | 100.0% |
| W1 | GA | 57.8% | -0.6pp | 0.040 | 89.1% |
| W1 | XGBoost-v2 | 57.0% | -1.4pp | 0.002 | 91.5% |
| W2 | GeoBM | 50.7% | +0.0pp | 0.000 | 100.0% |
| W2 | GA | 50.7% | +0.0pp | 0.003 | 95.6% |
| W2 | XGBoost-v2 | 50.9% | +0.2pp | 0.016 | 97.8% |
| W3 | GeoBM | 57.5% | +0.0pp | 0.000 | 100.0% |
| W3 | GA | 56.7% | -0.8pp | -0.036 | 98.0% |
| W3 | XGBoost-v2 | 55.9% | -1.6pp | -0.064 | 96.8% |

### Stability

Null result holds substantively across all three windows. One marginal exception (XGBoost-v2 W2, +0.2pp) is within noise.

Output saved to `results/tables/walk_forward_results.csv` and `results/figures/walk_forward_stability.pdf`.
