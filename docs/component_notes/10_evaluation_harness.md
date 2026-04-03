# Component Note: evaluate.py

**File:** `src/evaluation/evaluate.py`
**Date:** 2026-04-02

---

## Purpose

Loads prediction files from all three paradigms (GeoBM, GA, XGBoost), computes a standardised set of classification metrics for each, performs McNemar's pairwise statistical test for accuracy differences, and saves a comparison summary to `results/tables/model_comparison.csv`.

This module operationalises the comparative evaluation layer of the project. The three modelling modules produce predictions; this module determines what those predictions mean. Every metric is computed from the same (target, predicted) pairs under identical conditions — the harness introduces no new modelling decisions, no new data, and no post-hoc adjustments. It is the single point where the project's central question is answered: did any paradigm exceed the naive majority-class baseline?

---

## Why a common evaluation harness is a separate module

Each model module generates predictions and logs its own accuracy. A separate evaluation module is needed for three reasons:

1. **Cross-model comparability.** When each model computes its own metrics, there is no guarantee the calculations are identical. A common harness applies the same metric functions to all three prediction files, eliminating any possibility of metric-computation differences across models.

2. **Statistical comparison.** McNemar's test for pairwise accuracy differences requires access to paired predictions from two models simultaneously. This cannot be performed inside any single model module — it is inherently a cross-model operation.

3. **Balanced accuracy as a second lens.** Accuracy alone is misleading when the test set is imbalanced (57.5% up / 42.5% down). Balanced accuracy (mean of per-class recall) reveals whether a model that achieves high accuracy is genuinely discriminating between classes or simply predicting the majority class. This cross-model diagnostic is the evaluation module's responsibility, not any individual model's.

---

## Metrics computed

### Classification metrics (per model)

| Metric | Definition | Why included |
|---|---|---|
| Accuracy | (TP + TN) / N | Primary metric, comparable to the 57.5% baseline |
| Balanced accuracy | (Recall_up + Recall_down) / 2 | Reveals class-imbalance dependence |
| Precision (up) | TP / (TP + FP) | When the model predicts up, how often is it right? |
| Recall (up) | TP / (TP + FN) | Of actual up-days, how many did the model catch? |
| F1 (up) | Harmonic mean of precision and recall for up class | Single-number summary for up-class performance |
| Precision (down) | TN / (TN + FN) | When the model predicts down, how often is it right? |
| Recall (down) | TN / (TN + FP) | Of actual down-days, how many did the model catch? |
| F1 (down) | Harmonic mean of precision and recall for down class | Single-number summary for down-class performance |
| Confusion matrix | TP, TN, FP, FN counts | Raw counts for transparency |
| MCC | (TP×TN − FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | The single metric that uses all four confusion matrix cells and is invariant to class relabelling. A value of 0 indicates random-chance performance regardless of class imbalance. All three models produce MCC near zero, confirming the balanced-accuracy finding through a different lens. |

### Statistical comparison (pairwise)

**McNemar's test** (with continuity correction) tests whether two classifiers' errors are symmetrically distributed. McNemar's test is the recommended statistical test for comparing two classifiers evaluated on the same test set (Dietterich, 1998, *Neural Computation*). It is preferred over the paired t-test across cross-validation folds because it operates on a single train/test split — matching the project's evaluation protocol — and tests whether the classifiers' errors are symmetrically distributed rather than whether their mean accuracies differ. Under H0, the number of observations where model A is correct and B is wrong should equal the reverse. A significant result (p < 0.05) indicates the models make systematically different errors — one is genuinely better or worse than the other, not just randomly different.

---

## Key findings from the real run

### No model exceeded the majority-class baseline

| Model | Accuracy | vs Baseline | Balanced Acc | Pred Up% |
|---|---|---|---|---|
| GeoBM | 57.5% | +0.0pp | 50.0% | 100.0% |
| GA | 56.7% | -0.8pp | 49.7% | 96.4% |
| XGBoost | 53.1% | -4.4pp | 48.5% | 80.4% |
| Majority | 57.5% | — | 50.0% | 100.0% |

### Balanced accuracy reveals the true picture

All three models have balanced accuracy at or below 50.0%. This is the critical diagnostic:

- **GeoBM** achieves 57.5% accuracy but 50.0% balanced accuracy. It predicts every day as "up," achieving perfect recall on up-days (1.000) and zero recall on down-days (0.000). Its accuracy is entirely a product of the class imbalance — it has no discriminative ability.

- **GA** achieves 56.7% accuracy but 49.7% balanced accuracy — below the 50% random baseline. Its 3.3% down-day recall (7/213) is trivially small. It is a near-constant "up" predictor with negligible class discrimination.

- **XGBoost** achieves 53.1% accuracy but 48.5% balanced accuracy — the worst of all three. Despite producing genuinely varying predictions (80.4% up, 19.6% down), its down-day recall (17.8%) and down-day precision (38.8%) are poor. When it deviates from "predict up," it is wrong more often than right. The additional modelling capacity actively hurt performance.

### McNemar's tests

| Pair | Chi-squared | p-value | Significant? |
|---|---|---|---|
| GeoBM vs GA | 0.500 | 0.4795 | No |
| GeoBM vs XGBoost | 4.500 | 0.034 | **Yes** |
| GA vs XGBoost | 3.141 | 0.076 | No |

GeoBM vs XGBoost is statistically significant at alpha = 0.05. This means XGBoost's inferior performance is not just random variation — it is systematically worse than the constant "up" predictor. The additional capacity and learned decision boundaries actively degraded accuracy relative to the simplest possible strategy.

GeoBM vs GA and GA vs XGBoost are not statistically significant, meaning the observed accuracy differences could be due to chance on 501 observations.

---

## Interpretation for the dissertation

The three-paradigm comparison produced a coherent null result: no model extracted generalisable directional signal from the 14 engineered features.

The results form a **capacity–accuracy inversion**: models with greater capacity performed worse, not better.

| Model | Capacity | Test accuracy |
|---|---|---|
| GeoBM | Minimal (2 parameters) | 57.5% (best) |
| GA | Low (3 threshold rules) | 56.7% |
| XGBoost | High (100 trees × depth 3) | 53.1% (worst) |

This inversion is the signature pattern of a low-signal environment: greater model capacity enables fitting more noise in training data, which harms out-of-sample performance. The simplest model (GeoBM) performs best because it has no capacity to overfit — it defaults to the prior (predict the majority class), which happens to be the optimal strategy when no feature contains reliable directional information.

This finding is consistent with the weak-form efficient market hypothesis (Fama, 1970): historical price and volume information — the basis of all 14 features — does not systematically predict next-day direction for a broad-market ETF. This interpretation is scoped to this asset (SPY), feature set (14 technical indicators), evaluation period (2023–2024), and prediction horizon (next-day binary direction). The result does not prove market efficiency — it demonstrates that these three paradigms could not falsify it on this dataset.

---

## Scope boundaries

**Walk-forward validation** would evaluate each model across multiple rolling temporal windows. This is a legitimate extension that would strengthen the evaluation but is deferred as a second-pass refinement. The current single-holdout evaluation answers the primary comparison question.

**Financial performance metrics** (PnL, Sharpe ratio, maximum drawdown) are excluded. The project evaluates directional prediction accuracy, not trading strategy profitability. Financial metrics would require position-sizing assumptions that are outside scope.

**Visualisation** (confusion matrix plots, calibration curves, ROC curves) is deferred to a separate plotting module. This evaluation module produces the numerical evidence; presentation is a downstream concern.

---

## Rubric mapping (COM3001 assessor language)

| Assessor criterion | How this component satisfies it |
|---|---|
| **Evaluation methodology** | Common harness applies identical metrics to all three models. McNemar's test provides statistical rigour. Balanced accuracy reveals class-imbalance dependence. |
| **Critical analysis** | Capacity–accuracy inversion identified and interpreted. Null result framed as evidence for market efficiency, not as failure. Balanced accuracy used to look beyond raw accuracy. |
| **Comparison with related works** | Three paradigms compared under identical conditions — the systematised comparison is the project's contribution. Results interpreted against the efficient market hypothesis literature. |
| **Replication detail** | All inputs are CSV files with known schemas. Metrics computed from standard definitions. Summary saved to CSV. Another researcher can reproduce by running `python src/evaluation/evaluate.py` from the repo root. |

---

## How this feeds the dissertation

**Evaluation chapter, "Comparative Results" section (~500–700 words).**

> *Ready-to-lift wording (results):* "Table X presents the side-by-side evaluation of all three paradigms on the 501-day held-out test set. No model exceeded the naive majority-class baseline of 57.5%. GeoBM achieved 57.5% by predicting 'up' on every day — a constant predictor with zero discriminative ability (balanced accuracy 50.0%). The GA achieved 56.7% with a near-constant up-biased prediction pattern (96.4% up, balanced accuracy 49.7%). XGBoost achieved 53.1% despite producing genuinely varying predictions (80.4% up, 19.6% down), with balanced accuracy of 48.5% — the lowest of all three models. McNemar's test confirmed that the accuracy difference between GeoBM and XGBoost is statistically significant (chi-squared = 4.500, p = 0.034), indicating that XGBoost's inferior performance is systematic rather than random. The results exhibit a capacity–accuracy inversion: models with greater modelling capacity performed worse on the held-out test set. This pattern is characteristic of low-signal environments where additional capacity enables fitting noise in the training data rather than capturing generalisable patterns. The simplest model (GeoBM) performed best because its two-parameter structure has no capacity to overfit; the most complex model (XGBoost) performed worst because its 100-tree ensemble memorised training-period statistical regularities that did not persist into the test period."

> *Ready-to-lift wording (interpretation):* "The null result — no paradigm exceeding the majority-class baseline — is itself the central finding of the comparative framework. It provides evidence that daily SPY direction is not reliably predictable from these 14 technical features under this evaluation protocol. This is consistent with the weak-form efficient market hypothesis (Fama, 1970), which predicts that historical price and volume information should not systematically forecast future returns. The result does not prove market efficiency — it demonstrates that three fundamentally different computational approaches, evaluated under controlled conditions, could not falsify it on this dataset."

---

## Evidence produced

| Metric | GeoBM | GA | XGBoost |
|---|---|---|---|
| Test accuracy | 57.5% | 56.7% | 53.1% |
| vs baseline | +0.0pp | -0.8pp | -4.4pp |
| Balanced accuracy | 50.0% | 49.7% | 48.5% |
| MCC | 0.0000 | -0.0142 | -0.0373 |
| Precision (up) | 0.5749 | 0.5735 | 0.5658 |
| Recall (up) | 1.0000 | 0.9618 | 0.7917 |
| F1 (up) | 0.7300 | 0.7185 | 0.6599 |
| Precision (down) | 0.0000 | 0.3889 | 0.3878 |
| Recall (down) | 0.0000 | 0.0329 | 0.1784 |
| F1 (down) | 0.0000 | 0.0606 | 0.2444 |
| Predicted up % | 100.0% | 96.4% | 80.4% |
| Confusion: TP | 288 | 277 | 228 |
| Confusion: TN | 0 | 7 | 38 |
| Confusion: FP | 213 | 206 | 175 |
| Confusion: FN | 0 | 11 | 60 |

McNemar's tests: GeoBM vs GA p=0.4795 (NS), GeoBM vs XGBoost p=0.034 (significant), GA vs XGBoost p=0.076 (NS).

Output saved to `results/tables/model_comparison.csv`.
