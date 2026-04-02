# Component Note: geobm_baseline.py

**File:** `src/models/geobm/geobm_baseline.py`
**Date:** 2026-04-02

---

## Purpose

Estimates Geometric Brownian Motion parameters (drift mu, volatility sigma) from the training-period log-return series, computes the analytic probability that the next daily return is positive, and generates a directional prediction for every day in the test set. Saves predictions to `results/predictions/geobm_predictions.csv` in a format compatible with the future common evaluation harness.

This module is the first of three modelling baselines in the project's comparative framework. Its role is not to maximise accuracy — it is to establish what a minimal stochastic process with estimated drift and volatility achieves under the same held-out evaluation boundary as the later GA and XGBoost models. The GeoBM baseline operationalises the null hypothesis of the comparison: if returns are IID log-normal draws characterised entirely by drift and volatility, how well can directional prediction perform?

---

## Why GeoBM is a separate module

Each paradigm in the three-model comparison occupies its own module because they differ in assumptions, parameter estimation methods, prediction logic, and failure modes. Bundling GeoBM with the GA or XGBoost would conflate fundamentally different modelling philosophies: GeoBM assumes returns are IID and characterised by two parameters; the GA evolves rule-based strategies over the feature space; XGBoost learns non-linear decision boundaries from the engineered features.

Separation also ensures that each model's parameter estimation is independently auditable for train/test leakage. GeoBM estimates mu and sigma from training returns only. If this estimation were embedded in a shared model module, verifying that no test information leaks into the GBM parameters would require understanding the entire multi-model codebase rather than a single self-contained file.

---

## Why GeoBM is an appropriate baseline in this project

Geometric Brownian Motion is the foundational stochastic process model in financial economics. It underpins the Black-Scholes option pricing framework (Black & Scholes, 1973), forms the basis of the Efficient Market Hypothesis in its log-normal return interpretation, and serves as the default null model against which any claim of return predictability must be tested.

The appropriateness of GeoBM as a baseline in this comparative framework rests on three properties:

1. **It encodes the minimal-information hypothesis.** GeoBM assumes that the entire information content of the return-generating process is captured by two parameters: drift (the expected return) and volatility (the standard deviation of returns). If the GA or XGBoost cannot outperform a prediction derived from these two numbers alone, then the 14 engineered features add no predictive value beyond what the first two moments of the return distribution already provide.

2. **It is analytically transparent.** The prediction rule has a closed-form derivation. There are no hyperparameters to tune, no model selection decisions, no stochastic training procedures. This makes the baseline fully reproducible and eliminates any concern about overfitting the evaluation protocol itself.

The analytic directional probability follows directly from the log-normal return model (Hull, 2018, Ch. 15).

3. **It shares the same evaluation boundary.** GeoBM estimates parameters from the training set (2010-02-02 to 2022-12-30) and is evaluated on the test set (2023-01-03 to 2024-12-30) under the same chronological holdout split as the other two models. Performance comparisons are therefore fair — all three models see the same training data and are assessed on the same held-out period.

---

## The role of this baseline in the three-model comparison

The comparative framework evaluates three paradigms against a common held-out test set:

| Model | Assumptions | Inputs used | Prediction mechanism |
|---|---|---|---|
| GeoBM | Returns are IID log-normal | Raw returns only (2 parameters) | Analytic probability |
| GA | Rules over feature thresholds can generate excess returns | Engineered feature space | Evolved directional rules |
| XGBoost | Non-linear feature interactions predict direction | 14 engineered features | Learned decision trees |

GeoBM anchors the bottom of this hierarchy. If the GA or XGBoost fail to exceed the GeoBM baseline, the implication is that the feature-based models have not learned any structure beyond what the unconditional return distribution already captures. If they do exceed it, the margin of improvement quantifies the predictive value of the engineered features and the non-linear modelling capacity.

Crucially, the GeoBM baseline also establishes whether the majority-class baseline (57.5%) and the GeoBM baseline are equivalent — which, in this dataset, they are. This is itself an informative finding: the stochastic model does not outperform naive class-frequency prediction.

---

## Assumptions of GeoBM and their implications

Geometric Brownian Motion assumes that the asset price S_t follows:

    dS = mu * S * dt + sigma * S * dW

where dW is a Wiener process increment. In discrete form, this implies that log returns are IID normal:

    log(S_{t+1} / S_t) ~ N(mu - 0.5 * sigma^2, sigma^2)

This model makes four substantive assumptions, each of which is known to be violated in empirical equity data:

**1. IID returns.** GBM assumes that today's return is independent of yesterday's. Empirically, equity returns exhibit short-term autocorrelation (Lo & MacKinlay, 1988) and volatility clustering (Cont, 2001). The IID assumption means GBM cannot capture momentum, mean-reversion, or regime-dependent behaviour.

**2. Constant parameters.** Drift and volatility are assumed fixed over the entire training period. In practice, both drift and volatility are time-varying. The training period (2010–2022) spans a post-GFC recovery, a long bull market, the COVID crash, and a rapid recovery — the single-estimate mu and sigma represent an average over these heterogeneous regimes.

**3. Log-normal returns.** GBM implies that log returns are normally distributed. Empirical return distributions have heavier tails (excess kurtosis) and potential asymmetry (negative skewness), particularly for daily data. This means GBM underestimates the probability of extreme moves.

**4. No information beyond first two moments.** GBM uses only the mean and variance of the return distribution. It does not incorporate volume, momentum, volatility regimes, or any higher-order structure. This is the assumption that the GA and XGBoost are designed to test: does the 14-feature set contain directional information beyond what mu and sigma capture?

These violations are not flaws in the implementation — they are the reason GeoBM serves as a baseline rather than a competitor. The model is deliberately simple so that any improvement by the GA or XGBoost can be attributed to their capacity to model structure that GeoBM cannot.

---

## How mu and sigma are estimated

The CRSP `return` column provides simple daily returns r_t. These are converted to log returns:

    x_t = log(1 + r_t)

The sample mean and sample standard deviation of the log-return series are computed over the 3,252 training observations:

    log_return_mean = mean(x_t) = 0.00046223
    log_return_std  = std(x_t, ddof=1) = 0.01113145

Under the GBM discrete approximation, the expected value of the log return is (mu - 0.5 * sigma^2), not mu itself. The drift correction accounts for the difference between the arithmetic mean and the geometric mean of returns. The parameters are recovered as:

    sigma = log_return_std = 0.01113145
    mu = log_return_mean + 0.5 * sigma^2 = 0.00052419

The recovered drift μ is reported for diagnostic completeness and to verify that the drift correction term is small relative to the log-return mean, but the directional prediction depends only on the compound quantity (μ − 0.5σ²), which is estimated directly as the sample mean of log returns.

All estimates are at daily frequency. No annualisation is applied because the prediction horizon is daily. Annualising and then de-annualising would introduce a redundant pair of multiplications that add no information and risk unit-conversion errors.

---

## How the next-day direction is derived

Under GBM, the probability that the next day's log return exceeds zero is:

    P(x_{t+1} > 0) = P(Z > -log_return_mean / sigma)
                    = Phi(log_return_mean / sigma)

where Z is standard normal and Phi is the standard normal CDF. Since log_return_mean estimates (μ − 0.5σ²) directly, P(x > 0) = Φ((μ − 0.5σ²)/σ) = Φ(log_return_mean / σ). Since log(1 + r) > 0 if and only if r > 0, the event that the log return exceeds zero is identical to the event that the simple return exceeds zero. The directional prediction is therefore fully consistent with the target definition in this project, where the label is defined on simple next-day returns. Substituting the estimated values:

    P(x_{t+1} > 0) = Phi(0.00046223 / 0.01113145)
                    = Phi(0.04153)
                    = 0.516561

The prediction rule is: if P(up) > 0.5, predict 1 (up); otherwise predict 0 (down). Since 0.5166 > 0.5, the model predicts "up" on every test day.

**This is not a bug — it is the central finding.** Under the GBM assumptions, the estimated positive drift implies that the unconditional probability of an up day slightly exceeds 0.5. The model has no mechanism to vary its prediction across days because the IID assumption makes every day identical in expectation. The constant-prediction outcome is a mathematically necessary consequence of positive drift under IID log-normal returns, and it is exactly what makes GeoBM an informative baseline: any model that produces non-trivial variation in its predictions is, by definition, modelling structure that GeoBM cannot represent.

Had the estimated compound drift term been negative, the same rule would have produced a constant "down" prediction instead. This shows that the classifier is conditional on the training-period parameter estimates rather than hard-coded to predict "up."

---

## Rejected alternatives

### Monte Carlo simulation

One-step-ahead Monte Carlo simulation (drawing N random log returns from N(mu - 0.5*sigma^2, sigma^2) and computing the fraction that exceed zero) was considered and rejected for the directional baseline. The analytic CDF gives the exact answer without sampling noise. Monte Carlo would introduce stochastic variation across runs, require a fixed seed for reproducibility, and add implementation complexity — all for a result that converges to the same Phi(0.04153) = 0.5166 as the sample size grows. Monte Carlo is valuable when the quantity of interest has no closed-form solution; here it does, so the analytic method is strictly preferred.

### Rolling parameter re-estimation

Re-estimating mu and sigma using an expanding or rolling window as the test period progresses was considered and rejected. Rolling re-estimation would allow the model to adapt to changing market conditions, but it would also use test-period returns in parameter estimation — violating the held-out evaluation protocol. The training-only estimation is the correct design for a baseline that must be comparable to the GA and XGBoost under the same evaluation boundary.

### Using engineered features

The 14 engineered features (lagged returns, volatility, momentum, RSI, etc.) are available in the train/test CSVs but are not used by GeoBM. Using them would convert GeoBM from a stochastic process baseline into a feature-based model, eliminating the comparison anchor. The entire point of the three-paradigm comparison is to test whether feature-based models outperform a model that uses only the raw return distribution. Giving GeoBM access to features would collapse the bottom of the comparison hierarchy.

---

## Challenging aspects and how they were resolved

### 1. The drift correction term

The GBM log-return distribution has mean (mu - 0.5 * sigma^2), not mu. Confusing these two quantities is a common error in financial computing. In this implementation, the sample mean of log returns directly estimates (mu - 0.5 * sigma^2), and the directional probability depends on this compound quantity, not on mu alone. The drift correction term (0.5 * sigma^2 = 0.00006195) is small relative to the log-return mean (0.00046223) but is handled correctly for methodological rigour — and logged explicitly so it can be verified.

### 2. Constant-predictor degeneration

The model predicts "up" on every test day because the estimated drift is positive. This was anticipated, surfaced in the diagnostics, and framed as a finding rather than a failure. The constant-prediction outcome means GeoBM accuracy equals the up-day fraction in the test set (57.5%), which equals the majority-class baseline. This is informative: a stochastic model calibrated to 13 years of training data produces exactly the same out-of-sample performance as a naive classifier that counts class frequencies. The two baselines are equivalent, and any model that exceeds both has demonstrated genuine predictive value.

### 3. Simple vs log returns

The CRSP `return` column provides simple returns. GBM is formulated in terms of log returns. The conversion x_t = log(1 + r_t) is applied before parameter estimation. The directional prediction (sign of log return) is equivalent to the sign of the simple return (since log is monotonic and log(1) = 0), so no back-conversion is needed for the binary prediction. This equivalence is noted because it eliminates a potential source of confusion: "up" means the same thing under both return conventions.

### 4. Annualisation

No annualisation is applied anywhere in the module. Both the estimation frequency (daily) and the prediction horizon (daily) are at the same timescale, so annualisation would be a redundant identity operation. Many GeoBM implementations annualise sigma by multiplying by sqrt(252) — this is appropriate for option pricing or risk management where annualised volatility is the convention, but it is unnecessary and potentially confusing for a daily directional predictor. The choice to keep everything at daily frequency is logged explicitly.

---

## Scope boundaries

This section documents what this module deliberately does not do, and why each exclusion is a bounded design choice.

**Rolling or adaptive estimation** would allow the model to track time-varying drift and volatility, but would require using test-period data in parameter updates. This violates the held-out evaluation protocol and is excluded by design. A walk-forward variant of GeoBM would be a valid extension but belongs in the evaluation module, not the baseline.

**Price-path simulation** (multi-step GBM trajectories) is not needed. The prediction task is one-step-ahead direction, which has an analytic solution under GBM. Simulating full price paths would add complexity without changing the directional prediction.

**Trading strategy overlay** (position sizing, stop losses, transaction costs) is an evaluation-stage concern. This module produces directional predictions in a generic format; strategy-level evaluation operates on those predictions downstream.

**Feature usage** is excluded by design. GeoBM operates on the raw return series as a stochastic process baseline. The 14 engineered features are inputs for the GA and XGBoost only.

---

## Rubric mapping (COM3001 assessor language)

| Assessor criterion | How this component satisfies it |
|---|---|
| **Requirements analysed** | The comparative framework requires a transparent stochastic baseline that anchors the bottom of the three-paradigm hierarchy. GeoBM provides this by deriving directional predictions from the two-parameter return distribution alone. |
| **Design issues discussed** | Analytic prediction justified over Monte Carlo simulation. Training-only estimation justified over rolling re-estimation. Feature exclusion justified as necessary for comparison-framework integrity. Drift correction handled correctly. Constant-predictor degeneration surfaced and interpreted rather than hidden. |
| **Challenging aspects** | Drift correction term (mu vs mu - 0.5*sigma^2). Constant-predictor outcome as a necessary consequence of positive drift under IID assumptions. Log vs simple return equivalence for directional prediction. Annualisation excluded with explicit justification. |
| **Replication detail** | Parameters estimated from training data only. Analytic method requires no random seed. All paths config-driven. Exact parameter values logged. Another researcher can reproduce by running `python src/models/geobm/geobm_baseline.py` from the repo root. |
| **Advanced knowledge demonstrated** | GBM formulation with correct Itô drift correction. Analytic directional probability from the log-normal CDF. Understanding that constant-predictor degeneration is a mathematical consequence, not an implementation error. Separation of stochastic process baseline from feature-based models as a comparison design decision. |

---

## How this feeds the dissertation

**Methodology chapter, "Baseline Model" subsection (~400–500 words).**

> *Ready-to-lift wording (methodology):* "The first component of the comparative framework is a Geometric Brownian Motion baseline that derives directional predictions from the unconditional return distribution alone. GBM assumes that log returns are independent, identically distributed normal draws characterised by drift mu and volatility sigma. Both parameters were estimated from the training-period log-return series (3,252 daily observations, February 2010 to December 2022) at daily frequency without annualisation. The sample mean of log returns (0.00046223) estimates the compound drift term (mu - 0.5 * sigma^2), and the sample standard deviation (0.01113145) estimates sigma directly. No test-period data entered parameter estimation. The directional prediction was derived analytically: under GBM, the probability that the next log return exceeds zero equals Phi((mu - 0.5 * sigma^2) / sigma) = 0.5166, where Phi is the standard normal CDF. Since this probability exceeds 0.5, the model predicts 'up' on every test day. Monte Carlo simulation was considered and rejected because the analytic CDF provides the exact answer without sampling noise. Rolling parameter re-estimation was rejected because it would require test-period returns, violating the held-out evaluation boundary."

**Evaluation chapter, "GeoBM Baseline Results" paragraph (~200–300 words).**

> *Ready-to-lift wording (evaluation):* "The GeoBM baseline achieved 57.5% accuracy on the 501-day test set, predicting 'up' on every day. This equals the majority-class baseline exactly: a naive classifier that predicts the most frequent class achieves the same 57.5% by counting the up-day fraction in the test period. The equivalence is not a coincidence — it is a mathematical consequence of the model's structure. Under GBM, the estimated positive drift (mu = 0.00052419 daily) implies that the unconditional probability of an up day slightly exceeds 0.5 (P(up) = 0.5166). Since this probability is constant across all days — the IID assumption makes every day identical in expectation — the model has no mechanism to produce day-varying predictions. The constant-predictor outcome demonstrates the fundamental limitation of drift-based directional prediction: a model that knows only the mean and variance of the return distribution cannot distinguish days that will go up from days that will go down. Any model in the comparative framework that exceeds 57.5% accuracy has therefore demonstrated predictive value beyond what the first two moments of the return distribution provide. The GeoBM baseline and the majority-class baseline are equivalent anchors for interpreting the GA and XGBoost results."

---

## Evidence produced

This component estimated GBM parameters from 3,252 training observations and generated directional predictions for 501 test observations. The exact outputs:

| Metric | Value |
|---|---|
| Training rows | 3,252 |
| Test rows | 501 |
| Return definition | log(1 + r_simple) |
| Log-return mean | 0.00046223 |
| Log-return std (sigma) | 0.01113145 |
| Drift (mu) | 0.00052419 |
| Drift correction (0.5 * sigma^2) | 0.00006195 |
| P(next return > 0) | 0.516561 |
| Prediction method | Analytic (no simulation) |
| Predicted class | Up (1) on all 501 days |
| Test accuracy | 57.5% (288 / 501) |
| Majority-class baseline | 57.5% (288 / 501) |
| Exceeds baseline | No — model equals naive majority-class predictor |
| Constant predictor | Yes — predicts "up" on every day |
| Random seed used | Not applicable (analytic method) |

Output saved to `results/predictions/geobm_predictions.csv` with columns: date, target, predicted, p_up, model.
