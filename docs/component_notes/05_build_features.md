# Component Note: build_features.py

**File:** `src/features/build_features.py`
**Date:** 2026-03-31

---

## Purpose

Reads the target-bearing dataset (`data/processed/spy_targeted.csv`), engineers 14 technical features from price, return, and volume data using only information available at or before time t, drops warm-up rows where rolling lookback windows produce NaN, validates that no feature leaks future information, and saves the feature-bearing dataset to `data/processed/spy_featured.csv`.

---

## Why feature engineering is a separate module

The upstream pipeline stages enforce strict separation of concerns: ingestion preserves raw CRSP data, preprocessing standardises CRSP conventions, and target construction encodes the supervised learning task. Feature engineering is the fourth distinct concern — it transforms raw OHLCV columns into the predictor variables that models will consume.

This separation matters for three reasons. First, feature choices are modelling decisions, not data properties. Which indicators to compute, what lookback windows to use, and how to handle warm-up rows are choices that depend on the models and the prediction task. Embedding them in preprocessing would conflate data cleaning with modelling assumptions. Second, the feature set must be auditable for look-ahead bias. If features were scattered across multiple modules, verifying that no feature uses future information would require inspecting the entire pipeline. A single module with a single responsibility makes anti-leakage auditing tractable. Third, features may change during ablation studies in the evaluation chapter. A dedicated module means feature ablation requires editing one file, not reconstructing the pipeline.

This follows López de Prado's (2018, *Advances in Financial Machine Learning*) principle that each pipeline stage should have exactly one transformation rationale, and Pardo's (2008, *The Evaluation and Optimization of Trading Strategies*) emphasis that feature construction is where look-ahead bias most commonly enters backtesting systems.

---

## Anti-leakage discipline

Every feature in this module obeys a strict temporal constraint: the value of feature *f* at time *t* is computed using only information available at or before time *t*.

**The information boundary.** The prediction target is next-day direction — whether SPY's return on day *t+1* will be positive. The prediction is made at the close of day *t*. Therefore, the information boundary is: any data point that is known at or before market close on day *t* is permissible; any data point that depends on day *t+1* or later is forbidden. This means today's closing price, today's volume, and today's return are all permissible inputs (they are realised by market close on day *t*). Tomorrow's return, tomorrow's volume, and the target label itself are forbidden. The critical subtlety is that return-derived features (lagged returns, volatility, RSI) must use an explicit `shift(1)` to exclude the same-day return from rolling calculations, because the same-day return is only known at close — the same moment the prediction is made — creating an ambiguity that is safest resolved by excluding it.

The specific mechanisms enforcing this boundary are:

**Lagged returns** use `shift(k)` where k ≥ 1, so the return at time t-k is assigned to row t. No same-day return information is used.

**Realised volatility** applies `shift(1)` to the return series before the rolling window, so the volatility estimate at time t reflects returns from t-window-1 to t-1. The return on day t itself is excluded.

**Momentum** uses `close.shift(k)` in the denominator, comparing today's close (observable at market close on day t, before the next-day target is determined) to the close from k days ago.

**Moving average ratios** use the close price up to and including day t. Since the prediction target is next-day direction, today's closing price is available information at prediction time.

**RSI** applies `shift(1)` to the return series before computing the Wilder EMA, so the RSI at time t reflects price changes up to t-1 only.

**Volume features** use today's volume, which is observable at market close on day t — the same information boundary as the closing price.

An automated validation check runs after feature construction: no feature may be identical to the target or have correlation above 0.95 with it. This does not prove the absence of subtle leakage, but it catches the most common error pattern (accidentally using the next-day return directly as a feature).

---

## Feature set design: 14 features in 6 categories

The feature set is deliberately constrained to 14 features across 6 categories. Each category is justified by established financial econometrics literature, and each feature has a clear informational role in the three-paradigm comparison.

### Category 1: Lagged returns (ret_lag_1, ret_lag_2, ret_lag_5)

Lagged returns capture short-term serial dependence (autocorrelation) in equity returns. While the Efficient Market Hypothesis in its strong form implies returns are unpredictable, empirical evidence consistently documents weak but statistically significant autocorrelation at daily frequency for broad-market indices (Lo & MacKinlay, 1988, *Review of Financial Studies*: "Stock Market Prices Do Not Follow Random Walks"). Three lags (1, 2, and 5 trading days) capture overnight, two-day, and weekly serial patterns without introducing excessive dimensionality.

For the comparative framework: GeoBM does not use these features (it estimates drift and volatility from raw returns). The GA can evolve threshold rules on lagged returns. XGBoost can learn non-linear interaction patterns across lags.

### Category 2: Realised volatility (vol_5, vol_10, vol_20)

Realised volatility — the rolling standard deviation of returns — is the empirical counterpart to the theoretical volatility parameter sigma in Geometric Brownian Motion. Volatility clustering (high-volatility periods tend to persist) is one of the most robust stylised facts in financial time series (Cont, 2001, *Quantitative Finance*: "Empirical Properties of Asset Returns"). Three windows (5, 10, 20 days) capture weekly, fortnightly, and monthly volatility regimes.

For the comparative framework: GeoBM does not consume these engineered features directly — it estimates drift and volatility from the raw return series over a 252-day estimation window (configured in `data_config.yaml`). However, the realised volatility features provide the GA and XGBoost with explicit volatility-regime signals that GeoBM captures implicitly through its parameter estimation. This asymmetry is itself a meaningful comparison point: GeoBM absorbs volatility into its model structure, while the GA and XGBoost receive it as an input feature they can choose to weight or ignore.

### Category 3: Momentum (mom_5, mom_10)

Price momentum — the tendency of recent winners to continue outperforming — is one of the most documented anomalies in financial economics (Jegadeesh & Titman, 1993, *Journal of Finance*: "Returns to Buying Winners and Selling Losers"). Short-term momentum over 5 and 10 trading days is particularly relevant for daily direction prediction on liquid ETFs, where the reversal horizon is typically longer than the momentum horizon.

For the comparative framework: momentum provides the GA with a signal to evolve threshold-based rules, and XGBoost with a continuous feature for tree splits. GeoBM does not use momentum directly but benefits indirectly through the drift parameter.

### Category 4: Moving average ratios (close_sma5_ratio, close_sma20_ratio, sma5_sma20_ratio)

Moving average crossover strategies are among the most widely studied systematic trading rules in financial economics (Brock, Lakonishok & LeBaron, 1992, *Journal of Finance*: "Simple Technical Trading Rules and the Stochastic Properties of Stock Returns"). Expressing the crossover as a continuous ratio rather than a binary cross/no-cross signal preserves information about the magnitude of separation between the short and long averages, providing XGBoost with a richer input space and the GA with a finer threshold surface.

close_sma5_ratio and close_sma20_ratio capture the position of the current price relative to its short-term and medium-term average. sma5_sma20_ratio captures the classic short/long crossover signal in continuous form.

### Category 5: RSI (rsi_14)

The Relative Strength Index (Wilder, 1978, *New Concepts in Technical Trading Systems*) is a bounded oscillator ranging from 0 to 100 that measures the magnitude of recent gains versus losses. RSI-14 is the standard parameterisation. Its bounded range makes it particularly useful for the GA's chromosome design (thresholds can be expressed as values in [0, 100] with natural semantic meaning: below 30 is oversold, above 70 is overbought).

The RSI is computed using Wilder's smoothing method (exponential moving average with alpha = 1/14), which is the original and most widely implemented version.

### Category 6: Volume signals (volume_chg_1, volume_sma20_ratio)

Volume provides information complementary to price — it captures the conviction behind price moves. The volume-price relationship is well-documented in market microstructure literature (Karpoff, 1987, *Journal of Financial and Quantitative Analysis*: "The Relation Between Price Changes and Trading Volume"). volume_chg_1 captures daily volume surprises. volume_sma20_ratio expresses today's volume relative to its 20-day average, normalising for the well-known intraweek and monthly seasonality in trading volume.

---

## Warm-up row handling

Rolling windows of size *n* produce NaN values for the first *n-1* rows. The maximum lookback in this feature set is 20 (for vol_20, close_sma20_ratio, and volume_sma20_ratio), but realised volatility also applies a 1-day shift before the rolling window, extending the effective warm-up to 21 rows. After feature construction, all rows containing any NaN in any feature column are dropped.

This approach is preferable to imputing warm-up rows because: (1) there is no defensible imputation method for the beginning of a time series where the rolling window has insufficient history; (2) the warm-up rows represent a tiny fraction of the dataset (20/3773 = 0.5%); and (3) retaining imputed warm-up rows would introduce synthetic feature values that could bias model training, particularly for XGBoost's tree splits where every training observation influences the split boundaries.

---

## Alternatives rejected

**MACD (Moving Average Convergence Divergence).** MACD was included in the initial config.yaml feature specification but excluded from this first-pass implementation. MACD is a composite indicator (it combines two EMAs and a signal line, producing three values: MACD line, signal line, and histogram). Including it would add three features with overlapping informational content to the existing MA ratio features. The MA ratios already capture the short/long moving average relationship in continuous form. The trade-off: MACD is widely recognised in the literature, but adding it here would increase dimensionality without proportionate informational gain, violating the principle of a controlled, defensible feature set. It can be added in a second pass if ablation studies in the evaluation chapter demonstrate that the current set leaves identifiable gaps.

**MACD signal line crossover (binary).** A binary version of the MACD cross was in the original config. Binary features destroy information that the continuous sma5_sma20_ratio preserves — XGBoost can learn its own thresholds more effectively from the continuous representation.

**Additional lag horizons (lag_3, lag_10, lag_20).** More lags were considered to capture longer-term serial dependence. They were rejected because: (1) autocorrelation in SPY daily returns decays rapidly beyond 5 days (this is verifiable in the evaluation chapter); (2) additional lags increase collinearity among the lagged return features without strong evidence of marginal predictive value; (3) the 5-day lag already captures the weekly cycle, and momentum features capture longer horizons more cleanly.

**Bollinger Bands.** Bollinger Bands (Bollinger, 2002) express price position relative to a rolling mean ± 2 standard deviations. The close_sma20_ratio and vol_20 features jointly capture the same information — price relative to its moving average and current volatility regime — without the arbitrary 2-sigma threshold that Bollinger Bands impose. The trade-off: Bollinger Bands are familiar to practitioners, but the decomposed representation is more informative for tree-based models.

**External features (VIX, yield curve, sector rotation).** External data sources were considered but rejected for this first-pass feature set because: (1) they would require additional data pipelines and WRDS queries, introducing complexity without advancing the core comparison; (2) the project's contribution is the comparison framework, not feature maximisation; (3) Ioana explicitly directed against bolt-on complexity that does not serve the core deliverable. External features are documented as future work for the evaluation chapter.

**`ta` library for indicator calculation.** The `ta` Python library could compute RSI, MACD, Bollinger Bands, and dozens of other indicators in a few lines. It was rejected because: (1) the library's internal implementation of look-ahead bias prevention cannot be audited without inspecting its source code — computing features manually with explicit `shift()` calls makes the anti-leakage logic visible and verifiable; (2) using a library for 14 features provides no meaningful efficiency gain over manual implementation; (3) manual implementation demonstrates understanding of the indicator mathematics, which earns marks under "knowledge beyond taught courses."

---

## What this module does not do

This module does not perform the temporal train/test split. The split boundary (train ends 2022-12-31, test starts 2023-01-03) depends on the complete feature-bearing dataset being ready, but the split itself is a downstream concern with its own justification (preventing data leakage between training and evaluation periods).

This module does not scale or normalise features. Scaling is model-dependent: XGBoost is invariant to monotonic transformations and does not require scaling; the GA's fitness function operates on trading returns, not raw feature values; GeoBM uses only return statistics, not the engineered features directly. Scaling in this module would impose a modelling assumption that may not be appropriate for all three paradigms.

This module does not select or rank features. Feature importance analysis (SHAP values, permutation importance, ablation studies) is an evaluation-stage concern that depends on trained model outputs. Performing it here would conflate feature engineering with model evaluation.

This module does not modify the target column. The binary next-day direction label was constructed upstream by `build_target.py` and passes through this module unchanged.

This module does not use the `ta` library or any external indicator library. All features are computed manually from raw price, return, and volume columns using pandas and numpy operations. This ensures that look-ahead bias prevention is verifiable by inspection of this single file.

---

## Risk / Limitation

**Acceptable in this context:**

The 14-feature set is a first-pass selection. It may omit signals that would improve prediction accuracy (e.g., MACD, Bollinger Bands, order flow imbalance, cross-asset signals). This is acceptable because: (1) the project's contribution is the comparison framework, not feature maximisation — demonstrating that the framework works with a defensible feature set is sufficient; (2) the evaluation chapter can include ablation studies showing the marginal value of feature subsets; (3) a smaller, well-justified feature set is more defensible than a large pile of weakly justified indicators, aligning with Ioana's directive that sophistication comes from thought, not volume.

The warm-up row drop (20 rows, 0.5% of data) removes observations from early January 2010 through 1 February 2010. This shifts the effective start date to 2 February 2010 but does not affect the temporal split boundary (2022-12-31) or the test period (2023–2024). The lost observations are from the earliest training period and have minimal impact on model training.

RSI computation divides average gain by average loss. If avg_loss is exactly zero while avg_gain is positive, the relative strength term becomes inf, which correctly yields RSI = 100 in the standard formula. If both average gain and average loss are zero, the ratio is undefined and may produce NaN. In practice, the real SPY run produced RSI values from 6.93 to 91.45 with zero null or infinite values after warm-up removal, so this is a theoretical edge case rather than an empirical problem in this dataset.

**What a more advanced version would add (not needed here):**

Config-driven feature specification where the feature list, lookback windows, and indicator parameters are read from YAML rather than hardcoded. A feature registry pattern where each feature function self-registers its name, lookback requirement, and anti-leakage proof. Automated look-ahead bias testing using the method of López de Prado (2018) — shuffling the date index and checking whether feature-target correlations change. These would strengthen the methodology for a multi-asset or production system but are disproportionate for a single-asset proof-of-concept.

---

## Evidence produced

This component produced a feature-bearing dataset of 3,753 SPY daily observations from 2010-02-02 to 2024-12-30, with 8 original columns (date, open, high, low, close, volume, return, target) plus 14 engineered features, totalling 22 columns. 20 warm-up rows were dropped (0.5% of the input dataset). Zero null values in any column after warm-up drop. Anti-leakage validation passed: no feature mirrors or is suspiciously correlated with the target. Class balance preserved at 55.3% up / 44.7% down (2,076 / 1,677). RSI range: 6.93–91.45 (bounded as expected). Maximum single-day return magnitude in lagged features: ±10.9% (consistent with the March 2020 COVID crash). Output saved to `data/processed/spy_featured.csv`.

---

## How this feeds the dissertation

**Methodology chapter, "Feature Engineering" subsection (~600–800 words).** The 6 feature categories, their literature justifications, the anti-leakage discipline, the warm-up handling rationale, and the rejected alternatives translate directly into methodology prose. The feature inventory table and null-status diagnostics become figures.

> *Ready-to-lift wording:* "Fourteen technical features were engineered across six categories: lagged returns (1, 2, and 5-day lags capturing short-term serial dependence; Lo & MacKinlay, 1988), realised volatility (5, 10, and 20-day rolling standard deviation of returns; Cont, 2001), price momentum (5 and 10-day percentage price change; Jegadeesh & Titman, 1993), moving average ratios (close-to-SMA and short/long SMA ratios in continuous form; Brock, Lakonishok & LeBaron, 1992), the 14-day Relative Strength Index (Wilder, 1978), and volume signals (daily volume change and 20-day volume ratio; Karpoff, 1987). All features were computed using only information available at or before the prediction date: lagged returns and RSI used explicit one-day shifts to exclude same-day return information, and realised volatility windows were applied to the shifted return series. Rows where any feature was undefined due to insufficient lookback history were dropped, removing 20 observations (0.5% of the dataset) from the earliest training period."

> *Ready-to-lift wording (rejected alternatives):* "Several additional indicators were considered and rejected in the first-pass feature set. MACD was excluded because its informational content overlaps with the continuous moving average ratios already included. Additional return lags beyond 5 days were excluded because daily return autocorrelation decays rapidly for broad-market ETFs, and longer-horizon effects are better captured by the momentum features. External features (VIX, yield curve) were excluded to preserve the single-asset experimental control and avoid introducing data pipeline complexity that does not serve the comparative framework's core contribution."

**Evaluation chapter, "Feature Ablation" subsection.** The controlled feature set enables ablation studies: removing one category at a time and measuring the impact on each model's performance. The 6-category structure provides a natural ablation framework.

> *Ready-to-lift wording:* "Feature ablation studies were conducted by removing one feature category at a time and re-evaluating each model under the same test conditions. This controlled design — made possible by the modular feature engineering architecture — isolates the marginal contribution of each feature category to each modelling paradigm's predictive performance."

**Context chapter, "Technical Analysis in Computational Finance" paragraph.** The literature justifications for each feature category anchor the lit review's discussion of technical indicators as inputs to algorithmic trading systems.

---

## Rubric mapping (COM3001 assessor language)

| Assessor criterion | How this component satisfies it |
|---|---|
| **Requirements analysed** | The comparative framework requires a common feature set that all three paradigms can consume (directly for XGBoost, as rule inputs for GA, as parameter context for GeoBM). This module constructs that shared feature layer. |
| **Design issues discussed** | Six feature categories justified with literature. Anti-leakage discipline documented with specific mechanisms per feature. Warm-up handling justified (drop vs. impute). Six rejected alternatives with explicit trade-offs. Manual computation over `ta` library justified (auditability over convenience). |
| **Challenging aspects** | Anti-leakage discipline is the primary challenge. The distinction between "observable at time t" and "uses information from time t+1" is non-trivial for features like same-day volume and same-day closing price. The explicit `shift(1)` pattern for return-derived features and the rationale for not shifting price-derived features demonstrate understanding of the information boundary. |
| **Replication detail** | Canonical `FEATURE_COLS` list as single source of truth. Deterministic computation order. Diagnostics logged (null counts, warm-up rows, feature statistics). Another researcher can reproduce this step by running `python src/features/build_features.py` from the repo root. |
| **Advanced knowledge demonstrated** | Look-ahead bias prevention in feature engineering (López de Prado, 2018). Wilder's smoothing for RSI (not taught in standard CS modules). Continuous MA ratios over binary crossover signals (information-preserving transformation). Volatility clustering as a stylised fact (Cont, 2001). Separation of feature engineering from feature selection (methodological discipline). |
| **Innovation in approach** | The anti-leakage validation check (correlation scan against target), the explicit `shift(1)` pattern for return-derived features, and the manual implementation choice (auditability over library convenience) demonstrate methodological care that goes beyond standard implementations. |
| **Evaluation — ablation foundation** | The 6-category feature structure provides a natural framework for ablation studies in the evaluation chapter. Each category can be removed independently to measure its marginal contribution to each model. |
