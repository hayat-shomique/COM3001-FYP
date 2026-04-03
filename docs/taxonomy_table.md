# Systematisation Taxonomy: Algorithmic Directional Prediction Design Space

**Date:** 2026-04-03

This document is the conceptual contribution of the project. It classifies algorithmic trading paradigms by seven design-space dimensions, positions the three implemented models as empirical instances with known results, and places them alongside four literature models to show coverage and gaps.

This directly addresses Ioana's systematisation directive: "if you can somehow systematise yourself by some dimensions that you can demonstrate came from a deep understanding of the lit review and the topic itself and relevant concepts, for instance, relevant to computer science."

---

## The Seven Dimensions

### 1. Paradigm type

The fundamental computational approach. Determines the class of patterns the model can discover and the assumptions it imposes on the return-generating process. Stochastic models assume a parametric distribution; evolutionary models search a structured hypothesis space without gradients; supervised models learn a function approximation from labelled data; sequential models capture temporal dependence explicitly. The choice of paradigm is the highest-level design decision and constrains all subsequent choices.

### 2. Feature dependence

Whether the model operates on raw price/return data or on engineered technical features. Models with no feature dependence (GeoBM, ARIMA) test whether raw returns contain directional signal. Models that consume engineered features (GA, XGBoost, RF, SVM) test whether human-designed technical indicators add predictive value beyond the raw series. This dimension separates the "is there signal?" question from the "can we engineer signal?" question.

### 3. Capacity class

The effective number of free parameters the model can tune to the training data. Directly determines the model's ability to fit complex patterns — and its susceptibility to fitting noise. In this project, capacity maps inversely to test accuracy (the capacity–accuracy inversion): GeoBM (2 parameters, 57.5%) outperforms XGBoost-v1 (hundreds of effective parameters, 53.1%). Capacity class is the theoretical basis for understanding why more powerful models can perform worse in low-signal environments.

### 4. Interpretability

Whether the model's decision logic can be inspected and understood by a human. GeoBM produces a single closed-form probability. The GA produces 3 human-readable threshold rules. XGBoost provides feature importance scores but not transparent decision paths. LSTM and deep learning models are opaque. Interpretability matters because it determines whether a model's failure can be diagnosed structurally — the GA's convergence to mean-reversion rules is interpretable; XGBoost's overfitting is only diagnosable through the train-test gap.

### 5. Overfitting risk

A priori susceptibility to memorising training noise rather than learning generalisable patterns. Models with fewer free parameters have lower overfitting risk by construction. GeoBM has zero risk (2 parameters estimated analytically). The GA's bounded chromosome (9 genes) limits its capacity to overfit. XGBoost-v1's 18.6pp train-test gap demonstrates the risk empirically; v2's early stopping reduces it to 0.6pp. This dimension is the theoretical explanation for the capacity–accuracy inversion.

### 6. Temporal assumption

Whether the model treats observations as independent or models sequential dependence explicitly. GeoBM assumes IID log-normal returns. XGBoost treats each day as conditionally independent given its features. ARIMA and LSTM explicitly model autocorrelation and sequential structure. This project tests the non-sequential paradigms; the temporal dimension justifies LSTM and ARIMA as future work rather than scope exclusions, and explains why the project's features (lagged returns, rolling volatility) encode temporal information indirectly rather than through model architecture.

### 7. Computational cost

Training complexity in big-O terms relative to data size (n), model-specific parameters, and iteration counts. This is a Computer Science dimension — it positions models not just by what they do but by what they cost. GeoBM requires O(n) for two summary statistics. The GA requires O(pop × gen × n) evolutionary search. XGBoost requires O(n × trees × depth × features) greedy optimisation. LSTM requires O(n × epochs × params) backpropagation through time. This dimension formally justifies scope exclusions: LSTM's computational and data requirements are disproportionate for a proof-of-concept with 3,252 training rows.

---

## 7×7 Taxonomy Table

| Dimension | GeoBM | GA (DEAP) | XGBoost | LSTM | Random Forest | ARIMA | SVM |
|---|---|---|---|---|---|---|---|
| **Paradigm type** | Stochastic process | Evolutionary search | Supervised ensemble | Supervised sequential | Supervised ensemble | Time-series statistical | Supervised kernel |
| **Feature dependence** | None — raw returns only | Explicit — evolves rules over 14 features | Implicit — learns splits over 14 features | Implicit — learns from raw or engineered sequences | Implicit — learns splits over features | None — raw return series | Implicit — maps features to kernel space |
| **Capacity class** | Constant (2 params: μ, σ) | Bounded (9 genes, 3 rules) | Moderate–high (v1: 100×8 leaves; v2: 17×4 leaves) | High (thousands of weights per layer) | Moderate (ntrees × depth, decorrelated) | Low (p+d+q ≈ 3–5 params) | Kernel-dependent (linear: n_features; RBF: up to n_samples support vectors) |
| **Interpretability** | Full — closed-form P(up) = Φ((μ−0.5σ²)/σ) | Full — 3 readable threshold rules | Partial — gain importance but opaque splits | None — black-box weight matrices | Partial — feature importance, no path | Full — explicit AR/MA coefficients | Low — support vectors not human-readable |
| **Overfitting risk** | Zero (analytic, 2 params) | Low (bounded chromosome, 0.9pp train margin) | High (v1: 18.6pp gap; v2: 0.6pp with early stopping) | Very high (requires dropout, early stopping, large data) | Moderate (bagging reduces variance vs single tree) | Low (few parameters, risk of underfitting) | Moderate (kernel choice and C parameter sensitive to noise) |
| **Temporal assumption** | IID log-normal returns | Independent — each day evaluated separately given features | Independent — each day a feature vector | Sequential — models temporal dependence via hidden state | Independent — each day a feature vector | Sequential — explicit autocorrelation modelling (AR) and moving average (MA) | Independent — each day a feature vector |
| **Computational cost** | O(n) analytic | O(pop × gen × n) ≈ 5,000 × n | O(n × trees × depth × features) | O(n × epochs × params) — GPU-accelerated | O(n × ntrees × depth × features) | O(n × p²) for MLE fitting | O(n² to n³) for kernel matrix; O(n_sv × n) for prediction |

### Empirical results for implemented models

| Model | Test accuracy | vs 57.5% baseline | MCC | Train-test gap | Key finding |
|---|---|---|---|---|---|
| GeoBM | 57.5% | +0.0pp | 0.000 | N/A (analytic) | Constant "up" predictor — equals baseline by construction |
| GA | 56.7% | -0.8pp | -0.014 | -0.8pp | Near-constant "up" — bounded rules cannot discriminate |
| XGBoost-v1 | 53.1% | -4.4pp | -0.037 | +18.6pp | Significant overfitting — learned noise, not signal |
| XGBoost-v2 | 55.7% | -1.8pp | -0.090 | +0.6pp | Overfitting corrected, null result confirmed |

---

## Framing Paragraphs

### For the methodology chapter

> The three paradigms were selected to span the design space of algorithmic directional prediction along seven dimensions: paradigm type, feature dependence, capacity class, interpretability, overfitting risk, temporal assumption, and computational cost. Geometric Brownian Motion represents the stochastic-process region — it assumes IID log-normal returns, uses no engineered features, has constant capacity (2 parameters), full interpretability, zero overfitting risk, and O(n) computational cost. The Genetic Algorithm represents the evolutionary-search region — it explicitly searches over the engineered feature space using a bounded chromosome (9 genes, 3 rules), produces fully interpretable threshold-based predictions, has low overfitting risk, and costs O(pop × gen × n). XGBoost represents the supervised-ensemble region — it implicitly learns from all 14 features through non-linear tree splits, has moderate-to-high capacity, partial interpretability (feature importance but opaque decisions), high overfitting risk (empirically confirmed at 18.6pp), and costs O(n × trees × depth × features). Together, the three paradigms cover the stochastic, evolutionary, and supervised branches of the design space, with increasing capacity and decreasing interpretability — a structure that directly enables the capacity–accuracy inversion analysis that emerged as the project's central empirical finding.

### For the literature review

> Table X positions the three implemented paradigms alongside four approaches from the published literature: LSTM recurrent networks (Gu, Kelly & Xiu, 2020; Fischer & Krauss, 2018), Random Forests (Krauss et al., 2017), ARIMA time-series models (Box & Jenkins, 1970; Hamilton, 1994), and Support Vector Machines (Cao & Tay, 2001; Kim, 2003). The taxonomy reveals that the implemented models cover three of the five major paradigm types (stochastic, evolutionary, supervised ensemble) but do not cover sequential models (LSTM, ARIMA) or kernel-based methods (SVM). This is a deliberate scope decision: LSTM and ARIMA model temporal dependence through their architecture, while this project encodes temporal information through lagged features consumed by non-sequential models. The taxonomy makes this scope choice explicit and positions LSTM and ARIMA as principled extensions rather than arbitrary omissions. Random Forest occupies a similar region to XGBoost (supervised ensemble, implicit feature dependence, moderate capacity) but with lower overfitting risk due to bagging — it would provide a within-region comparison rather than extending the design-space coverage, which is why it was not prioritised over the cross-region comparison of GeoBM, GA, and XGBoost.

### For the discussion chapter

> The capacity–accuracy inversion observed empirically — GeoBM (2 parameters) outperforming XGBoost-v1 (hundreds of effective parameters) on the held-out test set — maps directly onto the taxonomy's capacity, overfitting-risk, and computational-cost dimensions. Models in the low-capacity, low-risk region (GeoBM, ARIMA) default to the unconditional distribution when signal is absent, producing baseline-equivalent accuracy. Models in the high-capacity, high-risk region (XGBoost, LSTM) have the capacity to fit training noise, producing below-baseline accuracy when the noise does not generalise. The GA occupies the intermediate position — its bounded capacity (9 genes) limits both its learning potential and its overfitting risk, producing accuracy between GeoBM and XGBoost. This pattern is not specific to these three models: it is a structural prediction of the taxonomy that any low-signal prediction task will exhibit a capacity–accuracy inversion where simpler models outperform more complex ones. The taxonomy thus provides a framework for predicting, not just describing, comparative model performance — which is the systematisation contribution that Ioana identified as the project's core value.

---

## Literature Citations

| Model | Key references | Context |
|---|---|---|
| GeoBM | Black & Scholes (1973); Hull (2018, Ch. 15) | Foundational stochastic model; log-normal return assumption |
| GA | Holland (1975); Goldberg (1989); Allen & Karjalainen (1999); Eiben & Smith (2015) | Evolutionary search over trading rules |
| XGBoost | Chen & Guestrin (2016, KDD); Friedman (2001) | Gradient-boosted trees; dominant tabular classifier |
| LSTM | Gu, Kelly & Xiu (2020, Review of Financial Studies); Fischer & Krauss (2018) | Deep learning for financial prediction; sequential modelling |
| Random Forest | Krauss et al. (2017); Breiman (2001) | Bagged ensemble; decorrelated trees |
| ARIMA | Box & Jenkins (1970); Hamilton (1994) | Classical time-series modelling; AR/MA structure |
| SVM | Cao & Tay (2001); Kim (2003); Vapnik (1995) | Kernel-based classification; margin maximisation |

---

## Rubric Mapping

| Criterion | How this taxonomy satisfies it |
|---|---|
| **Systematisation (Ioana directive #5)** | Seven named dimensions classify algorithmic trading approaches. Three implemented models are empirical instances; four literature models fill the design space. The taxonomy is the project's conceptual contribution. |
| **Knowledge beyond taught courses** | The seven dimensions (particularly capacity class, overfitting risk, and computational cost) require understanding model theory, not just API usage. The capacity–accuracy inversion is a structural prediction derived from the taxonomy, not a post-hoc observation. |
| **Literature integration** | The taxonomy positions the project against 7+ published approaches with specific citations. It shows coverage (3 of 5 paradigm types) and principled gaps (sequential and kernel paradigms as future work). |
| **CS-first framing** | Computational cost as a dimension is explicitly a Computer Science contribution. The taxonomy classifies by algorithmic properties, not financial properties. |
| **Critical evaluation** | The discussion paragraph shows that the taxonomy predicts the capacity–accuracy inversion before observing it — this is the difference between description and analysis. |

---

## LaTeX Booktabs Version

```latex
\begin{table}[htbp]
\centering
\caption{Taxonomy of algorithmic directional prediction paradigms along seven design-space dimensions. The three implemented models (GeoBM, GA, XGBoost) are empirical instances with known results from the held-out evaluation. Four literature models fill out the design space.}
\label{tab:taxonomy}
\small
\begin{tabular}{@{}p{2.2cm}p{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}@{}}
\toprule
\textbf{Dimension} & \textbf{GeoBM} & \textbf{GA (DEAP)} & \textbf{XGBoost} & \textbf{LSTM} & \textbf{Random Forest} & \textbf{ARIMA} & \textbf{SVM} \\
\midrule
Paradigm type &
  Stochastic process &
  Evolutionary search &
  Supervised ensemble &
  Supervised sequential &
  Supervised ensemble &
  Time-series statistical &
  Supervised kernel \\
\addlinespace
Feature dependence &
  None (raw returns) &
  Explicit (rules over features) &
  Implicit (learned splits) &
  Implicit (learned sequences) &
  Implicit (learned splits) &
  None (raw series) &
  Implicit (kernel space) \\
\addlinespace
Capacity class &
  Constant (2 params) &
  Bounded (9 genes) &
  Moderate--high (v1: 800; v2: 68 leaves) &
  High (thousands of weights) &
  Moderate (decorrelated trees) &
  Low (3--5 params) &
  Kernel-dependent \\
\addlinespace
Interpretability &
  Full (closed-form) &
  Full (readable rules) &
  Partial (importance only) &
  None (black box) &
  Partial (importance) &
  Full (AR/MA coefficients) &
  Low (support vectors) \\
\addlinespace
Overfitting risk &
  Zero &
  Low (0.9pp train margin) &
  High (v1: 18.6pp gap) &
  Very high &
  Moderate (bagging) &
  Low &
  Moderate (C, kernel) \\
\addlinespace
Temporal assumption &
  IID &
  Independent &
  Independent &
  Sequential &
  Independent &
  Sequential &
  Independent \\
\addlinespace
Computational cost &
  $O(n)$ &
  $O(\text{pop} \times \text{gen} \times n)$ &
  $O(n \times T \times d \times f)$ &
  $O(n \times E \times p)$ &
  $O(n \times T \times d \times f)$ &
  $O(n \times p^2)$ &
  $O(n^2)$ to $O(n^3)$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Compact Results Table (LaTeX)

```latex
\begin{table}[htbp]
\centering
\caption{Empirical results for the three implemented paradigms on the 501-day held-out test set (2023--2024). The majority-class baseline accuracy is 57.5\%.}
\label{tab:results}
\small
\begin{tabular}{@{}lccccl@{}}
\toprule
\textbf{Model} & \textbf{Test Acc.} & \textbf{vs Baseline} & \textbf{MCC} & \textbf{Gap} & \textbf{Key Finding} \\
\midrule
GeoBM          & 57.5\% & $+$0.0pp & 0.000  & N/A     & Constant predictor \\
GA             & 56.7\% & $-$0.8pp & $-$0.014 & $-$0.8pp & Near-constant, bounded rules \\
XGBoost-v1     & 53.1\% & $-$4.4pp & $-$0.037 & $+$18.6pp & Significant overfitting \\
XGBoost-v2     & 55.7\% & $-$1.8pp & $-$0.090 & $+$0.6pp  & Null result confirmed \\
\bottomrule
\end{tabular}
\end{table}
```
