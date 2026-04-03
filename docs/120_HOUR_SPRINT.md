# 120-Hour Sprint Plan — COM3001 Final Year Project
## A Systematised Comparative Backtesting Framework for Algorithmic Trading Strategies on Institutional Market Data

**Created:** 2026-04-03
**Deadline:** 2026-09-08 (22 weeks from now)
**Current state:** Full pipeline complete, XGBoost-v2 done, 11 component notes, zero dissertation prose written, vanilla LaTeX template only.

---

## Design Principles Behind This Plan

1. **Build evidence → write it up immediately.** Never separate code from prose by more than one session. The reasoning is freshest at the point of creation.
2. **Literature before methodology writing.** Every design choice in the methodology chapter needs a citation. Writing methodology without literature is writing without ammunition.
3. **Rubric-weighted time allocation.** 40% of marks come from Methodology & Implementation. 20% from Context. 20% from Evaluation. The hours reflect this.
4. **Interleaved not sequential.** Writing and coding alternate. This prevents the "built everything, forgot why" problem and produces tighter prose.
5. **Each block ends with a tangible deliverable.** No block produces only "progress" — each one produces a specific artefact that goes into the dissertation.

---

## Current Inventory (What Exists)

| Artefact | Status |
|---|---|
| Data pipeline (fetch → preprocess → target → features → split) | Complete, 5 modules, 5 notes |
| GeoBM baseline | Complete, note 07 |
| GA baseline | Complete, note 08 |
| XGBoost-v1 | Complete, note 09 |
| XGBoost-v2 (validated, early-stopped) | Complete, note 11 |
| Evaluation harness (evaluate.py) | Complete, note 10 |
| Figure generation (6 PDFs) | Complete |
| Component notes | 11/11 |
| LaTeX report | Vanilla Surrey template — zero project prose |
| Literature review | Not started |
| BibTeX entries | Zero project-relevant entries |
| Dissertation chapters | Not started |

---

## Hour Allocation Summary

| Block | Hours | Rubric Target | Deliverables |
|---|---|---|---|
| A. Quantitative Evaluation Arsenal | 18 | 40% Methodology + 20% Evaluation | 5 new scripts, 6 new tables/figures, taxonomy |
| B. Literature Foundation | 22 | 20% Context | 40–50 BibTeX entries, 4 lit review sections |
| C. LaTeX Setup + Introduction | 8 | 10% Report + 20% Context | Compilable report skeleton, introduction chapter |
| D. Data Pipeline + Methodology Chapters | 16 | 40% Methodology | 2 complete chapters (~14 pages) |
| E. Results + Discussion Chapters | 16 | 20% Evaluation | 2 complete chapters (~15 pages) |
| F. LSEP + Conclusion + Reflection | 7 | 10% LSEP + 20% Evaluation | 2 complete chapters (~5 pages) |
| G. Video + Demonstration | 5 | 10% Report & Video | Script, recording, demo prep |
| H. Integration + Polish + QA | 18 | All categories | Professional final draft |
| I. Contingency buffer | 10 | Risk management | Overrun absorption, examiner-proofing |
| **Total** | **120** | | |

---

## BLOCK A — Quantitative Evaluation Arsenal (18 hours)

**Why this comes first:** Every new analysis here produces a table, figure, or finding that feeds directly into the results and discussion chapters. Writing those chapters without this evidence would produce thin, descriptive prose. With it, you have the ammunition for critical analysis that earns first-class evaluation marks.

**Rubric justification:** "Appropriate type of evaluation such as... ablation study" (Evaluation criterion). "Knowledge beyond that covered in taught courses" (Methodology criterion).

### A1. Feature Ablation Study (4h)
**Code (2.5h) + Write-up (1.5h)**

Build `src/evaluation/feature_ablation.py`. Remove one feature category at a time (6 categories), retrain XGBoost-v2, record accuracy, balanced accuracy, MCC. Produce ablation table and bar chart.

Cross-paradigm check: compare ablation results with GA's evolved feature selections (GA converged to MA-ratio features). If ablation shows those features are dispensable, that's a cross-paradigm finding.

Immediately write the ablation paragraph for the evaluation chapter and extend note 10 or create note 12.

**Produces:** ablation script, summary table, bar chart PDF, ~300 words of evaluation prose, component note.

### A2. Bootstrap Confidence Intervals (2.5h)
**Code (1.5h) + Write-up (1h)**

Build bootstrap CI computation (1,000 resamples of 501 test predictions). 95% CIs for accuracy and MCC for all four models (GeoBM, GA, XGBoost-v1, XGBoost-v2). One summary table.

Key insight to extract: accuracy CIs for GeoBM and GA likely overlap, confirming McNemar's non-significance from a different statistical angle.

Immediately write the CI paragraph for the evaluation chapter.

**Produces:** CI table, ~200 words of evaluation prose.

### A3. Calibration and Threshold Analysis (3h)
**Code (2h) + Write-up (1h)**

Build `src/evaluation/calibration_analysis.py`. Brier score for XGBoost v1 and v2. Reliability diagram (calibration curve). Threshold sweep 0.3–0.7 (steps of 0.05), report accuracy/balanced accuracy/MCC at each.

Key question: does any threshold produce MCC > 0? If not, the model has no useful signal at any operating point.

Compute the naive Brier score (always predict base rate 0.575) = 0.244. If XGBoost Brier > 0.244, the probabilities are actively worse than naive.

**Produces:** calibration script, Brier scores, calibration curve PDF, threshold sensitivity figure, ~300 words.

### A4. Walk-Forward Evaluation (5h)
**Code (3.5h) + Write-up (1.5h)**

Build `src/evaluation/walk_forward.py`. Three rolling windows:

| Window | Train | Test | Purpose |
|---|---|---|---|
| 1 | 2010–2018 | 2019–2020 | Includes COVID crash |
| 2 | 2012–2020 | 2021–2022 | Post-COVID recovery |
| 3 | 2014–2022 | 2023–2024 | Existing holdout period |

Retrain all three models (GeoBM recomputes μ/σ, GA re-evolves, XGBoost-v2 re-tunes) on each window. Report accuracy and MCC across windows. 3×3 results table + stability figure.

This is beyond-taught-material work (López de Prado 2018, Pardo 2008). If the null result holds across all three windows, it is temporally robust. If any model beats baseline in one window but not others, that's regime-dependent performance — still informative.

**Produces:** walk-forward script, 3×3 table, stability figure, ~400 words, component note.

### A5. Systematisation Taxonomy Table (2.5h)

This is Ioana's explicit contribution suggestion. Build a formal taxonomy classifying algorithmic trading approaches by:
- Paradigm type (stochastic / evolutionary / supervised / deep learning)
- Feature dependence (none / explicit / implicit)
- Capacity class (constant / bounded / moderate / high)
- Interpretability (analytic / rule-based / feature-importance / opaque)
- Overfitting risk (none / low / moderate / high)
- Temporal assumption (IID / none / sequential)

Your three models are empirical instances. Add 4–5 literature models (LSTM, Random Forest, ARIMA, SVM) from published papers. Cite 8–10 sources.

This table goes into both the literature review (showing where approaches sit in the design space) and the methodology chapter (justifying why these three paradigms were chosen).

**Produces:** taxonomy table, ~500 words of methodology text, direct Ioana alignment.

### A6. Update Evaluation Harness + Regenerate All Figures (1h)

Update `evaluate.py` to include XGBoost-v2 in the comparison. Update `generate_figures.py` to produce updated figures showing all four models. Regenerate all PDFs. Update `results/tables/model_comparison.csv`.

**Produces:** updated evaluation output, refreshed figures, updated comparison CSV.

---

## BLOCK B — Literature Foundation (22 hours)

**Why this comes before chapter writing:** The methodology chapter says "as justified by [citation]" on every design choice. The discussion chapter says "consistent with [Author] (year) who found..." on every interpretation. Without the literature foundation, you cannot write either chapter to first-class standard.

**Rubric justification:** "Exceptional breadth and depth of literature" (Context criterion, 20%). "Critically evaluated" (first-class descriptor). Quality gate: 40–50 sources, 7 thematic buckets, 3,000–4,000 words.

### B1. Source Collection — 40–50 Papers (8h)

Systematic search across 7 thematic buckets via Google Scholar. Download PDFs (or note access method), create BibTeX entries. Prioritise Tier 1 journals over MDPI.

**Bucket 1: Market efficiency and predictability**
Fama (1970), Lo (2004) AMH, Malkiel (2003), Lo & MacKinlay (1988), Gu Kelly & Xiu (2020). 6–8 papers.

**Bucket 2: Technical analysis and price-volume features**
Brock Lakonishok & LeBaron (1992), Lo Mamaysky & Wang (2000), Park & Irwin (2007). 4–6 papers.

**Bucket 3: Stochastic models in finance**
Hull (2018), Black & Scholes (1973), Merton (1973), Geometric Brownian Motion applications. 4–5 papers.

**Bucket 4: Evolutionary computation in trading**
Allen & Karjalainen (1999), Chen (2002), Neely et al. (1997), DEAP framework. 4–6 papers.

**Bucket 5: Gradient boosting and ML for financial prediction**
Chen & Guestrin (2016), Friedman (2001), Krauss et al. (2017), Zhong & Enke (2019). 5–7 papers.

**Bucket 6: Backtesting methodology and pitfalls**
López de Prado (2018), Bailey et al. (2014), Pardo (2008). 4–5 papers.

**Bucket 7: Evaluation metrics for financial classifiers**
Dietterich (1998), Sokolova & Lapalme (2009), McNemar (1947), Chicco & Jurman (2020) on MCC. 4–5 papers.

**Produces:** 40–50 BibTeX entries in `references.bib`, organised by bucket with comment headers.

### B2. Lit Review Section 1 — Market Efficiency and Predictability (3.5h)

Fama (1970) EMH taxonomy → Lo (2004) AMH as evolution → Malkiel (2003) random walk → Lo & MacKinlay (1988) departures → Gu Kelly & Xiu (2020) ML predictability limits.

Critical evaluation pattern: "Author found X. However, conditions Y differ from the present study in Z." Position your null result within the EMH debate. Your result is consistent with weak-form efficiency but does not prove it — scope the claim precisely.

**Produces:** `literature.tex` section 1 (~800–1,000 words).

### B3. Lit Review Section 2 — Technical Features and Price-Volume Analysis (3h)

Brock et al. (1992) on trading rules → Lo et al. (2000) on patterns → Park & Irwin (2007) survey. Critically evaluate: early studies found profitability but most pre-date transaction costs and data-snooping corrections. Your feature set draws from this tradition but results show these features are insufficient for daily SPY direction.

**Produces:** `literature.tex` section 2 (~700–800 words).

### B4. Lit Review Section 3 — Modelling Paradigms (4h)

Three subsections:
(a) GeoBM / Hull (2018), Black-Scholes (1973), stochastic processes as null models
(b) GA / Allen & Karjalainen (1999), Chen (2002), Neely et al. (1997), evolutionary computation in finance
(c) XGBoost / Chen & Guestrin (2016), Friedman (2001), Krauss et al. (2017)

For each: what the literature claims vs what your results show. Integrate the taxonomy table from A5.

**Produces:** `literature.tex` section 3 (~1,000–1,200 words).

### B5. Lit Review Section 4 — Evaluation Methodology and Pitfalls (3.5h)

López de Prado (2018) on backtesting pitfalls → Bailey et al. (2014) on overfitting → Pardo (2008) on evaluation design → Dietterich (1998) on classifier comparison → Sokolova & Lapalme (2009) on metrics.

This section justifies your entire evaluation framework: temporal split, McNemar's test, MCC, balanced accuracy, walk-forward validation. Position your framework as addressing the pitfalls the literature warns about.

**Produces:** `literature.tex` section 4 (~800–1,000 words).

---

## BLOCK C — LaTeX Setup + Introduction (8 hours)

### C1. Set Up LaTeX Project with Handbook-Compliant Structure (2h)

The existing template is vanilla Surrey — sample content, placeholder metadata. Restructure:
- Update `finalreport.tex` with correct title, author (Shomique Hayat), URN, supervisor (Ioana Boureanu), date
- Create chapter files: `chapter1.tex` (intro), `chapter2.tex` (lit review), `chapter3.tex` (data pipeline), `chapter4.tex` (methodology), `chapter5.tex` (results), `chapter6.tex` (discussion), `chapter7.tex` (LSEP + conclusion)
- Replace sample `references.bib` with the real one from B1
- Copy all results/figures/ PDFs into report Figures/ directory
- Verify compilation with `make`
- Add packages needed: booktabs (tables), listings (code), algorithm2e or algorithmicx (pseudocode), natbib adjustments if needed

**Produces:** compilable LaTeX skeleton with correct metadata and all figures available.

### C2. Transfer Ready-to-Lift Blocks from Component Notes (2h)

Every component note has "ready-to-lift wording" blocks. Paste:
- Notes 01–06 methodology blocks → `chapter3.tex` (data pipeline)
- Notes 07–09, 11 methodology blocks → `chapter4.tex` (methodology)
- Note 10 evaluation blocks → `chapter5.tex` (results)

This gives ~12–15 pages of first-draft prose immediately. These are seeds — they need expansion, transitions, figure references, and citation anchoring. But they eliminate the blank page problem.

**Produces:** ~15 pages of draft content across 3 chapters.

### C3. Write Introduction Chapter (4h)

Widom framing (from rubric-criteria.md quality gate):
1. What is the problem? Daily equity directional prediction — a fundamental test of whether computational approaches extract signal from historical price/volume data.
2. Why is it interesting? Intersection of EMH and applied ML. Tests whether algorithmic sophistication overcomes market efficiency.
3. Why is it hard? Low signal-to-noise at daily frequency. Non-stationarity. Overfitting is the central methodological danger.
4. Why do naive approaches fail? GeoBM degenerates to constant predictor. Simple rules (GA) converge to near-constant. Even tuned ML (XGBoost-v2) cannot beat the baseline.
5. What is your approach? Three-paradigm controlled comparison. Same data, same target, same split, same metrics. Contribution = the framework + honest null result interpretation.
6. Aims, objectives, scope limitations.

Quality gate: 2,500–3,000 words, all six Widom questions answered, CS-first framing, contribution stated explicitly.

**Produces:** `chapter1.tex` (4–5 pages).

---

## BLOCK D — Data Pipeline + Methodology Chapters (16 hours)

### D1. Write Data and Pipeline Chapter (4h)

- WRDS/CRSP justification (institutional data, survivorship-bias-free, academic standard)
- SPY selection and PERMNO resolution
- CRSP conventions (negative prices = bid/ask average, cumulative adjustment factors)
- Preprocessing steps with row-count evidence
- Target construction: binary direction, why not regression, shift mechanics
- Feature engineering: 6 categories, 14 features, what was included and what was excluded and why
- Temporal split: chronological, why not random, why not k-fold, embargo considerations
- Pipeline flow diagram (from note 04)
- Anti-leakage checkpoints at every stage

Reference all component notes. Cite WRDS documentation, Hull for CRSP conventions, López de Prado for temporal split justification.

Quality gate: part of 5,000–6,000 word methodology total.

**Produces:** `chapter3.tex` (5–6 pages).

### D2. Write Methodology — Experimental Design Section (2.5h)

The three-paradigm comparison as a controlled experiment:
- Controlled variables: same data, same split, same target, same evaluation metrics
- Independent variable: modelling paradigm (stochastic / evolutionary / supervised)
- Dependent variable: test-set directional accuracy and MCC
- The taxonomy table from A5, positioning your three models in the design space
- Why three paradigms rather than one or ten (coverage of the design space with tractable scope)
- CRISP-DM as the overarching methodology framework

**Produces:** `chapter4.tex` section 1 (2–3 pages).

### D3. Write Methodology — Model Sections (6.5h)

**GeoBM (1.5h):** Analytic baseline derivation. Parameter estimation from training data. Drift correction (μ − 0.5σ²). Why P(up) = Φ((μ − 0.5σ²)/σ) = 0.5166. Constant-predictor degeneration under positive drift. What this model does and does not claim. Cite Hull (2018), Black-Scholes (1973).

**GA (2h):** Chromosome design (3 rules × 3 genes). DEAP framework. Majority-vote aggregation. Evolved rule interpretation (mean-reversion via MA ratios). Capacity limitation (bounded rule space). Why the GA converged to near-constant "up" — the rules discovered real patterns but the bounded representation cannot exploit them beyond the majority class. Cite Allen & Karjalainen (1999), DEAP documentation.

**XGBoost v1 + v2 (2h):** V1 as fixed-defaults baseline. Overfitting diagnosis (18.6pp gap). V2 as methodological response: chronological validation design, early stopping mechanism, grid search over 18 combinations, best configuration (depth=2, 17 trees, lr=0.1). V1-vs-v2 comparison table. The null result survives refinement. Cite Chen & Guestrin (2016), Friedman (2001).

**Evaluation harness (1h):** Metrics selection and justification (accuracy, balanced accuracy, MCC, McNemar's). Why MCC over F1 for imbalanced binary classification (Chicco & Jurman 2020). Why McNemar's over paired t-test (Dietterich 1998). Common prediction schema enabling fair comparison.

**Produces:** `chapter4.tex` sections 2–5 (8–10 pages).

### D4. Build All Results Tables in LaTeX (3h)

- Three-paradigm comparison table (GeoBM, GA, XGBoost-v1, XGBoost-v2 — all metrics)
- V1-vs-v2 XGBoost comparison table
- Ablation table (from A1)
- Walk-forward stability table (from A4)
- McNemar's test results table
- Bootstrap CI table (from A2)
- Calibration / Brier score table (from A3)
- Threshold sensitivity table (from A3)
- Taxonomy table (from A5)

Each table with `\caption{}`, `\label{}`, and cross-referenced from text. Use `booktabs` package for professional formatting.

**Produces:** 8–9 LaTeX tables, dissertation-ready.

---

## BLOCK E — Results + Discussion Chapters (16 hours)

### E1. Write Results Chapter (5h)

Structure: present evidence systematically, interpret minimally (save interpretation for discussion).

1. Three-paradigm headline comparison: accuracy, balanced accuracy, MCC for all models. Majority baseline as reference line.
2. V1-vs-v2 XGBoost comparison: overfitting gap collapse, accuracy improvement, null result persistence.
3. Ablation results: which feature categories contributed, cross-reference with GA's evolved features.
4. Calibration analysis: Brier scores, reliability diagram interpretation, threshold sensitivity.
5. Walk-forward stability: does the null result hold across market regimes?
6. Bootstrap CIs: quantified uncertainty around all accuracy estimates.
7. Capacity–accuracy inversion as the central empirical pattern: more capacity → worse generalisation.

Every figure referenced and interpreted. Every table explained. No orphan figures.

Quality gate: 2,500–3,000 words, all evaluation criteria addressed.

**Produces:** `chapter5.tex` (7–8 pages).

### E2. Write Discussion Chapter (7h)

This is the chapter that separates first-class from upper-second. Five analytical threads:

**Thread 1 — The capacity–accuracy inversion (1.5h):**
More model capacity produced worse generalisation. GeoBM (2 params, constant predictor) matched baseline. GA (bounded rules) fell slightly below. XGBoost-v1 (100 trees, depth 3) fell well below. XGBoost-v2 (17 trees, depth 2, regularised) recovered partially. Explain why: in a low-signal environment, additional parameters fit noise. Cite Bailey et al. (2014), López de Prado (2018).

**Thread 2 — Relationship to EMH (1h):**
Null result consistent with weak-form efficiency on this dataset. Scope precisely: one asset (SPY), one feature set (14 technical indicators), one horizon (daily), one period (2023–2024). Cannot claim "markets are efficient." Can claim: "three paradigms of increasing sophistication could not falsify weak-form efficiency on this dataset." Cite Fama (1970), Lo (2004). Lo's AMH suggests predictability may be time-varying — walk-forward results speak to this.

**Thread 3 — Comparison with published results (1.5h):**
Krauss et al. (2017): deep nets + gradient boosting on S&P 500 constituents found modest profitability — but different frequency, features, evaluation protocol. Gu et al. (2020): ML has modest predictive power for monthly returns, much less for daily. Your daily-frequency null result is consistent with Gu et al.'s finding. Zhong & Enke (2019): higher accuracy reported but with different feature sets and potentially looser evaluation. Position your result honestly within this landscape.

**Thread 4 — What the framework reveals that individual studies cannot (1.5h):**
A study testing only XGBoost would conclude "XGBoost overfit." Your framework shows the failure is not paradigm-specific — every paradigm converges to the same strategy (predict majority class). The failure is about the feature set × asset × frequency combination, not about any particular model. This comparative insight is the systematisation contribution Ioana asked for.

**Thread 5 — Limitations as bounded design choices (1.5h):**
Single holdout addressed by walk-forward. Mild class imbalance (57.5/42.5). 14 features are a subset — sentiment, macro, options-implied volatility excluded by design (proof-of-concept scope, not exhaustive search). 501 test observations limit statistical power. Post-GFC training period biases all models toward "up." These are not apologies — they are design boundaries that define the scope of the conclusion and point toward future work.

**Produces:** `chapter6.tex` (7–8 pages).

### E3. Write Abstract (1h)

250–300 words. Problem, approach, three paradigms, null result, capacity–accuracy inversion, contribution (the systematised framework, not individual model accuracy). No jargon. Comprehensible standalone.

**Produces:** `abstract.tex`.

### E4. Regenerate and Verify All Figures (3h)

Update `generate_figures.py` to include:
- Ablation bar chart
- Calibration curve
- Threshold sensitivity plot
- Walk-forward stability chart
- Updated 4-model comparison charts

Copy all PDFs to LaTeX Figures/ directory. Verify every `\includegraphics` compiles. Check captions are descriptive and labels are referenced.

**Produces:** 10+ figures integrated into LaTeX with professional captions.

---

## BLOCK F — LSEP + Conclusion + Reflection (7 hours)

### F1. Write LSEP Chapter (3h)

Every point must be tailored to THIS project — generic LSEP scores poorly.

- **Data licensing (Legal):** WRDS CRSP institutional licence, redistribution prohibition, raw data excluded from repo via .gitignore. Another researcher needs own WRDS access.
- **Reproducibility (Professional):** Config-driven (seed 42, YAML paths), logged diagnostics, version-controlled, README documentation. Higher reproducibility standard than most published financial ML papers.
- **Responsible claims (Ethical):** Null result reported honestly. No overclaiming predictive ability. Proof-of-concept scope. Bailey et al. (2014) on dangers of overfitting and false discovery in financial ML.
- **Algorithmic trading implications (Social):** Market fairness, flash crashes, access inequality. Your framework is for research evaluation, not live trading.
- **Training-period bias:** Post-GFC bull market biases all models toward "up." Acknowledged honestly.
- **UN SDGs:** SDG 8 (Decent Work and Economic Growth — responsible financial technology), SDG 10 (Reduced Inequalities — algorithmic trading and market access). The finding that simple technical features don't predict direction suggests this type of analysis does not provide unfair advantage.

Quality gate: 1,000–1,500 words, tailored, SDG linked.

**Produces:** LSEP section within `chapter7.tex` (3–4 pages).

### F2. Write Conclusion and Future Work (2.5h)

- Summary of contributions: systematised framework, three-paradigm null result, capacity–accuracy inversion as diagnostic pattern, v1-vs-v2 as methodological refinement demonstration
- What was achieved: working comparative pipeline, honest null result, methodological discipline
- What was not achieved: walk-forward across more windows, alternative assets, different feature families, SHAP analysis, nested cross-validation
- Future work: multi-asset extension, intraday frequency, alternative features (sentiment, macro, options-implied), SHAP-based feature selection, formal Bayesian hyperparameter optimisation

Quality gate: answers "what has been achieved and not achieved" (evaluation criterion).

**Produces:** conclusion section within `chapter7.tex` (2–3 pages).

### F3. Write Personal Reflection (1.5h)

First-person, honest, not defensive. What you learned about:
- Research methodology (the gap between training accuracy and real predictive value)
- Financial ML (why most published results don't replicate)
- Software engineering discipline (config-driven pipelines, anti-leakage checkpoints, component documentation)
- What you would do differently (feature selection via SHAP before model fitting, start walk-forward earlier, use nested CV from the beginning)

This directly addresses evaluation bullet 6: "personal reflection of what has been achieved and not achieved."

**Produces:** 1–1.5 pages within `chapter7.tex`.

---

## BLOCK G — Video + Demonstration (5 hours)

### G1. Plan and Script the Video (1.5h)

Structure (≤5 minutes total):
- What the project is (30s): three-paradigm comparison, not a trading bot
- The three paradigms and why (45s): stochastic / evolutionary / supervised, covering the design space
- Pipeline demo (30s): show terminal running fetch → preprocess → features → split
- Models running (30s): show XGBoost-v2 grid search output, early stopping
- Evaluation harness (30s): show evaluate.py output, comparison table
- Key figures (30s): capacity–inversion chart, ablation results
- Results and what they mean (45s): null result, capacity–accuracy inversion, what the framework contributes
- What was learned (30s): honest reflection

Write the script word-for-word. Keep it tight — no waffle.

**Produces:** video script (~800 words).

### G2. Record and Edit the Video (2.5h)

Screen recording of pipeline executing, evaluation output, figures. Voiceover. Show that the code works (rubric requirement). Professional quality — clear audio, no background noise, readable terminal text.

**Produces:** project video (MP4, H.264, <500MB, ≤5 minutes).

### G3. Prepare Supervisor Implementation Demonstration (1h)

The handbook requires demonstrating your implementation to your supervisor near submission. Prepare:
- Run full pipeline end-to-end
- Show evaluation harness output
- Show v1-vs-v2 comparison
- Show walk-forward results
- Prepared answers for: "Why did nothing beat the baseline?" and "What would you do differently?"

**Produces:** demonstration preparation notes.

---

## BLOCK H — Integration + Polish + QA (18 hours)

### H1. Full Structural Proofread — Argument Thread (3.5h)

Read the entire dissertation start to finish. Check the argument thread:

Introduction sets up the question → Literature review shows what has been tried and what gaps exist → Data pipeline shows how evidence was prepared → Methodology shows how the experiment was designed → Results show what happened → Discussion explains what it means → Conclusion answers the question from the introduction.

Every chapter must reference the ones before and after. If any chapter feels disconnected, add a transition paragraph. Cut any padding. Add missing cross-references.

**Produces:** structural revision.

### H2. Figure, Table, Caption, and Reference Audit (3h)

- Every figure has a `\caption{}` and is `\ref{}`d from text
- Every table has a `\caption{}` and is `\ref{}`d from text
- Every `\cite{}` in text appears in `references.bib`
- Every BibTeX entry is cited at least once (no orphan references)
- Consistent citation style throughout (Harvard/author-date via `agsm`)
- All figures render at correct size and resolution
- All tables fit within page margins

**Produces:** clean bibliography and cross-references.

### H3. Language, Grammar, and Formatting Pass (3h)

- British English throughout (organise, analyse, colour, behaviour, modelling)
- No typos, no placeholder text, no "TODO" markers
- Consistent terminology enforcement (GeoBM never GBM, temporal split never random split)
- Consistent tense (past tense for methods and results, present for general truths)
- Check page count against 40–50 page target
- Check formatting against handbook requirements (margins, font, spacing)
- Remove all template sample content

**Produces:** polished final draft.

### H4. Component Notes vs Dissertation Consistency Check (2h)

Check every component note against the corresponding dissertation section:
- Are the notes still accurate after all revisions?
- Did the dissertation introduce any claims not supported by the notes?
- Are exact numbers consistent between notes, code output, and dissertation text?
- Do the ready-to-lift blocks match what ended up in the chapters?

**Produces:** consistency verification.

### H5. Code Archive Preparation (3h)

The handbook says: "archive contains enough information so that your examiners have the information needed to run any programs."

- Clean the repo: remove any debug files, ensure .gitignore is comprehensive
- Verify `requirements.txt` is accurate (run `pip freeze > requirements_frozen.txt` as backup)
- Ensure README explains: how to install, how to configure WRDS access, how to run each stage
- Create `run_pipeline.sh` that executes every stage in order (demonstrates professionalism)
- Decision: include frozen copies of `spy_train.csv` and `spy_test.csv` in archive so examiner can run stages 5–10 without WRDS access. Document this in README.
- Verify seed 42 produces identical results on clean install

**Produces:** submission-ready code archive.

### H6. Bibliography Completion and Verification (2h)

- Verify every BibTeX entry has complete metadata (authors, year, journal/conference, volume, pages, DOI)
- Check for consistency in author names, journal abbreviations, capitalisation
- Verify DOIs resolve correctly
- Ensure minimum 40 references, ideally 50+
- Check source quality distribution: majority Tier 1/2, minimal Tier 3

**Produces:** verified bibliography.

### H7. Final Submission Checklist (1.5h)

Check every deliverable against the handbook:
- [ ] Final report PDF (compiled, no LaTeX errors, no overfull hboxes)
- [ ] Code archive (zip, all files present, README, requirements, config)
- [ ] Video (MP4, H.264, ≤5 minutes, <500MB)
- [ ] Ethics evidence (if applicable)
- [ ] All submissions go through SurreyLearn
- [ ] Nothing is late (deadline: 8 September 2026)
- [ ] Supervisor demonstration completed
- [ ] CLAUDE.md updated with final state

**Produces:** submitted project.

---

## BLOCK I — Contingency Buffer (10 hours)

These hours are not pre-assigned. They absorb overruns from the most likely risk areas:

| Risk | Likelihood | Likely overrun | Mitigation |
|---|---|---|---|
| Walk-forward (A4) takes longer than 5h | Medium | +2h | Stop at 6h, simplify to 2 windows |
| Literature sourcing takes longer than 8h | High | +3h | Prioritise Tier 1 sources, accept 35 instead of 50 |
| Discussion chapter is hard to write well | High | +2h | Use the 5-thread structure as scaffolding |
| LaTeX compilation issues | Medium | +1h | Resolve formatting incrementally |
| Examiner-proofing: re-running code to verify all numbers match | Medium | +2h | Do this in H4 |

If no overruns occur, spend remaining buffer hours on:
1. Additional walk-forward windows (extend A4)
2. SHAP feature importance analysis (adds one more evaluation dimension)
3. Deeper literature engagement (more Tier 1 sources)

---

## Recommended Execution Schedule

This is a suggested weekly mapping assuming 15–20 hours/week across 7 weeks. Adjust to your actual availability.

| Week | Dates (approx) | Blocks | Hours | Key deliverables |
|---|---|---|---|---|
| 1 | Apr 3–9 | A (all) | 18 | Ablation, CIs, calibration, walk-forward, taxonomy, updated figures |
| 2 | Apr 10–16 | B1–B3 | 14.5 | 40–50 BibTeX entries, lit review sections 1–2 |
| 3 | Apr 17–23 | B4–B5, C1–C2 | 11.5 | Lit review sections 3–4, LaTeX skeleton, block transfer |
| 4 | Apr 24–30 | C3, D1–D2 | 10.5 | Introduction chapter, data pipeline chapter, experimental design |
| 5 | May 1–7 | D3–D4 | 9.5 | All model methodology sections, all LaTeX tables |
| 6 | May 8–14 | E1–E2 | 12 | Results chapter, discussion chapter |
| 7 | May 15–21 | E3–E4, F (all) | 11.5 | Abstract, all figures, LSEP, conclusion, reflection |
| 8 | May 22–28 | G (all), H1–H3 | 14.5 | Video, structural proofread, language pass |
| 9 | May 29–Jun 4 | H4–H7, I | 18.5 | Consistency check, code archive, bibliography, submission checklist, buffer |

**This completes the 120-hour sprint by early June — 3 full months before the September 8 deadline.** The remaining 14 weeks are available for Ioana's feedback cycles, second-draft revisions, and final polish. That buffer is not luxury — it is where first-class reports are made.

---

## What This Plan Does NOT Include (By Design)

- Dashboard or visualisation UI (Ioana said no bolt-ons; the figures are sufficient)
- Additional models beyond the locked three (scope discipline)
- Live trading simulation or PnL computation (proof of concept, not trading bot)
- Sentiment analysis, macro features, or alternative data (excluded by design; mentioned in future work)
- SHAP analysis (desirable but optional; lives in buffer if time permits)

---

## First Move

Action 1 (XGBoost-v2) is complete. **Start A1 (Feature Ablation) now.** It is the highest-value remaining implementation task — the rubric literally names it.
