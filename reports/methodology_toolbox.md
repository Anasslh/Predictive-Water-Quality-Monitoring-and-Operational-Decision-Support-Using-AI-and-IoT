# Methodology Toolbox — Implementation Notes
**Project:** Predictive Water Quality Monitoring and Operational Decision Support Using AI and IoT
**Status:** Internal notes, to consult during weeks 3–6 — **NOT water literature, do not cite as NILES related work**
**Inspiration source:** Synthesis of Boujamza Abdeltif thesis (industrial PHM + port logistics — outside the water domain, methodological use only)

---

## Why this document exists

This is not a reading summary in the usual format (S/A/N/B). The source thesis does not cover water, environmental IoT, or XAI applied to monitoring — it must therefore **never appear in the literature matrix or the NILES paper** as related work. However, certain methodological choices it illustrates are directly transferable to our implementation decisions (weeks 3–6). This document captures those ideas so they are not lost before then.

---

## High-relevance ideas — to adopt concretely

### 1. Semi-AutoML (broad screening → manual fine-tuning)
**Where to use:** weeks 3–4, multi-model comparison per parameter.

Our "one independent model per parameter" architecture means comparing several algorithms (RF, XGBoost, SVR, ANN...) for each of our targets (pH, EC, turbidity) — potentially 3 targets × several models. Rather than doing this entirely by hand as in S-6/A-1/N-5:
- **Phase 1 (automated):** broad screening via a tool like PyCaret — evaluate ~10–12 models in parallel per target with objective metrics (RMSE, MAE, R²).
- **Phase 2 (manual):** targeted fine-tuning of the 2–3 best models per target (grid/random search or Bayesian optimization).

This accelerates the work without changing our systematic comparison philosophy already established in the literature review.

### 2. 3-block split (train / validation / test)
**Where to use:** from week 3, as soon as the C-1 dataset is ready for feature engineering.

Many of our papers (S-6, A-1, N-5) use a 2-block split (train/test) with CV for tuning — which conflates tuning and final evaluation. An explicit 3-block split is more rigorous:
- **Train:** model training
- **Validation:** hyperparameter tuning
- **Test:** reserved exclusively for final evaluation, never touched during optimization

With only 365 rows (C-1 dataset), plan a careful chronological split (e.g. 70/15/15) to respect the temporal nature of the data — no random split that would mix future and past.

### 3. Bayesian optimization (alternative to grid search)
**Where to use:** week 5 (model optimization).

If we tune 3 targets × several models each, exhaustive repeated grid search can become costly. Bayesian optimization scales better for larger search spaces. Keep as an option if classic grid search becomes too slow.

### 4. Asymmetric score/loss function
**Where to use:** week 6 (anomaly detection module), potentially also in the operational decision layer (IE).

Idea: penalize a **missed or late degradation alert** more severely than an early false alarm — logic directly transposable to water quality (a late alert on contamination is more serious than an early false alarm). A relevant alternative to symmetric classic metrics (RMSE/MAE) for this specific project component.

---

## Low-relevance or conflicting ideas — do not adopt by default

### 5. Heavy deep learning architectures (NHITS, DLSTM+A)
Calibrated for much larger data volumes (hourly/15-min data over multiple years, hundreds of instances). With 365 daily observations on a single site, strong overfitting risk. Simpler LSTMs already seen in the water literature (S-6, A-3) remain the appropriate reference for our current scale.

### 6. Ensemble learning (multi-model aggregation)
Often improves raw accuracy, but complicates per-model SHAP explainability — in direct tension with our native XAI per parameter pillar (§2.3 S1 report). Only worth considering if we explicitly want to discuss this accuracy/explainability trade-off in our NILES methodology.

### 7. Feature selection by histograms
Useful only once the 4 raw parameters have been enriched with engineered features (lags, rolling averages, etc.) in week 3. On current raw data, limited immediate added value — revisit after temporal feature engineering.

---

*Internal working document — not to be confused with `week2_report.md` (water literature review, citable in NILES).*
