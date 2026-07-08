# Implementation Plan — Predictive Water Quality Monitoring
**SESC, Nile University — CS Role**
**Status:** End of week 2 (lit review + dataset decided) → implementation starting week 3
**To share with the team (CS + IE)**

---

## 0. Recap of decisions already made (S1/S2 reports)

This plan does not invent anything new — it operationalizes what has already been decided:

| Decision | Source | Practical consequence |
|---|---|---|
| One independent model per parameter | S1 §2.2 | 3 parallel modeling pipelines (pH, EC, turbidity — TDS removed from scope, team decision) |
| No WQI aggregation on CS side | S1 §2.3 | The WQI/Class from C-1 (inconsistent, cf. S2 §5.1) is **not our target** — we predict the 3 raw parameters, period. The composite WQI is an IE problem, not CS. |
| Native XAI (SHAP) per parameter | S1 §2.3 | Each model outputs a prediction + a local SHAP explanation |
| Batch + quasi real-time inference | S1 §2.1 | Classic training, inference at each new sensor reading, 24h horizon (imposed by C-1 daily granularity) |
| Scheduled retraining (weekly) | S1 §2.4 | Reusable callable function, not an ad hoc notebook |
| C-1 = main dataset | S2 §5.3 | Kaggle/CPCB remains a reference for calibrating standards (Si), not a training source |

---

## 1. GitHub Repository Architecture

Designed so team members can work in parallel without stepping on each other: each parameter has its own isolated folder, and a shared base class ensures the output format (prediction + SHAP) stays identical regardless of the algorithm each person chooses.

```
water-quality-cs/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                     # data.xlsx (C-1), CPCB (reference) — versioned, small volume
│   ├── processed/               # cleaned datasets + features (gitignored if large)
│   └── external/                # Kaggle/CPCB for Si calibration
│
├── notebooks/
│   ├── eda_ph.ipynb             # scratch — 1 notebook per parameter, owned by whoever models it
│   ├── eda_ec.ipynb
│   └── eda_turbidity.ipynb
│
├── src/
│   ├── data/                    # ingestion, cleaning, feature engineering, 3-block split
│   │   ├── ingest.py
│   │   ├── clean.py
│   │   └── features.py
│   │
│   ├── models/
│   │   ├── base.py              # abstract class ParameterModel (fit/predict/explain) — SHARED CONTRACT
│   │   ├── ph/                  # models + comparison for pH
│   │   ├── ec/                  # same for EC
│   │   └── turbidity/           # same for turbidity
│   │
│   ├── xai/
│   │   └── shap_wrapper.py      # standardized SHAP wrapper, shared across all 3 models
│   │
│   ├── anomaly/
│   │   └── detector.py          # anomaly detection module (week 4-5)
│   │
│   └── evaluation/
│       ├── metrics.py           # standard RMSE/MAE/R²
│       └── asymmetric_score.py  # metric penalizing late alerts (anomaly detection)
│
├── configs/
│   └── ph.yaml / ec.yaml / turbidity.yaml   # hyperparameters per parameter
│
├── models_store/                # trained models (.joblib) — gitignored, or Git LFS if needed
│
├── tests/                       # unit tests, notably for base.py and shap_wrapper.py
│
└── reports/                     # existing .md reports (S1, S2, plan, toolbox)
```

**The key point: `src/models/base.py`.** It is an abstract class with 3 mandatory methods (`fit`, `predict`, `explain`) that all parameter subfolders must implement, regardless of the algorithm chosen (RF for pH, XGBoost for turbidity, etc.). This ensures `shap_wrapper.py` works across all models without parameter-specific code, and that the output format stays consistent across all parameters.

---

## 2. Detailed Pipeline, Step by Step

**Step 1 — Ingestion & cleaning**
Load `data.xlsx` (C-1). The file contains pH/TDS/EC/Turbidity, but **TDS is removed from the modeling scope** (team decision) — only pH/EC/Turbidity are targets; TDS can stay as a potential input feature if useful. Keep raw WQI/Class in a separate column, not used as a target (just for reference/future comparison if the IE team needs it).

**Step 2 — EDA**
- Temporal decomposition per parameter (trend, weekly/monthly seasonality — no exploitable annual seasonality with only 1 year)
- Stationarity tests (ADF) per parameter — conditions model choice (differencing needed or not)
- ACF/PACF to calibrate lags to create
- Correlation matrix between the 4 parameters

**Step 3 — Feature engineering**
Per parameter: lags (D-1, D-2, D-7), rolling means/standard deviations (7d, 14d), day-of-week/month indicators. Feature selection by histograms (toolbox #7) only at this stage, once these features are created.

**Step 4 — Chronological 3-block split**
70% train / 15% validation / 15% test, **never random** (respect temporal order). See toolbox #2.

**Step 5 — Baseline modeling per parameter**
For each of the 3 parameters: compare 3–4 models (RF, XGBoost, SVR, optionally LSTM if volume allows). Either manually or via semi-AutoML (PyCaret phase 1) for speed — see toolbox #1. Consistent with the literature review finding: no universally dominant model, so systematic comparison is mandatory.

**Step 6 — Optimization**
Fine-tuning of the 2–3 best models per parameter (grid/random search, or Bayesian if compute becomes a bottleneck — toolbox #3).

**Step 7 — SHAP integration**
Standardized wrapper: each final model produces `(predicted_value, top_SHAP_features)`. Identical format for all 3 parameters.

**Step 8 — Anomaly detection**
Separate module from the prediction models (not just "high prediction error" — a dedicated detector, e.g. on residuals or statistical thresholds). Evaluation with an asymmetric metric (penalize late detection more than early false alarm — toolbox #4).

**Step 9 — Integration & end-to-end testing**

**Step 10 — Technical report + NILES draft**

---

## 3. Timeline — implementation + reading in parallel

Reading (weeks 3–8 of the programme) does not need to be finished before coding — it informs decisions as we go.

| Week | Implementation | Parallel reading |
|---|---|---|
| 3 | Cleaning, EDA, feature engineering | A-5→B-1 (hybrid architectures, diagonal) |
| 4 | Baseline models per parameter | N-6, A-10, A-11 (model/parameter mapping) |
| 5 | Optimization + SHAP integration | N-7 (XGBoost-SHAP — direct writing template for methodology section) |
| 6 | Anomaly detection module | A-12, N-1, N-2 (SOTA anomaly detection) |
| 7 | System integration + end-to-end tests with IE | A-13→A-16, N-3, N-4 (IoT field deployment) |
| 8 | Technical report + NILES draft | B-2, B-3 (structure only) |

---


## 4. Task Breakdown — team of 3 (ML+XAI, CS side)

**TDS removed from modeling scope (team decision). 3 parameters for 3 people — one parameter each.**

Assignments are **not yet fixed** — to be decided as a team. The functional scope per parameter is:

| Parameter | Scope | Weeks |
|---|---|---|
| **pH** | Multi-model comparison + optimization (steps 5–6) + SHAP integration | 4–6 |
| **EC** | Same | 4–6 |
| **Turbidity** | Same (steps 5–6), then anomaly detection module (step 8) — lighter modeling load, single parameter | 4–8 |

**Shared work (regardless of who does it):**
- Steps 1–4 (ingestion, EDA, feature engineering, split) — to be done once, shared across all parameters
- SHAP wrapper — already written, used by all three models as-is

**⚠️ The IE dashboard must NOT wait until the end of the project to start.** The concept note already plans Dashboard Development in week 4 and Decision Support Framework in week 5 for the IE team — this is not arbitrary; a decision dashboard + KPIs cannot be designed in one final sprint week before the paper deadline (week 8).

**The right sequence: contract first, real data second.**
The IE team will be provided with model outputs progressively as each parameter is finalized (weeks 4–6), with a full integration in week 7.

---

## 5. Immediate actions (this week)

1. Set up the repo structure (§1)
2. Load and clean C-1, run the EDA (steps 1–2)
3. Write the interface contract (§4) and share it with the IE team **before** they move too far ahead on their dashboard without a reference
4. Decide the team breakdown (§5) based on who is actually involved

---

*Working document — replaces/completes `plan_transition_implementation.md`, up to date with dataset decisions (S2 §5) and methodology toolbox.*
