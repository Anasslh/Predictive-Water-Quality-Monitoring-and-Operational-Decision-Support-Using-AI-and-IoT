# Report — Week 2
**Project:** Predictive Water Quality Monitoring and Operational Decision Support Using AI and IoT
**Role:** Computer Science Student — SESC, Nile University, Giza
**Period:** Week 2 (June–July 2026)
**Status:** In progress — updated as papers are read

---

## 1. Literature Review — Paper Summaries

### S-6 · Elmotawakkil A., Enneya N., Bhagat S.K., Ouda M.M., Kumar V. (2025)
**Title:** Advanced machine learning models for robust prediction of water quality index and classification
**Journal:** Journal of Hydroinformatics (IWA) — Open Access — DOI: 10.2166/hydro.2025.290
**Reading method:** 📖 Full read

**1. What they do (method, objective)**
Full methodological comparison of 6 ML models (ANN, DT, SVM, RF, XGBoost, LSTM) for two tasks on multi-station water quality data: WQC classification (5 categories: Excellent/Good/Poor/UFC) and WQI regression. Pipeline: imputation (mean/mode) → z-score standardization → 80/20 chronological split (train up to Feb. 2006, test over 10 months) → grid search + CV-3 for hyperparameter optimization → multi-metric evaluation. Fixed prediction horizon: 24h (lead time).

**2. Key metrics / results**
- **WQC (classification, test):** XGBoost best — Accuracy=Precision=Recall=F1=**0.9996**. ANN=0.9893–0.9903, DT=0.9896–0.9897, SVM=0.9808–0.9814, RF=0.9751–0.9754 (weakest, overfitting observed).
- **WQI (regression, test):** LSTM best — **R²=0.9999, RMSE=0.0378, MSE=0.0014**. XGBoost 2nd — R²=0.9997, RMSE=0.1772, MSE=0.0314. DT — R²=0.9998, RMSE=0.1448. RF weakest — R²=0.9973, RMSE=0.5045.
- **Feature importance (RF/DT/XGBoost):** DO_mgl dominant everywhere, then pH, then temperature. Correlations with WQI: DO_mgl=−0.95, DO_pct=−0.79, Temp=+0.84 (only strong positive correlation).

**⚠️ Correction vs. initial reading programme:** the figures noted in the programme (*"XGBoost → R²=0.9685, MAE=0.0677"*) do not match the actual paper content. The real regression baseline to beat is **R²≈0.9999 (LSTM)**, not 0.9685.

**3. Limitations identified by the authors**
- Overfitting observed on ANN and XGBoost (train/test gaps)
- Linear interpolation imputation → potential bias on features with non-linear trends or abrupt changes
- No interpretability addressed: LSTM explicitly described as a black box, no XAI discussion
- No test of geographic/hydrological generalizability (single dataset)
- No discussion of real-time deployment or IoT production integration

**4. Direct relevance to our project**
Provides the official quantitative baseline to beat (R²≈0.9999 in WQI regression via LSTM). Confirms via methodological contrast our architecture decision (§2.2 S1 report): this paper uses a single multivariate model per task, whereas we use one independent model per parameter — a direct argument to cite for justifying our choice (better per-parameter XAI compatibility). The explicit limitations (no XAI, no real-time/IoT) reinforce our research gap already established in week 1 (S-5, S-7).

**Literature matrix row:**

| Ref | Authors | Year | ML Method | Water Parameters | Dataset | Metrics | Key Results | Gaps | Relevant for NILES |
|---|---|---|---|---|---|---|---|---|---|
| S-6 | Elmotawakkil, Enneya, Bhagat, Ouda, Kumar | 2025 | ANN, DT, SVM, RF, XGBoost, LSTM | Temp, SpCond, Sal, DO%, DO_mgl, Depth, pH, Turb | 61,542 obs., 30 min, multi-station, 2004–2006 | Accuracy, Precision, Recall, F1 (WQC); R², RMSE, MSE (WQI) | WQC: XGBoost Acc=0.9996; WQI: LSTM R²=0.9999, RMSE=0.0378 | No XAI, no real-time/IoT, single dataset, overfitting ANN/XGBoost | Quantitative baseline to beat; contrast multivariate vs. per-parameter architecture |

**NILES running draft paragraph:**

> Recent work by Elmotawakkil et al. (2025) provides a comprehensive multivariate benchmark of six ML algorithms for both water quality classification and WQI regression, with XGBoost and LSTM achieving near-perfect performance (accuracy = 0.9996 and R² = 0.9999, respectively) through systematic grid search optimization. While this study establishes a strong quantitative baseline for predictive accuracy, it relies on a single multivariate model per task and does not address model interpretability, real-time IoT deployment, or operational decision integration — the LSTM model in particular is explicitly treated as a black box. This reinforces the research gap identified across the reviewed literature and motivates our per-parameter modeling approach, which pairs each prediction with a native, locally interpretable SHAP explanation suited for production IoT deployment and operational decision support.

---

### A-1 · Mokhtar A., Elbeltagi A., Gyasi-Agyei Y., Al-Ansari N., Abdel-Fattah M.K. (2022)
**Title:** Prediction of irrigation water quality indices based on machine learning and regression models
**Journal:** Applied Water Science — DOI: 10.1007/s13201-022-01590-x
**Reading method:** 👁 Diagonal

**1. What they do (method, objective)**
Prediction of 6 irrigation water quality indices (SSP, SAR, RSC, PI, KR, PS) from 4 simple input variables (EC, Na⁺, Ca²⁺, HCO₃⁻), on 105 samples collected in July 2020 on the Bahr El-Baqr drain (Nile Delta, Egypt — direct agricultural context). Comparison of 3 ML models (SVM/SVR, XGBoost, Random Forest) and 4 multiple regression models (Stepwise, PCR, PLS, OLS). 75/25 split, grid search for hyperparameter tuning.

**2. Key metrics / results**
- **Best overall model: Stepwise Regression (SW)** — RMSE=0.21%, MAE=0.17, SI=0.03, **R²=0.98 (SAR)**, R² ranging 0.87–0.98 by index.
- **Best ML model: SVR**, followed by XGBoost then RF. SI < 0.1 for all indices except RSC (excellent by their classification).
- RF: weakest ML — R² from 0.53 to 0.78, RMSE up to 3.27% for SSP, MAE up to 2.62% for PI.
- **RSC is the hardest parameter to predict** for all models (SI=0.52 for XGB and RF — "poor").

**3. Limitations identified by the authors**
- Small dataset (105 samples, single time point — July 2020, not a time series)
- Classical ML models only (no DL/LSTM, no sequential)
- RSC systematically poorly predicted — authors suggest input variable selection is the limiting factor
- No XAI, IoT, or real-time deployment discussion — purely retrospective study on lab data

**4. Direct relevance to our project**
Paper mirroring the agricultural/irrigation context announced in the concept note — variables directly transposable (EC, pH, Na, Ca, HCO₃ belong to the low-cost IoT sensor family). Empirically confirms that XGBoost and RF are not always the best ML models (SVR wins here, unlike S-6 where XGBoost/LSTM dominated) — solid argument to justify rigorous multi-model comparison rather than fixing XGBoost by default. The systematic weakness on RSC is a useful signal: some composite parameters are structurally harder to predict.

**Literature matrix row:**

| Ref | Authors | Year | ML Method | Water Parameters | Dataset | Metrics | Key Results | Gaps | Relevant for NILES |
|---|---|---|---|---|---|---|---|---|---|
| A-1 | Mokhtar, Elbeltagi, Gyasi-Agyei, Al-Ansari, Abdel-Fattah | 2022 | SVR, XGBoost, RF + 4 regressions (SW, PCR, PLS, OLS) | EC, Na⁺, Ca²⁺, HCO₃⁻ → SSP, SAR, RSC, PI, KR, PS | 105 samples, July 2020, Bahr El-Baqr drain, Egypt | R², RMSE, MAE, SI | SW best overall (R²=0.98 SAR); SVR best ML; RSC poorly predicted by all | Small static dataset, no time series, no XAI/IoT | Direct irrigation context; argument for rigorous multi-model comparison |

**NILES running draft paragraph:**

> In an Egyptian irrigation context closely aligned with ours, Mokhtar et al. (2022) compared three ML algorithms (SVR, XGBoost, RF) and four regression techniques for predicting six irrigation water quality indices from a small set of readily measurable parameters (EC, Na⁺, Ca²⁺, HCO₃⁻). Notably, SVR outperformed XGBoost and RF, and simple stepwise regression matched or exceeded all ML models (R² = 0.98 for SAR), while composite indices such as RSC remained difficult to predict across all approaches. This finding underscores that no single model architecture universally dominates water quality prediction tasks, motivating a rigorous per-parameter model comparison in our methodology rather than a fixed choice of algorithm. The study's reliance on a single-timepoint dataset (105 samples, no temporal dimension) further highlights the added value of our continuous IoT-based, time-series prediction approach over static laboratory-based assessments.

---

### A-2 · Ibrahim H., Yaseen Z.M., Scholz M., Ali M., Gad M., Elsayed S., Khadr M., Hussein H., Eid M.H. et al. (2023)
**Title:** Evaluation and Prediction of Groundwater Quality for Irrigation Using an Integrated Water Quality Indices, Machine Learning Models and GIS Approaches: A Representative Case Study
**Journal:** Water (MDPI) — Open Access — DOI: 10.3390/w15040694
**Reading method:** 👁 Diagonal

**1. What they do (method, objective)**
Evaluation and prediction of groundwater quality in the Nubian sandstone aquifer (El Kharga oasis, Egypt) for irrigation. Combines quality indices (IWQI, SAR, SSP, KI, PS, RSC), hydrogeochemical analysis (Piper, Chadha, Durov, USSL diagrams) and GIS mapping, then compares 2 ML models — **SVM** and **ANFIS** (Adaptive Neuro-Fuzzy Inference System) — to predict the 6 indices from 9 physicochemical parameters (EC, TDS, K⁺, Na⁺, Ca²⁺, Cl⁻, SO₄²⁻, HCO₃⁻, CO₃²⁻). 140 wells sampled in July 2020.

**2. Key metrics / results**
- **IWQI (test):** ANFIS clearly superior — R²=0.97, RMSE=4.54, E=0.96 vs SVM R²=0.76, RMSE=12.45, E=0.70 (significant performance drop for SVM on test).
- **SAR (test):** ANFIS dominates — R²=0.94, E=0.94 vs SVM R²=0.36, E=0.20 (SVM nearly unusable for generalization).
- **PS (test):** ANFIS near-perfect — R²=1.00, E=1.00.
- Across all indices without exception, ANFIS > SVM, with a much smaller train/test gap for ANFIS — better generalization.

**3. Limitations identified by the authors**
- SVM shows pronounced overfitting (good train performance, sharp drop on test) — not robust enough for reliable deployment
- Single site, single time point (July 2020), no time series
- Dependent on availability and quality of hydrogeological dataset (limiting factor for generalization explicitly mentioned in conclusion)
- No XAI, IoT, or real-time deployment discussion

**4. Direct relevance to our project**
Provides a strong signal consistent with S-6 and A-1: no model universally dominates — here ANFIS crushes SVM, whereas in A-1 SVR dominated XGBoost, and in S-6 XGBoost/LSTM dominated. Strongly reinforces the methodological argument for systematic multi-model comparison per parameter rather than an a priori choice. The SVM train/test generalization gap is a concrete example to cite for justifying robust validation in our methodology. ANFIS had not appeared in previous readings — worth considering as an additional candidate if time allows.

**Literature matrix row:**

| Ref | Authors | Year | ML Method | Water Parameters | Dataset | Metrics | Key Results | Gaps | Relevant for NILES |
|---|---|---|---|---|---|---|---|---|---|
| A-2 | Ibrahim, Yaseen, Scholz, Ali, Gad, Elsayed, Khadr, Hussein, Eid et al. | 2023 | SVM, ANFIS | T°, pH, EC, TDS, K⁺, Na⁺, Mg²⁺, Ca²⁺, Cl⁻, SO₄²⁻, HCO₃⁻, CO₃²⁻ → IWQI, SAR, SSP, KI, PS, RSC | 140 wells, groundwater, July 2020, El Kharga oasis, Egypt | R², RMSE, MAD, Nash-Sutcliffe (E) | ANFIS dominates SVM on all indices (IWQI test R²=0.97 vs 0.76); SVM overfits heavily | Single site, single time point, no XAI/IoT | Confirms no universal model dominance; ANFIS as unexplored candidate |

**NILES running draft paragraph:**

> Ibrahim et al. (2023) compared SVM and ANFIS (adaptive neuro-fuzzy inference system) for predicting six irrigation water quality indices from groundwater physicochemical data in an arid Egyptian oasis. ANFIS consistently and substantially outperformed SVM across all indices (e.g., IWQI test R² = 0.97 versus 0.76), with SVM exhibiting pronounced overfitting between training and testing phases. Combined with contrasting findings from other reviewed studies — where XGBoost and LSTM dominate in one dataset (Elmotawakkil et al., 2025) and SVR outperforms both in another (Mokhtar et al., 2022) — this reinforces that no single algorithm generalizes best across water quality prediction tasks, further justifying our systematic, per-parameter model comparison methodology and highlighting the importance of robust train/test validation protocols in our own pipeline.

---

### A-3 · Wang X., Li Y., Qiao Q., Tavares A., Liang Y. (2023)
**Title:** Water Quality Prediction Based on Machine Learning and Comprehensive Weighting Methods
**Journal:** Entropy (MDPI) — Open Access — DOI: 10.3390/e25081186
**Reading method:** 👁 Diagonal

**1. What they do (method, objective)**
Prediction of 4 water quality parameters (DO, NH₃-N, TN, TP) on a coastal Chinese river (Pearl River Basin), from 5,058 samples collected every 4h between Nov. 2020 and Feb. 2023. Main methodological innovation: a combined weighting method (entropy weighting + Pearson coefficient) for feature selection — each target parameter has its own optimized input variable subset, rather than a fixed feature set for all. Comparison of 5 models: SVM, MLP, RF, XGBoost, LSTM. Split 4552/506 (train/test), grid search, 10 parallel runs for statistical robustness.

**2. Key metrics / results**
- **DO (test), comparison of 5 models:** LSTM best — R²=0.882, RMSE=1.827, NSE=0.877. Then SVM — R²=0.820, RMSE=2.195, NSE=0.823. MLP — R²=0.775. RF — R²=0.720 (lower). XGBoost weakest of all — R²=0.690, RMSE=2.899.
- **LSTM across 4 parameters:** DO R²=0.882; NH₃-N R²=0.830; TP R²=0.773; TN R²=0.745 (weakest — LSTM struggles on extreme TN values).
- Notable: RF and XGBoost consistently underperform here, unlike S-6 where they dominated — attributed by the authors to class imbalance in the DO distribution.

**3. Limitations identified by the authors**
- LSTM specifically struggles on extreme TN values — attributed to potentially suboptimal feature selection, noise/outliers, and insufficient training
- Class imbalance in DO data penalizing RF/XGBoost
- Low model interpretability explicitly mentioned as a general limitation
- Linear interpolation imputation — same limits as S-6

**4. Direct relevance to our project**
Two major interests: (1) the entropy+Pearson weighting method for per-parameter feature selection is directly transposable to our "one independent model per parameter" architecture — a reproducible and citable protocol rather than an ad hoc selection. (2) Confirms once more (after S-6, A-1, A-2) that no model universally dominates — here LSTM wins but XGBoost, often cited as a reference, is the weakest of the 5. The specific weakness of LSTM on extreme TN values is also an attention point for our anomaly detection module (week 6): prediction models can structurally miss extremes, which the anomaly detector must compensate for.

**Literature matrix row:**

| Ref | Authors | Year | ML Method | Water Parameters | Dataset | Metrics | Key Results | Gaps | Relevant for NILES |
|---|---|---|---|---|---|---|---|---|---|
| A-3 | Wang, Li, Qiao, Tavares, Liang | 2023 | SVM, MLP, RF, XGBoost, LSTM + entropy weighting/Pearson feature selection | DO, NH₃-N, TN, TP (+ Temp, pH, KMnO₄, Cond, Turb as input) | 5,058 samples, 4h, Nov.2020–Feb.2023, Pearl River, China | R², MSE, RMSE, NSE | LSTM best (DO R²=0.882); XGBoost and RF weakest of 5 | LSTM struggles on extremes (TN), no XAI, limited linear imputation | Per-parameter feature selection method reusable; confirms no universal model dominance |

**NILES running draft paragraph:**

> Wang et al. (2023) proposed a comprehensive feature-selection method combining entropy weighting and Pearson correlation to tailor input variable sets independently for each predicted water quality parameter (DO, NH3-N, TN, TP), then compared five ML models (SVM, MLP, RF, XGBoost, LSTM). LSTM achieved the best overall performance (R² = 0.882 for DO), while XGBoost and RF — often reported as top performers elsewhere in the literature — were the weakest of the five models tested, further illustrating that model performance is highly dataset- and context-dependent. Notably, even the best-performing LSTM struggled to accurately predict extreme values for total nitrogen, a limitation the authors attribute to feature selection and insufficient training data. This finding directly motivates the complementary role of our anomaly detection module, which is specifically designed to flag the extreme deviations that per-parameter prediction models are structurally prone to underestimate, and supports adopting a principled, reproducible feature-selection protocol per parameter rather than a uniform input set across all models.

---

### N-5 · Kazapoe R.W., Sagoe S.D., Abu M. (2024)
**Title:** Predicting irrigation water quality indices in a typical mining dominated area in the Upper West region of Ghana using multiple machine learning techniques
**Journal:** Discover Water — Open Access — DOI: 10.1007/s43832-024-00104-x
**Reading method:** 👁 Diagonal

**1. What they do (method, objective)**
Assessment of groundwater quality for irrigation in an artisanal small-scale gold mining (ASGM/"galamsey") area in Upper West, Ghana. 105 samples collected over 3 campaigns (Nov. 2021–Jan. 2022), 16 physicochemical parameters analyzed (pH, EC, Ca, Mg, Na, K, HCO₃, CO₃, SO₄, Cl, NO₃, F, TDS, alkalinity, total hardness). Calculation of 8 irrigation indices (SAR, SSP, Na%, MH, KR, RSC, PI, PIG) from measured concentrations. Then 3 ML models — SVR, Gradient Boosting Regression (GBR), ANN — trained to predict each of the 8 indices from measured variables (24 models total), 80/20 split, cross-validation + automated hyperparameter optimization. Post-hoc interpretability via SHAP for each model.

**2. Key metrics / results**
- **Raw indices (without ML):** SAR, SSP, RSC, KR — 100% of samples "suitable/good/excellent". MH — 69.52% "unsuitable" (Mg dominant). Na% — 91.19% suitable, Chasea/Tiza sites problematic. PI — 69.62% "good", PIG — 96.19% insignificant pollution (except Mengwe, mining zone).
- **Best overall model: SVR-RSC** — R²=0.99, RMSE=0.09, MAE=0.07.
- **Worst model: GBR-RSC** — R²=0.60, RMSE=0.60, MAE=0.32.
- **SVR dominates overall**: 6/8 models with R²≥0.95 (RSC, SSP=0.98, PIG=0.98, SAR=0.97, PI=0.97, MH=0.95). Weakest SVR: KR (R²=0.86), Na% (R²=0.89).
- **SAR is the most robust index across all methods**: ANN=0.96, GBR=0.91, SVR=0.97 — consistent signal, unlike other indices where performance varies greatly by model.
- **SHAP:** sodium = primary driver (positive effect) for 50% of models; calcium = primary driver for all MH models (negative effect); magnesium = primary driver for all PI models (negative effect); nitrate = primary driver for PIG (positive) and secondary for RSC.

**3. Limitations identified by the authors**
- Dependency on static historical data, stationarity assumption of water quality metrics — explicitly cited as a limitation for use under real dynamic conditions
- No real-time data integration or IoT discussion
- SHAP used only as a post-hoc academic interpretation tool, never positioned as a component of an operational decision system or production deployment
- GBR inconsistent (good on SAR/PIG, very weak on RSC) — authors recommend hybrid/ensemble approaches as future work
- Outlier removal and data quality identified as improvement axes

**4. Direct relevance to our project**
Main interest: methodological, not geographic (the project deployment context is not a mining zone — the concept note remains general, irrigation/agriculture as main use case). The paper confirms, via an additional independent dataset from S-6, A-1, A-2, A-3, the pattern of no universal model dominance — with a notable nuance: SVR dominates fairly consistently here (6/8 indices ≥0.95), the closest signal to systematic dominance observed in the entire literature review — worth mentioning explicitly in our discussion rather than smoothing into the general pattern. SAR emerges as a robust index across all methods — useful for our per-parameter baseline choice. Most exploitable point for our research gap: SHAP is used per model/per parameter (approach close to ours), but only in retrospective academic post-hoc interpretation, never coupled to real-time IoT deployment or an operational decision layer — reinforcing the already established gap (S-5, S-7, S-6), regardless of the geographic context (irrigation vs. mining) which is not ours.

**Literature matrix row:**

| Ref | Authors | Year | ML Method | Water Parameters | Dataset | Metrics | Key Results | Gaps | Relevant for NILES |
|---|---|---|---|---|---|---|---|---|---|
| N-5 | Kazapoe, Sagoe, Abu | 2024 | SVR, GBR, ANN (+SHAP post-hoc) | pH, EC, Ca, Mg, Na, K, HCO₃, CO₃, SO₄, Cl, NO₃, F, TDS, Alk, TH → SAR, SSP, Na%, MH, KR, RSC, PI, PIG | 105 samples, 3 campaigns Nov.2021–Jan.2022, mining zone Upper West, Ghana | R², RMSE, MAE | SVR-RSC best (R²=0.99); SVR dominates 6/8 indices; SAR robust across all methods | Static retrospective data, no real-time/IoT, academic post-hoc SHAP only, GBR inconsistent | Methodological nuance on SVR dominance; per-model SHAP without operational decision = confirmed gap |

**NILES running draft paragraph:**

> Kazapoe et al. (2024) compared SVR, gradient boosting regression, and ANN for predicting eight irrigation water quality indices from physicochemical measurements in a groundwater dataset, coupling each model with post-hoc SHAP explanations. Unlike several other reviewed studies where model performance varied considerably by dataset, SVR here consistently outperformed the alternatives across most indices (R² ≥ 0.95 for 6 of 8 indices), with sodium adsorption ratio emerging as a particularly robust target across all three algorithms — a useful nuance to our broader observation that no single model universally dominates. Critically, while the study integrates SHAP explanations per parameter, similar to our proposed approach, these remain confined to retrospective academic interpretation rather than being embedded in a real-time IoT deployment or operational decision-support pipeline, and the authors explicitly flag the static, stationarity-assuming nature of their dataset as a limitation for dynamic field conditions. This further substantiates the persistent gap between per-parameter explainability and its practical integration into production monitoring systems that our work aims to address.

---

### A-4 · Bui D.T., Khosravi K., Tiefenbacher J., Nguyen H., Kazakis N. (2020)
**Title:** Improving prediction of water quality indices using novel hybrid machine-learning algorithms
**Journal:** Science of the Total Environment — DOI: 10.1016/j.scitotenv.2020.137612 · Citations: 408
**Reading method:** ⚡ Abstract + conclusion

**1. What they do (method, objective)**
Prediction of the Iranian WQI (IRWQIsc) on the Talar River (Iran, humid climate), from 6 years of monthly data (2012–2018, 2 stations). Comparison of 4 standalone tree-based models (M5P, Random Forest, Random Tree, REPTree) and 12 hybrids built by combining each standalone with 3 meta-classifiers (Bagging, CV Parameter Selection, Randomizable Filtered Classifier) — 16 models total. 70/30 split, 10-fold cross-validation, 10 input variable combinations tested by Pearson correlation.

**2. Key metrics / results**
- **Best model: Bagging-Random Tree hybrid (BA-RT)** — R²=0.941, RMSE=2.71, MAE=1.87, NSE=0.941, PBIAS=0.500 (lowest bias).
- **Worst model: CVPS-REPT** — R²=0.823 (weakest, but still acceptable).
- All models have R²>0.75 — overall performance judged very good.
- **Fecal coliform (FC)** = most determinant variable for WQI, followed by BOD, NO₃, DO, EC, COD, PO₄, turbidity, TS, pH (least determinant).
- Hybridization improves most standalones but not systematically (e.g. standalone RF > BA-REPT hybrid) — no absolute rule "hybrid > standalone".
- Best input variable combination varies by model (no universal set).

**3. Limitations identified by the authors**
- Results not generalizable to other basins/hydrological contexts — "these results cannot be generalized and applied to other study areas"
- BA-RT did not predict extreme WQI values well despite good overall performance
- Relatively short dataset (6 years) — authors explicitly note that longer series would make models more robust
- No XAI, real-time IoT, or operational deployment discussion — purely retrospective batch study

**4. Direct relevance to our project**
Leading methodological argument to justify a hybrid architecture if we consider going beyond a simple model per parameter (weeks 5–6, multivariate comparison mentioned in S1 report §2.2): the performance hierarchy here (bagging hybrids on top, but not systematically superior to standalones) concretely illustrates the same message as S-6/A-1/A-2/A-3/N-5 — no architecture universally dominates, including between hybrid and standalone versions of the same algorithm. Also confirms, like A-3, the importance of per-model variable selection rather than a fixed set. 408-citation reference — good authority support for the methodology section of our NILES paper on choosing to compare multiple architectures rather than fixing one a priori.

**Literature matrix row:**

| Ref | Authors | Year | ML Method | Water Parameters | Dataset | Metrics | Key Results | Gaps | Relevant for NILES |
|---|---|---|---|---|---|---|---|---|---|
| A-4 | Bui, Khosravi, Tiefenbacher, Nguyen, Kazakis | 2020 | M5P, RF, RT, REPT (standalone) + Bagging/CVPS/RFC (hybrids), 16 models | BOD, COD, TS, DO, FC, pH, PO₄, NO₃, turbidity, EC → IRWQIsc | 6 years monthly (2012–2018), 2 stations, Talar River, Iran | R², RMSE, MAE, NSE, PBIAS | BA-RT best (R²=0.941); hybridization improves but not systematically; FC = key variable | Not generalizable, poor prediction of extremes, no XAI/IoT, short dataset | Authority argument (408 cit.) for hybrid vs. standalone comparison; confirms no universal dominance |

**NILES running draft paragraph:**

> Bui et al. (2020) compared four standalone tree-based algorithms and twelve hybrid variants (combining each with bagging, CV parameter selection, or randomizable filtered classification) for predicting a river water quality index in northern Iran, with the hybrid Bagging-Random Tree model achieving the best performance (R² = 0.941). Critically, hybridization did not uniformly improve prediction accuracy — the standalone random forest model, for instance, outperformed several hybrid combinations — reinforcing that neither model family nor hybridization strategy alone guarantees superior performance across water quality prediction tasks. This large-scale comparison (16 model variants, 408 citations) lends further authority to the growing evidence across the reviewed literature that systematic, context-specific model comparison — rather than a fixed architectural choice — is essential, a principle directly reflected in our per-parameter modeling methodology.

---

## 5. Dataset Assessment — Data Source Decision

### 5.1 Dataset C-1 (IEEE DataPort) — Ramgarh, Jharkhand, India

**Files:** `data.xlsx`, `Documentation.docx`, `Data_Dictionary.docx`, `Readme.md`

**Structure:** continuous daily time series, 365 rows (Feb. 1, 2025 – Jan. 31, 2026), 1 station (mining zone), no missing values, no duplicates.

**Parameters:** pH, TDS, EC, Turbidity (4 only) + WQI + WQI_Class.

**Positives:**
- Resolves the attention point from week 1 (§3 S1 report): this is not a min/max, it is a real continuous daily series.
- The 4 raw parameters match exactly the ranges announced in the documentation (pH 5.14–8.98, TDS 30–958, EC 416–1200, Turbidity 0.5–169.65).

**Blocking issues identified:**
- WQI far exceeds the "0–100" announced by the Data Dictionary (actual: 10 to 1417.7) — due to the Weighted Arithmetic Index method (unbounded), documented in `Documentation.docx` but poorly summarized in `Data_Dictionary.docx`.
- Actual WQI_Class values (`Safe`, `Moderate`, `Poor`, `Hazardous`, empirical thresholds ~50/100/200) do not match **either** of the two provided documents (which themselves contradict each other: "Safe-to-Poor" vs "Excellent/Good/Poor/Very Poor" at thresholds 25/50/75/100). → **Do not use WQI/Class as-is without transparent and citable recalculation.**
- Severe class imbalance (65% "Hazardous") if `Class` is used as a classification target.
- Daily granularity only (despite the README mentioning "real-time") → supports a 24h prediction horizon, not shorter horizons (2h/6h) considered in §2.1 S1 report.
- Single site, single context (mining zone) → no spatial generalization test possible with this dataset alone.

### 5.2 Kaggle/CPCB Dataset (2021) — Lakes/tanks/ponds, India

**Files:** `Water_pond_tanks_2021.csv` (raw), `Water_pond_tanks_2021_clean.csv` (cleaned)

**Structure:** cross-sectional, **620 distinct stations** across several Indian states (Karnataka, Telangana, Assam...), 4 types of water bodies (Lake, Tank, Pond, Wetland). **One row = min and max observed over 2021 per station — no time series, no WQI provided.**

**Parameters:** Temperature, Dissolved Oxygen (DO), pH, Conductivity, BOD (+ Nitrate/Nitrite and fecal/total coliforms in raw file, removed in clean version due to ~8% missing values marked "-").

**Quality:** clean version reliable (verified on a correctly fixed data entry error — pH entered as "7.54\n4" → "7.54"), nearly complete (2 to 5 missing values out of 620 per retained column). The raw file requires `latin1` encoding (not UTF-8) and contains character artifacts (µ, °) to clean.

### 5.3 Decision: no merge — complementary use

The two datasets are structurally incompatible for merging (time series vs. cross-section, almost no parameter overlap — only pH and conductivity/EC are shared, no common sites, no shared WQI methodology). Merging would introduce noise without added value.

**Decision adopted:**
- **C-1 (Ramgarh)** = main dataset for the temporal per-parameter prediction pipeline (addresses the core need in §2.1 S1 report).
- **Kaggle/CPCB** = complementary reference, no direct merge: useful to (a) calibrate/justify standards (Si) to recalculate a clean and citable WQI for C-1 (given the documentary inconsistency identified in §5.1), and (b) support a geographic generalizability discussion in the NILES paper (limitation already cited as a gap by A-1, A-2, A-4).
- No need to look for a third dataset for now — C-1 is sufficient to start EDA and feature engineering (week 3).

---

## 6. Next reading

*(to be completed)*

---

*Working document — updated throughout week 2 readings, SESC Nile University*
