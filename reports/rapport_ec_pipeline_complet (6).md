# Rapport — Pipeline EC complet (bout en bout)
**Projet :** Predictive Water Quality Monitoring and Operational Decision Support Using AI and IoT
**Rôle :** Computer Science Student — SESC, Nile University, Giza
**Paramètre couvert :** EC (Electrical Conductivity) uniquement — pH et Turbidity non commencés
**Statut :** EC figé de bout en bout (données → modèle → XAI). Document de référence avant de répliquer sur pH/Turbidity ou d'attaquer l'anomaly detection.

---

## 1. Contexte et objectif de ce document

Ce rapport consolide toutes les décisions méthodologiques prises pendant le développement du pipeline EC, dans l'ordre chronologique, avec la justification de chaque choix — y compris les pistes testées et écartées, qui constituent une contribution méthodologique à part entière pour l'article NILES (démonstration de robustesse plutôt que simple accumulation de résultats positifs).

**Principe respecté tout du long :** le test set n'a été touché qu'une seule fois, à la toute fin, une fois toutes les décisions de features/modèle/hyperparamètres arrêtées sur le train/validation set uniquement.

---

## 2. Données et préparation

### 2.1 Source
- Dataset : `data/processed/c1_clean.csv` (IEEE DataPort, station de Ramgarh, Jharkhand, Inde)
- 365 lignes, série journalière continue, 2025-02-01 → 2026-01-31, un seul site
- Colonnes : Date, pH, EC, Turbidity, WQI, Class
- **WQI et Class ignorés pour EC** — incohérents avec leur documentation d'origine (voir rapport semaine 2), non recalculés à ce stade car non nécessaires pour la modélisation par paramètre

### 2.2 Feature engineering (`src/data/feature_engineering.py`)
Fonction `build_features(df, target)`, paramétrable par colonne cible. Pour EC, 9 features retenues :
- 3 lags de la cible : EC_lag1, EC_lag2, EC_lag3
- Rolling mean et rolling std sur fenêtres 3j et 7j (calculées sur valeurs décalées — anti-fuite)
- Lag1 de pH et Turbidity (co-variables explicatives, utiles pour SHAP)

365 → 358 lignes utiles (7 premières lignes perdues à cause de la fenêtre glissante 7j).

**Vérification anti-fuite effectuée :** aucune feature avec |corrélation| ≥ 0.99 avec la cible ; dates train/val/test sans chevauchement confirmées.

### 2.3 Split (`src/data/split.py`)
Split chronologique strict 70/15/15 (pas aléatoire) :

| Split | n | Dates |
|---|---|---|
| Train | 250 | 2025-02-08 → 2025-10-15 |
| Val | 53 | 2025-10-16 → 2025-12-07 |
| Test | 55 | 2025-12-08 → 2026-01-31 |

**Diagnostic important — décalage de niveau train/val/test :**

| Split | mean EC | std EC |
|---|---|---|
| Train | 783 | 212 |
| Val | 698 | 50 |
| Test | 788 | 49 |

Train couvre une période à forte variabilité (crue/sécheresse) ; val et test sont des périodes plus calmes. Ce décalage explique structurellement les R² négatifs obtenus sur tout le reste du pipeline (voir §5).

---

## 3. Comparaison de modèles (out-of-the-box)

Trois algorithmes testés avec le contrat `ParameterModel` (`src/models/base.py`) : RandomForest (baseline existante), XGBoost, SVR.

| Modèle | RMSE val | MAE val | R² val |
|---|---|---|---|
| SVR (défaut) | 58.15 | 45.79 | -0.39 |
| Random Forest (défaut) | 70.11 | 57.61 | -1.03 |
| XGBoost (défaut) | 77.32 | 64.98 | -1.46 |

**Point de vigilance découvert et corrigé :** SVR sans scaling sur la cible (y) prédisait quasi-constant (ratio variance prédictions/réel = 0.19) — ajout d'un y-scaler interne à `ECModelSVR`, corrigé avant le tuning.

**Baselines de référence ajoutées** pour contextualiser :
- Persistence (EC_lag1) : RMSE = 76.02
- Moyenne train : RMSE = 98.1

---

## 4. Tuning des hyperparamètres

Grid/random search sur val (train pour fit), sans toucher au test.

| Modèle | RMSE val | Hyperparamètres retenus |
|---|---|---|
| **XGBoost tuned** | **60.9** | depth=3, n=100, lr=0.01 |
| SVR tuned | 64.8 | C=1, ε=0.5, γ=0.1 |
| RF tuned | 67.3 | n=300, depth=5, min_samples_leaf=4 |

Le tuning a inversé la hiérarchie initiale : XGBoost, le plus faible à froid (lr=0.1 par défaut trop agressif, surapprentissage du bruit sur 250 lignes), devient le meilleur une fois lr abaissé à 0.01.

---

## 5. Exploration de features additionnelles — testées et écartées

Objectif : améliorer la précision au-delà du XGBoost tuned (RMSE val = 60.94, considéré comme référence "baseline 9 features").

| Piste testée | RMSE val | Résultat | Explication |
|---|---|---|---|
| Differencing (ΔEC) | 78.53 (reconstruit) | ❌ Écarté | Résout le décalage de niveau train/val (gap réduit de +84.9 à -2.4) mais dégrade fortement le rapport signal/bruit — la 1ère différence d'une série quasi-stationnaire est surtout du bruit |
| EWMA(3) | 65.40 | ❌ Écarté | Redondant avec EC_lag1, crée de l'instabilité dans le choix de features par XGBoost |
| EWMA(5) | 60.95 | ≈ Neutre | Aucun gain |
| EWMA(3,5) combiné | 65.87 | ❌ Écarté | Idem EWMA(3), aggravé |
| Lags longs (lag7+lag14) | 64.22 | ❌ Écarté | Perte de lignes d'entraînement + signal trop lointain sur une série au changement irrégulier |
| Rolling slope (7j) | 60.91 | ≈ Neutre (-0.03) | Redondant avec roll7_mean déjà présent |
| Tuning bayésien élargi (Optuna, 40 essais) | 60.74 | ≈ Neutre (-0.20) | Convergence vers un voisinage similaire (depth=2-3, lr≈0.005-0.01) — confirme la robustesse du choix initial plutôt que d'en révéler un meilleur |
| Ensemble (XGB+SVR+RF, moyenne simple) | 59.65 | ⚠️ Gain marginal (-1.29) mais écarté | Gain non significatif (+2.1%) au regard de la complexité XAI ajoutée (3 modèles à expliquer au lieu d'1) — incompatible avec le pilier XAI natif par paramètre du projet (§2.3 rapport S1) |

**Conclusion de cette phase :** aucune amélioration testée n'apporte de gain significatif et généralisable. C'est documenté comme résultat de robustesse méthodologique, pas comme échec — cohérent avec les limites déjà identifiées dans la littérature (dataset mono-site, faible volume, cf. A-1/A-2/A-4).

---

## 6. Modèle final figé

**XGBoost — depth=3, n_estimators=100, learning_rate=0.01, feature set baseline (9 features), sans ajout d'EWMA/lags longs/slope/ensemble.**

Classe : `ECModelXGBoost`, version `xgb_ec_v1_final`, hyperparamètres figés en dur dans le constructeur (`src/models/ec/train.py`). Modèle entraîné sur train+val combinés, sauvegardé dans `models_store/ec_xgboost_v1_final.json`.

### 6.1 Évaluation finale sur test (une seule fois)

| Métrique | Valeur |
|---|---|
| RMSE | 53.48 µS/cm |
| MAE | 42.42 µS/cm |
| R² | -0.23 |
| Médiane erreur | 36.1 µS/cm |
| p90 erreur | 91.1 µS/cm |
| Gain vs persistence | +14.24 RMSE |

**Sur le R² négatif persistant :** ce n'est pas un défaut du modèle mais un artefact structurel — la variabilité naturelle du test set (std = 48.6 µS/cm) est du même ordre que l'erreur du modèle, donc R² (qui compare l'erreur à cette variance) reste mécaniquement bas. La métrique de référence pertinente ici est le gain RMSE vs persistence (+14.24), pas le R² isolément. À documenter explicitement dans le papier pour anticiper la question d'un reviewer.

**Précision en relatif :** RMSE de 53.5 µS/cm sur une plage test de ~680-930 µS/cm ≈ 6.8% d'erreur relative moyenne.

### 6.2 Explicabilité SHAP (`src/xai/shap_wrapper.py`, explainer_type="tree")

Testé sur 5 lignes de test. Résultats cohérents :
- **EC_roll3_mean dominant (5/5 lignes)** — meilleur résumé de l'état récent, moins bruité qu'un lag brut
- **EC_roll7_mean présent (5/5 lignes)** — apporte le contexte de tendance longue
- **EC_lag1 seulement en 4e position** — XGBoost préfère la moyenne lissée au lag brut, cohérent avec un signal EC bruité (capteurs bas coût en zone minière)

**Limite identifiée via SHAP, à retenir pour la suite du projet :** les inversions de signe de EC_roll3_mean/EC_lag2 montrent que le modèle **ne capture pas bien les retournements de tendance** — quand la rolling mean est haute mais que le EC réel est en train de baisser (ou l'inverse), le modèle se trompe de direction. C'est cohérent avec la limite déjà identifiée dans la littérature (A-3 : LSTM peinant sur les valeurs extrêmes/TN) et **justifie directement le module d'anomaly detection prévu en semaine 6** — la prédiction seule ne suffit pas à capter les changements de régime, un détecteur dédié est nécessaire.

---

## 7. Ce qui est acquis pour la suite

- **Protocole rodé et réplicable** pour pH et Turbidity : EDA → feature engineering paramétrable (déjà généralisé, pas hardcodé sur EC) → split chronologique → comparaison multi-modèles → tuning → exploration de features → évaluation finale unique sur test → SHAP
- **Argument concret pour l'anomaly detection** : limite des retournements de tendance démontrée empiriquement sur données réelles du projet, pas seulement citée depuis la littérature
- **Section méthodologie NILES solide** : pistes testées et écartées (differencing, EWMA, ensemble) constituent une démonstration de rigueur, pas un manque de résultat

## 8. Décisions en attente

- Répliquer le protocole sur pH et Turbidity, ou attaquer l'anomaly detection sur EC en premier (capitaliser sur l'observation SHAP pendant qu'elle est fraîche) — à trancher
- Recalcul du WQI propre (identifié comme point bloquant en semaine 2) toujours en attente — nécessaire pour la couche IE, pas urgent pour la modélisation par paramètre CS
- Contrat d'interface CS/IE (format prédiction + explication SHAP) à formaliser avec l'équipe IE, mentionné depuis le rapport S1 §2.5, toujours pas fait formellement

## 9. Module anomaly detection (résidu + Isolation Forest + SHAP)

Ajouté après la lecture semaine 6 (S-3, A-12, N-1, N-2 — voir rapport_semaine6_anomaly_detection.md), en s'appuyant sur l'architecture qui s'en dégage : résidu de prédiction → Isolation Forest, plutôt qu'Isolation Forest appliqué directement aux valeurs brutes (qui sous-performe nettement, cf. N-2).

### 9.1 Architecture implémentée

- `src/anomaly/residual.py` — `compute_residuals()` : résidu = valeur réelle − valeur prédite par ECModelXGBoost (déjà figé), par ligne
- `src/anomaly/detector.py` — `ECAnomalyDetector` : Isolation Forest entraîné sur les résidus de train+val (considérés "normaux"), score d'anomalie normalisé [0,1], seuil par défaut 0.5 (cohérent avec PADSV, A-12)
- `src/anomaly/explain.py` — `explain_anomaly()` / `batch_explain()` : couple le score d'anomalie à l'explication SHAP de la prédiction sous-jacente (réutilise `compute_shap_explanation()` déjà en place). **C'est la contribution originale du module** — aucun des 4 papiers lus en semaine 6 (S-3, A-12, N-1, N-2) ne couple score d'anomalie et explication XAI.
- `src/anomaly/validate_ec_anomaly_detection.py` — validation reproductible sur anomalies synthétiques

### 9.2 Validation par balayage d'intensité (résultat retenu)

**Note méthodologique importante :** la première validation (3 anomalies isolées à intensité fixe ±2-3σ) a été remplacée par un balayage d'intensité plus rigoureux (0.5σ à 3.0σ, 10-15 répétitions par intensité et par type, positions variées dans le test set) — la validation à occurrence unique s'est révélée non représentative (voir §9.3) et ne doit pas être citée telle quelle.

| Intensité (σ_résidu = 89.4 µS/cm) | Spike recall | Burst recall (point) | Burst recall (événement) | Dip recall |
|---|---|---|---|---|
| 0.5σ (45 µS/cm) | 0% | 0% | 0% | 6.7% |
| 1.0σ (89 µS/cm) | 0% | 3.3% | 13% | 20% |
| 1.5σ (134 µS/cm) | 20% | 16.7% | 60% | 33.3% |
| 2.0σ (179 µS/cm) | 33.3% | 43.3% | 100% | 66.7% |
| 2.5σ (223 µS/cm) | 73.3% | 68.3% | 100% | 86.7% |
| 3.0σ (268 µS/cm) | 86.7% | 90% | 100% | 100% |

**Precision = 1.000 dans tous les cas** — le détecteur ne déclenche jamais de fausse alerte sur les points non modifiés (0/55 sur le test propre). C'est un choix de calibration à assumer explicitement : le détecteur actuel est **conservateur**, calibré pour éviter la fatigue d'alerte plutôt que pour maximiser la détection précoce des faibles déviations.

**Points de bascule (recall point-level ≥ 80%) :**
- Spike : 3.0σ (268 µS/cm), F1=0.929
- Burst : 3.0σ (268 µS/cm) au niveau point, mais recall événement = 100% dès 2.0σ (179 µS/cm) — pertinent car une pollution réelle dure plusieurs pas de temps, pas un seul point isolé
- Dip : 2.5σ (223 µS/cm), F1=0.929 — **le type le plus facile à détecter à toutes les intensités testées**

**Sous 1.0σ, les anomalies sont statistiquement indiscernables des résidus normaux** — résultat attendu et interprétable, pas une limite à corriger.

### 9.3 Investigation d'une contradiction apparente (dip) — résolue

La première validation (occurrence unique, dip à −3σ_EC = −590 µS/cm) donnait un score de 0.878, plus bas que spike/burst (1.000) — suggérant à tort que le dip serait plus difficile à détecter. Le balayage d'intensité montre l'inverse : le dip est le type le plus facile à détecter à toutes les amplitudes. Investigation menée pour résoudre la contradiction :

1. **Distribution des résidus train+val** : légère asymétrie confirmée (54% résidus négatifs vs 46% positifs), mais queue positive plus longue en amplitude (p99=+209 µS/cm vs p1=−178 µS/cm) — l'hypothèse initiale (résidus négatifs plus fréquents) était incomplète.
2. **Position du point testé** : non responsable — un test à 15 positions différentes avec la même amplitude (−590 µS/cm) donne le même score (0.878) partout. Le score dépend uniquement de l'amplitude, pas de la position.
3. **Cause réelle identifiée — définitions de σ incompatibles entre les deux tests** : le premier test utilisait σ = écart-type d'EC brut (196.8 µS/cm), le balayage utilise σ = écart-type des résidus (89.4 µS/cm) — un facteur 2.2×. Le "dip à 3σ" du premier test correspondait en réalité à **6.6σ_résidu**, un régime totalement différent des 0.5-3.0σ_résidu testés dans le balayage.
4. **Mécanisme exact — raffiné** : le plafond du score côté négatif n'est pas une propriété diffuse de la distribution des résidus, mais est fixé par **un seul point d'entraînement** : le résidu le plus anomalique observé dans train+val avait un score brut de 0.8075, direction positive — c'est ce point unique qui calibre le maximum du MinMaxScaler (normalisé à 1.000). Tout résidu négatif, aussi extrême soit-il, produit un score brut plafonnant sous ce point de référence (vérifié : de −1σ à −8σ_résidu, le score sature à 0.878 dès −3σ et n'évolue plus). **Fragilité à noter** : cette calibration dépend d'un seul point d'entraînement extrême — si ce point était lui-même un résidu atypique/bruité plutôt qu'un signal réel, tout le plafond de normalisation en hériterait. À surveiller si le modèle est réentraîné sur une période différente.

**Conclusion retenue pour le rapport/papier :** utiliser exclusivement les résultats du balayage d'intensité (§9.2). Le dip est le type le plus facile à détecter à faible/moyenne amplitude, point de bascule à 2.5σ_résidu. **Implication à retenir pour l'usage du score** : les scores d'anomalie ne sont comparables entre eux que du même côté (dip vs dip, spike vs spike) — comparer un score de dip à un score de spike pour hiérarchiser leur sévérité serait trompeur, à cause du plafond asymétrique documenté ci-dessus.

**Leçon méthodologique** : la validation à occurrence unique (n=1 par type) était insuffisante — la conclusion initiale sur le dip était un artefact d'un choix de σ non documenté et d'un seul point de mesure, pas un résultat robuste. Le protocole à répétitions multiples (n=10-15) et intensités progressives est celui à conserver pour toute validation future.

### 9.4 Confirmation du mécanisme SHAP — résultat clé

Sur les 3 anomalies détectées, l'explication SHAP montre systématiquement le même mécanisme déjà identifié en §6.2 : `EC_roll3_mean` domine avec une direction **négative** alors que le résidu est **positif** — la rolling mean à 3 jours tire la prédiction vers le bas au moment précis où l'EC réel monte brusquement. C'est la preuve empirique directe que les deux couches du pipeline (prédiction XGBoost + détection résidu/Isolation Forest) sont complémentaires : là où la prédiction se trompe le plus (retournement de tendance mal capté), le résidu est le plus grand, et c'est exactement là que le détecteur d'anomalie sonne l'alarme. Ce n'est plus une hypothèse de conception mais un résultat vérifié.

### 9.5 Prochaines étapes possibles

- Validation plus rigoureuse avec anomalies synthétiques d'intensité variable (γ plus faible, à la A-12) pour tester la limite réelle de détection
- Réplication du même protocole (résidu + Isolation Forest + SHAP) sur pH et Turbidity
- Formalisation du contrat d'interface CS/IE pour exposer `AnomalyExplanation` (score + SHAP) à l'équipe dashboard

## 10. WQI recalculé (point bloquant depuis semaine 2 — résolu)

### 10.1 Méthode

Weighted Arithmetic Water Quality Index (WAWQI), méthode Brown et al. (1972), standards BIS IS 10500:2012. Calculé sur pH, EC, Turbidity (TDS hors scope). Fichiers produits :
- `data/processed/c1_with_wqi.csv` — dataset avec colonnes d'audit (Qi_pH, Qi_EC, Qi_Turbidity, W_*)
- `reports/wqi_comparison.png` — comparaison visuelle ancien vs nouveau WQI
- `reports/wqi_methodology.txt` — méthodologie documentée et citable

### 10.2 Résultats et cohérence physique

- **Turbidité domine le WQI** (poids W=62.8%) — cohérent avec un site minier où les tailings en suspension dominent la variabilité de qualité d'eau (médiane 40 NTU, max 170 NTU)
- **WQI médian = 519, ~75% de l'année classée "Unsuitable"** — cohérent avec de l'eau de surface non traitée en zone minière
- **Corrélation r=0.9964 avec l'ancien WQI incohérent** — le signal temporel (ranking relatif) était donc déjà globalement correct dans le dataset original, mais sans méthode transparente ni citable. Le recalcul ne change pas la tendance, il la rend défendable.

### 10.3 Limitation à documenter explicitement dans le papier (ne pas laisser en simple commentaire de code)

**Le poids d'EC dans le WQI final est structurellement très faible (W≈0.21%)**, du fait de la formule standard Wi ∝ 1/Si appliquée à un standard EC (Si) élevé. Conséquence directe : EC — l'un des 3 paramètres pourtant modélisés en détail dans ce projet (prédiction XGBoost + anomaly detection + SHAP) — **contribue presque nul au WQI composite**. Ce n'est pas une erreur de calcul (la formule WAWQI est standard et correctement appliquée), mais un résultat qui mérite d'être assumé et expliqué dans le papier plutôt que découvert par un reviewer : la valeur du travail sur EC ne réside pas dans sa contribution au WQI composite, mais dans le monitoring et la détection d'anomalies propres à ce paramètre (ex. contamination saline, infiltration, rejets industriels) — un argument à formuler explicitement pour ne pas donner l'impression que EC est un paramètre secondaire mal choisi.

## 11. Prévision multi-pas générique (résidu par étape)

Module ajouté pour répondre à un besoin opérationnel : produire une trajectoire de prévision sur 48-72h, indépendamment de la fréquence de mesure du dataset (journalière actuellement, potentiellement horaire en déploiement réel) — objectif de tendance approximative (ex. anticiper le besoin de surveillance nuit/week-end), pas de précision fine pas à pas.

### 11.1 Architecture

- `frequency_detector.py` : détecte automatiquement l'intervalle entre mesures (médiane des écarts entre timestamps), calcule le nombre de pas nécessaires (horizon_heures ÷ fréquence)
- Feature engineering généralisé : fenêtres exprimées en heures physiques (pas en nombre de lignes fixe), converties dynamiquement selon la fréquence détectée
- `recursive_forecaster.py` : `MultiStepForecaster`, générique (réutilise n'importe quel `ParameterModel` déjà entraîné) — boucle récursive : prédit, injecte la prédiction dans la fenêtre historique, prédit le pas suivant

### 11.2 Résultats sur EC (résolution journalière — 2-3 pas seulement)

Fréquence détectée : 24.00h exactement (365 dates, aucun trou). Pour horizon 48h/72h → 2/3 pas seulement (limite imposée par la résolution journalière du dataset actuel).

| Pas | Horizon | RMSE | MAE | Biais |
|---|---|---|---|---|
| 1 | 24h | 52.71 µS/cm | 41.69 | +15.36 |
| 2 | 48h | 52.45 µS/cm | 42.84 | +16.05 |
| 3 | 72h | 54.12 µS/cm | 41.70 | +17.51 |

RMSE quasi stable (+2.7% du pas 1 au pas 3) — proche de la référence one-step (53.48 µS/cm).

### 11.3 Deux nuances importantes, à ne pas omettre dans le papier

**Biais croissant, cohérent avec un mécanisme déjà identifié.** Le biais grandit systématiquement (+15.4 → +16.1 → +17.5 µS/cm) malgré un RMSE stable. Cohérent avec l'observation SHAP déjà documentée (§9.4) : le modèle sous-estime les hausses brusques (rolling mean qui tire la prédiction vers le bas pendant un retournement). En récursif, cette sous-estimation se reboucle légèrement à chaque pas — le biais croissant est une confirmation supplémentaire du mécanisme, pas un artefact séparé.

**Résultat valide pour 2-3 pas journaliers, PAS extrapolable à un horizon à haute fréquence sans nouveau test.** La raison pour laquelle le récursif fonctionne bien ici (EC_lag1 domine largement les features, peu sensible à l'erreur accumulée sur seulement 2-3 itérations) ne garantit rien sur un horizon à 72 pas horaires — bien plus d'itérations pour que l'erreur se propage. **À retester explicitement** dès que des données à fréquence plus fine seront disponibles (via le module de réentraînement déjà en place) — ne pas présenter la conclusion actuelle comme une validation générale de la méthode récursive, seulement comme valide dans ce régime précis (peu de pas, dataset journalier).

### 11.4 Figures et schéma

- `fig_13_recursive_forecast_example` — trajectoire de prévision récursive avec incertitude croissante
- `fig_14_error_growth_by_step` — RMSE/MAE par pas, caractérise honnêtement la dégradation avec l'horizon
- Schéma d'architecture (Mermaid) : dataset → détection fréquence → calcul n_steps → boucle récursive → sortie (trajectoire + incertitude)

## 12. Figures et schémas (papier NILES + rapport de stage)

Ensemble complet généré dans `reports/figures/` (300 dpi, PNG+PDF) :

| # | Figure | Contenu |
|---|---|---|
| 01 | Séries temporelles EDA | pH, EC, Turbidity sur 365 jours |
| 02 | Diagnostic de bruit EC | Rolling std 7j, justifie l'absence de débruitage |
| 03 | Distribution EC par split | KDE train/val/test, illustre le décalage de niveau |
| 04 | Comparaison out-of-the-box | RMSE val, persistence/RF/XGBoost/SVR avant tuning |
| 05 | Comparaison après tuning | RMSE val, XGBoost tuned meilleur |
| 06 | Exploration de features | 9 approches testées, baseline retenue |
| 07 | Prédit vs réel + résidus | Test set, illustre le retournement de tendance |
| 08 | Importance SHAP | **Corrigée** : moyenne sur tout le test set (n=55), pas n=5 |
| 09 | Exemple d'anomalie détectée | Burst synthétique + score au cours du temps |
| 10 | Mécanisme du plafond de score | **Corrigée** : onset du plateau négatif à −2.50σ (pas −2.52σ) |
| 11 | Erreur relative journalière | Moyenne 5.33%, médiane 4.69%, 0/55 jours >20%. **Corrigée** : citations de seuils normatifs (ISO 15839, EPA 120.1, Wang et al., El-Shafeiy et al.) retirées faute de vérification — repères visuels non attribués |
| 13 | Exemple de prévision récursive multi-pas | Trajectoire sur 2-3 pas (48-72h) avec incertitude croissante |
| 14 | Croissance de l'erreur par pas | RMSE/MAE/biais pas par pas — dégradation douce confirmée (biais croissant, RMSE quasi stable) sur 2-3 pas journaliers |

Schémas complémentaires (architecture, hors figures de données) :
- Architecture du pipeline complet (data → prédiction → SHAP → anomaly detection → interface CS/IE)
- Protocole de validation train/val/test (discipline "test touché une seule fois")
- Gantt planning réel vs concept note (dates à ajuster avant usage — dates placeholder actuellement)

### Traçabilité des corrections apportées

Deux erreurs ont été détectées et corrigées avant de figer ce lot de figures — à mentionner comme illustration de la rigueur de vérification systématique appliquée sur l'ensemble du projet (cf. incidents similaires déjà documentés : plafond de score investigué deux fois, incohérences MAE/meilleur-modèle relevées dans S-3/N-2 lors de la revue de littérature) :

1. **Fig 08** : la première version utilisait seulement 5 lignes de test pour l'importance SHAP — non représentatif. Recalculée sur l'ensemble du test set (n=55) ; changements notables : EC_roll7_mean double d'importance (5.1→10.3), EC_lag1 triple (1.5→3.2), EC_roll3_mean reste dominant mais moins écrasant (73%→63%). pH_lag1 confirmé à zéro sur n=55 — résultat robuste, pas un artefact d'échantillon.

2. **Fig 11** : citations de 4 sources (ISO 15839, EPA Method 120.1, Wang et al. 2023, El-Shafeiy et al. 2023/S-3) pour justifier des seuils MAPE 10%/20%, non vérifiées lors de la génération initiale. Vérification demandée explicitement : aucune des 4 sources n'a pu être confirmée comme mentionnant ces seuils précis. Citations retirées, lignes reformulées comme repères visuels génériques sans attribution.

**Principe retenu pour la suite du projet** : toute citation, seuil normatif, ou affirmation générée automatiquement dans une figure ou un texte doit être vérifiée explicitement avant d'être figée — ne jamais accepter une source "plausible" sans confirmation.

## 13. Note technique — piège de vérification XGBoost (à connaître pour l'équipe)

**Piège découvert lors d'une vérification de cohérence des figures de généralisation** : appeler `model.get_booster().save_config()` après avoir chargé un modèle via `XGBRegressor().load_model(path)` sur un wrapper sklearn fraîchement instancié **ne retourne PAS les hyperparamètres réels avec lesquels le modèle a été entraîné** — il retourne les paramètres par défaut du wrapper Python (`max_depth=6, eta=0.3` par défaut), pas ceux du booster C++ sous-jacent effectivement chargé.

**Conséquence pratique** : une inspection rapide de `ec_xgboost_v1_final.json` via cette méthode a semblé indiquer que le modèle de production n'était pas tuné (depth=6/lr=0.3 au lieu de depth=3/lr=0.01 attendu) — fausse alerte, confirmée par une vérification indépendante (RMSE recalculé depuis le JSON = 53.4838 µS/cm, identique à la valeur documentée ; prédictions bit-for-bit identiques entre le modèle chargé et un modèle réentraîné à zéro avec les vrais hyperparamètres).

**À retenir pour toute vérification future de modèle sauvegardé/rechargé** : ne jamais se fier à `save_config()` sur un wrapper vide pour auditer les hyperparamètres réels — vérifier plutôt via les prédictions elles-mêmes (comparaison bit-for-bit à un modèle réentraîné avec la config attendue), ou consulter les hyperparamètres au moment de l'entraînement original, pas après rechargement.
