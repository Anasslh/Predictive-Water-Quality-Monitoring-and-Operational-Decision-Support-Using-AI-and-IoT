# Rapport — Généralisation du pipeline (src/pipeline/)
**Projet :** Predictive Water Quality Monitoring and Operational Decision Support Using AI and IoT
**Rôle :** Computer Science Student — SESC, Nile University, Giza
**Portée :** pipeline générique d'onboarding de capteurs — testé sur les 3 paramètres du dataset C-1 (EC, pH, Turbidity)
**Statut :** figé, généricité prouvée empiriquement

---

## 1. Objectif

Passer d'une architecture "un fichier/classe par paramètre écrit à la main" à une architecture pilotée par configuration — nécessaire pour envisager un déploiement à plusieurs capteurs (cf. discussion sur la scalabilité, `notes_decisions_architecture.md` §5) sans dupliquer le code à chaque nouveau paramètre.

**Principe non négociable posé dès la conception** : aucune décision automatique silencieuse. Le choix d'algorithme, la validation du modèle réentraîné, et la promotion en production passent toujours par une confirmation humaine explicite — cohérent avec `approval.py` déjà en place pour le réentraînement.

## 2. Architecture

| Fichier | Rôle |
|---|---|
| `config/sensors_config.json` | Config déclarative : colonne, nom de paramètre, unité, standard WQI associé — aucun paramètre modélisé "par surprise" sans être listé explicitement |
| `timestamp_detector.py` | Auto-détection de la colonne timestamp parmi une liste de candidats (parsing + monotonie croissante) |
| `model_benchmark.py` | `benchmark_models()` : 33 configurations (RF×9, XGBoost×18, SVR×6) comparées, retourne un rapport classé — **aucune sélection automatique** |
| `orchestrator.py` | `onboard_new_parameter()` (génère le rapport), `submit_benchmark_choice()` (fige le modèle après confirmation humaine explicite), `run_manual_benchmark()` (ré-évaluation à la demande) |
| `quick_noise_diagnostic()` | Diagnostic de bruit (CV du signal) — classe Low/Moderate/High, déclenche l'évaluation systématique log-vs-raw si CV élevé |

**Choix de conception : timestamp auto-détecté, paramètres explicitement configurés.** L'auto-détection totale des paramètres a été jugée trop risquée — un dataset réel contient souvent des colonnes qui ressemblent à des paramètres sans en être (cf. le WQI original du dataset, incohérent, qui aurait pu être auto-détecté et modélisé à tort). Le timestamp, à plus faible risque, est auto-détecté ; les paramètres à modéliser restent une décision humaine explicite dans la config.

## 3. Résultat empirique inattendu — le log-transform n'aide pas, même à CV élevé

Testé sur Turbidity (CV signal = 86.6%, classé "High" par le diagnostic automatique) :

| Version de la cible | Val RMSE | Rel RMSE | Modèle gagnant |
|---|---|---|---|
| Turbidity brute | 36.23 NTU | 78.4% | XGBoost depth=2, n=50 |
| log1p(Turbidity) | 39.92 NTU (reconverti) | 86.3% | SVR |

**Le log-transform dégrade la performance de +10.2%**, contredisant l'heuristique standard "CV élevé → transformation log recommandée". Explication mécanistique identifiée : les features de lag/rolling sont calculées sur la cible transformée — si la dynamique temporelle utile (pics de turbidité brutaux et symétriques) réside dans l'espace brut, la compresser en espace logarithmique détruit la structure que les features lag exploitent, plutôt que de l'améliorer.

**Conséquence pour l'architecture** : `benchmark_models()` teste désormais systématiquement les deux versions (brute et log) dès que le diagnostic signale un CV élevé, et présente les deux classements côte à côte dans le même rapport — la décision finale (quelle version retenir) reste humaine, cohérent avec le principe de non-automatisation silencieuse.

**Valeur pour le papier** : ce résultat négatif, bien expliqué mécanistiquement, est une contribution méthodologique en soi — un contre-exemple empirique documenté à une heuristique répandue, pas juste un test technique routinier.

## 4. Validation de généricité — testée sur les 3 paramètres du dataset, pas juste affirmée

### 4.1 Les trois modèles de production, officiellement figés

Contrairement aux dry runs initiaux (confirmation simulée, but : prouver l'absence de code codé en dur), les 3 paramètres du dataset C-1 ont chacun un modèle réellement figé après rapport de benchmark et confirmation humaine explicite — même discipline que EC (test touché une seule fois, après le freeze) :

| Paramètre | Modèle | Hyperparamètres | RMSE test | Skill vs persistance |
|---|---|---|---|---|
| EC | XGBoost | depth=3, n=50, lr=0.01 | ~56 µS/cm | +14 RMSE (~20%) |
| pH | XGBoost | depth=2, n=50, lr=0.01 | 0.854 pH | +22.7% |
| Turbidity | XGBoost | depth=2, n=50, lr=0.01 | 36.63 NTU | +22.4% |

**Convergence notable** : XGBoost gagne sur les 3 paramètres, chacun avec sa propre profondeur optimale trouvée indépendamment (depth=3 pour EC, depth=2 pour pH/Turbidity) — pas d'uniformité artificielle imposée, chaque paramètre a convergé vers sa configuration simple optimale via un vrai tuning.

### 4.2 Le skill score vs persistance — métrique de référence transversale, plus robuste que le RMSE relatif brut

**Problème découvert sur Turbidity** : le RMSE relatif (RMSE/moyenne) atteint 99.89% sur le test — un chiffre qui, pris isolément, semble indiquer un échec de modélisation total. Investigation :

- **Décalage saisonnier confirmé** (même mécanisme que sur EC) : Turbidity moyenne train=46.0 NTU vs test=36.7 NTU (-21%), médiane train=42.2 vs test=27.9 NTU, part des valeurs >100 NTU passant de 15% (train) à 5% (test). Le test set (derniers 15% de l'année, saison sèche) contient mécaniquement moins d'épisodes de forte turbidité que le train (saison des pluies).
- **Le RMSE relatif est gonflé artificiellement par ce décalage** : la baseline train-mean (46 NTU) appliquée telle quelle donne un RMSE de 37.0 NTU sur le test — quasi identique au XGBoost (36.6 NTU). Ce n'est donc pas que le modèle échoue à capter un signal, c'est que le dénominateur (moyenne test) s'est effondré, gonflant mécaniquement le ratio.
- **Le vrai signal utile se révèle en comparant à la persistance (lag-1), pas à la moyenne** : XGBoost bat la persistance de +10.6 NTU (22.4% de réduction du RMSE) — un skill réel sur la dynamique court-terme, masqué par l'artefact de normalisation du RMSE relatif.

**Conclusion méthodologique retenue pour le papier** : le **skill score vs persistance** ((RMSE_persistance − RMSE_modèle) / RMSE_persistance) est la métrique de référence à privilégier pour comparer l'apport réel du ML entre paramètres sur ce dataset — elle reste stable et interprétable (+20-23% sur les 3 paramètres) là où le RMSE relatif brut peut être trompeur en présence d'un décalage saisonnier marqué entre train et test. Le RMSE relatif reste utile pour discuter de la difficulté intrinsèque du paramètre (CV du signal : 87% pour Turbidity vs 12% pour pH), mais ne doit jamais être interprété seul, ni comparé sans préciser explicitement sur quel split (val ou test) chaque score a été calculé.

### 4.3 Résultats de comparaison des 3 paramètres (issus des dry runs de benchmark)

| Paramètre | CV signal | Diagnostic | Algorithme gagnant | Rel RMSE (val) |
|---|---|---|---|---|
| EC (référence, déjà figé) | 23.4% | — | XGBoost (tuné séparément) | 7.47% |
| pH | 11.9% | Low–moderate (log non évalué) | XGBoost depth=5, n=50, lr=0.01 | 10.42% |
| Turbidity | 86.6% | High (log évalué, rejeté) | XGBoost depth=2, n=50, lr=0.01 | 78.4% |

**Résultat généralisable identifié** : relation claire entre variabilité du signal (CV) et précision atteignable — pH et EC (signaux stables) prédictibles à ~10%, Turbidity (signal chaotique, dépendant d'événements pluvieux non capturés par les features lag) intrinsèquement à ~78%. Ce n'est pas un échec de modélisation mais une limite physique du signal lui-même — à formuler ainsi systématiquement dans le papier, jamais comme un chiffre nu.

**XGBoost gagne systématiquement sur les 3 paramètres** — confirme empiriquement (pas par supposition) le choix de standard de facto pour ce contexte de déploiement (faible volume, fréquence journalière, edge deployment), cohérent avec la décision documentée dans `notes_decisions_architecture.md` §4.

**Vérification de non-hardcoding** : 4 fichiers d'infrastructure (`orchestrator.py`, `model_benchmark.py`, `timestamp_detector.py`, `__init__.py`) inspectés pour les 3 noms de capteurs (EC, pH, Turbidity) — confirmé : 0 référence codée en dur. Erreur méthodologique évitée en cours de route : une première vérification n'avait testé que 2 noms sur 3 (oubli de pH), corrigée avant de conclure.

**Prudence méthodologique maintenue, cohérente avec le reste du projet** : l'écart entre le modèle gagnant et le 2e rang n'est jamais statistiquement significatif sur ces volumes de validation (n≈53) — testé explicitement par bootstrap (ex. Turbidity : IC 95% à [-2.35, +7.68] NTU, croise zéro). Les hyperparamètres exacts restent à confirmer sur le test set réservé par la personne en charge de chaque paramètre, jamais figés sur le seul résultat du val set.

## 5. Ce qui est acquis

- Un seul point d'entrée générique (`onboard_new_parameter`) pour ajouter un nouveau capteur, sans écrire de nouveau fichier/classe
- Prouvé fonctionnel sur 3 profils de signal très différents (stable/pH, saisonnier/EC, chaotique/Turbidity)
- Compatible et branchable directement sur les modules déjà génériques : anomaly detection (résidu + Isolation Forest + SHAP), réentraînement (drift + validation + vérification humaine + persistance), prévision multi-pas (récursif, fréquence auto-détectée)
- Une vraie contribution méthodologique indépendante pour NILES (le contre-exemple log-transform), pas seulement un exercice d'ingénierie

## 6. Limites et travail restant

- Testé sur un seul site (Ramgarh/C-1) — la relation CV→précision, bien que cohérente et interprétable, mériterait une confirmation sur un second site pour être présentée comme un résultat pleinement généralisable
- Le WQI recalculé (pondération par paramètre) reste actuellement pensé pour 3 paramètres fixes — pas encore généralisé si un 4e capteur qualité d'eau s'ajoutait (documenté comme perspective future, pas un défaut caché)
- Le seuil de CV déclenchant l'évaluation log-vs-raw (actuellement 60%) est empirique et basé sur un seul contre-exemple (Turbidity) — à surveiller/ajuster si d'autres paramètres à CV élevé sont ajoutés plus tard

## 7. Nettoyage final — architecture vraiment uniforme (pas de traitement spécial pour EC)

Suite à une question légitime soulevée en cours de projet (*"le système est-il vraiment généralisable, ou est-ce qu'on continue de traiter EC à part ?"*), trois résidus d'un traitement spécial hérité du développement initial d'EC (premier paramètre construit, servant de cas de test canonique) ont été identifiés et corrigés :

1. **Approbations fantômes nettoyées** — 8 fichiers résiduels de tests/démos précédents (5 JSON de `PendingApproval` + 3 modèles candidats PKL) supprimés de `models_store/retrain_demo/pending_approvals/` — `get_system_status("EC").pending_approvals` confirmé à 0.
2. **Structure de stockage uniformisée** — `models_store/retrain_demo/` (nom hérité des scripts de démonstration) migré vers `models_store/ec/`, strictement symétrique à `models_store/ph/` et `models_store/turbidity/`. Migration vérifiée sans régression : RMSE test recalculé bit-for-bit identique avant/après (53.4838 µS/cm).
3. **Vérification du code jugé "possiblement mort"** — le dict `_COVARIATES` dans `feature_engineering.py`, suspecté legacy sur la base d'une supposition non vérifiée, s'est révélé activement utilisé par `build_features()`, elle-même importée par 6 fichiers du projet (scripts d'exemple, évaluation, forecasting, génération de figures, notebooks). Aucune suppression effectuée — le soupçon initial était infondé, corrigé après vérification plutôt que d'agir sur une supposition.

**Point de clarification important** : EC, pH, et Turbidity ont maintenant un statut strictement équivalent dans l'architecture — EC n'est pas "le paramètre principal avec les deux autres en exemple", c'est un paramètre parmi trois traités uniformément par le système générique, EC servant simplement de premier cas de validation historique du fait d'avoir été développé en premier.

**Leçon retenue, cohérente avec toute la démarche du projet** : même un diagnostic qui semble raisonnable (code qui a l'air legacy) doit être vérifié avant action, pas juste supposé — le même principe qui a guidé toutes les autres investigations du projet (le plafond du score d'anomalie, les incohérences relevées dans la littérature, le faux-positif sur les hyperparamètres XGBoost via `save_config()`).

---

*Document de synthèse — à déposer dans `reports/` du repo, à côté des rapports EC, anomaly detection (semaine 6), et réentraînement.*
