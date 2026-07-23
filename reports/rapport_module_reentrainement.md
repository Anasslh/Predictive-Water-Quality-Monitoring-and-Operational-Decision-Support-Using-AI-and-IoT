# Rapport — Module de réentraînement générique (src/retraining/)
**Projet :** Predictive Water Quality Monitoring and Operational Decision Support Using AI and IoT
**Rôle :** Computer Science Student — SESC, Nile University, Giza
**Portée :** module générique, indépendant du paramètre — testé sur EC, réutilisable tel quel pour pH/Turbidity
**Statut :** figé, calibré empiriquement

---

## 1. Objectif et positionnement

Contrairement au pipeline EC (§1-11 de `rapport_ec_pipeline_complet.md`), ce module n'est pas spécifique à un paramètre — c'est la brique qui transforme un modèle entraîné une fois en un système maintenu dans le temps : détection de dérive, réentraînement périodique, validation avant remplacement, gestion de versions.

**Contrainte de conception posée dès le départ** : le module doit fonctionner aussi bien pour une station à haute fréquence de collecte que pour une station qui accumule des données lentement — pas d'hypothèse sur le rythme d'arrivée des données.

## 2. Architecture

| Fichier | Rôle |
|---|---|
| `data_loader.py` | `load_available_data()` — seul point du module qui connaît la source des données. Aujourd'hui : lecture d'un CSV qui grossit avec le temps. Isolé volontairement pour qu'un changement de source (fichiers séparés par période, base de données) ne touche que ce fichier. |
| `drift_detector.py` | `check_drift()` — détecte si le modèle actuel nécessite un réentraînement (voir §3 pour la logique retenue après tests). |
| `retrain_manager.py` | `RetrainManager` — orchestre le cycle complet : vérifie le volume de nouvelles données, déclenche le drift check, réentraîne un candidat, valide avant remplacement. |
| `model_versioning.py` | Sauvegarde horodatée, conserve les 3 versions les plus récentes (rollback possible). |
| `demo_retrain_ec.py` | Script de démonstration/validation sur données EC réelles. |

**Vérification d'indépendance** : aucun des 4 fichiers du module ne mentionne EC, pH, ou Turbidity. La configuration par paramètre se limite à 2 lignes dans le script de démo (classe du modèle + fonction de feature engineering) — confirmé en inspectant le code, pas juste supposé.

## 3. Décisions calibrées empiriquement (le cœur de ce rapport)

Ce module s'est construit par itérations testées, pas par un choix de conception figé a priori — la même discipline que celle appliquée sur le pipeline EC.

### 3.1 Volume minimum avant de vérifier (`MIN_NEW_ROWS`)

- **Valeur initiale proposée** : 30 nouvelles lignes
- **Valeur finale retenue : 60**
- **Raison du changement** : à 30, le test set EC (55 lignes) suffisait à lui seul à déclencher une vérification, ce qui exposait le RMSE ratio à une variabilité statistique trop importante sur un échantillon aussi petit. Passer à 60 impose une accumulation d'environ 2 mois de données journalières avant la première tentative de vérification — un compromis assumé entre réactivité et fiabilité statistique.

### 3.2 Critère de détection de drift — de OR à AND à RMSE-seul

**Itération 1 (OR — écartée)** : drift = KS significatif **OU** RMSE dégradé.
- Problème détecté : sur données réelles EC, le passage saison de mousson → saison sèche produit un vrai changement de distribution des résidus (KS p=0.0083) alors que le modèle **s'améliore** sur la nouvelle période (RMSE ratio=0.566). Avec OR, ce cas déclenchait un réentraînement inutile — sur un site à cycle saisonnier marqué, ça se serait reproduit deux fois par an sans justification.

**Itération 2 (AND — écartée après test approfondi)** : drift = KS significatif **ET** RMSE dégradé.
- Corrige le cas ci-dessus (le modèle qui s'améliore n'est plus signalé à tort).
- **Angle mort découvert par un test dédié** : une dégradation modérée du modèle, exprimée comme un simple scaling des résidus existants (même forme de distribution, magnitude différente), ne change quasiment pas la *forme* de la distribution — donc KS n'a presque aucune puissance statistique pour la détecter à n≈55 (taux de détection empirique : 7-9%, à peine au-dessus du taux de faux positifs de référence α=5%, donc statistiquement indiscernable du hasard). Résultat concret : une dégradation de +20% de RMSE ne déclenchait aucun réentraînement avec AND, parce que KS restait silencieux.

**Itération 3 (finale, retenue) : RMSE ratio seul comme critère de déclenchement, KS conservé comme information diagnostique journalisée mais non bloquante.**
- Le KS renseigne l'opérateur sur la *nature* du changement (rupture structurelle vs dérive lente) sans intervenir dans la décision d'agir.
- La décision d'agir repose uniquement sur ce qui compte opérationnellement : la performance du modèle se dégrade-t-elle réellement.
- Seuil retenu : `rmse_ratio_threshold = 1.10` (10% de tolérance), resserré depuis 1.20 initial pour compenser la perte du signal KS comme filet de sécurité supplémentaire.

### 3.3 Validation des trois régimes (tests dédiés, pas seulement le cas favorable)

| Scénario | RMSE ratio | KS | Décision (RMSE-seul) | Correct ? |
|---|---|---|---|---|
| Glissement saisonnier réel, modèle qui s'améliore | 0.566 | p=0.0083 (shift détecté) | Skip (pas de réentraînement) | Oui |
| Dégradation modérée simulée (+17-20%, scaling des résidus) | 1.196 | p=0.119 (pas de signal) | Retrain déclenché | Oui — corrige l'angle mort d'AND |
| Variation normale sous le seuil (~+13%) | 1.030 | p=0.709 | Skip | Oui |

### 3.4 Garde-fou indépendant, conservé quel que soit le critère de déclenchement

Le critère de drift décide *quand tenter* un réentraînement — la **porte d'acceptation** du `RetrainManager` reste le vrai filet de sécurité : le candidat réentraîné n'est promu en production que si son RMSE sur le test set réservé est meilleur ou équivalent à l'actuel (tolérance 2%). Même si le critère de déclenchement se trompait, ce garde-fou empêche qu'un modèle réellement pire ne remplace l'ancien.

### 3.5 Gestion de versions

3 versions les plus récentes conservées (horodatées), rollback possible. Modèle de production (`models_store/ec_xgboost_v1_final.json`) jamais modifié par les tests de démonstration — confirmé (date de modification inchangée après tous les runs de test).

## 4. Ce qui est acquis pour la suite

- Module testable et démontré fonctionnel sur EC, sans aucune dépendance codée en dur au paramètre
- Pour pH/Turbidity : copier `demo_retrain_ec.py`, changer 2 lignes (classe du modèle, fonction de features)
- Une vraie section méthodologie indépendante pour NILES, distincte du pipeline EC — le raisonnement OR→AND→RMSE-seul est en soi une contribution méthodologique testée empiriquement, pas un choix de conception arbitraire

## 5. Limites et travail restant

- Calibration (seuils 60 lignes, ratio 1.10) faite sur un seul site (Ramgarh/EC) — à revalider si déployé sur un site à dynamique très différente
- Pas encore intégré à la boucle de vérification humaine (à la PADSV/A-12) — le réentraînement se décide et s'exécute automatiquement, sans étape de confirmation humaine avant promotion. À considérer si le projet veut aligner ce module avec le principe de vérification sélective déjà établi pour l'anomaly detection.
- Le test de puissance statistique du KS (7-9% de détection sur scaling pur) mériterait d'être documenté comme résultat méthodologique à part entière dans le papier — c'est un résultat généralisable au-delà de ce projet (limite connue de KS sur des changements de magnitude plutôt que de forme).

## 6. Vérification humaine avant promotion (point ouvert du §5, résolu)

Ajout d'une étape de vérification humaine avant toute promotion en production, à la manière de la vérification sélective de PADSV (A-12) — l'humain n'est sollicité que sur la décision à plus fort enjeu (remplacer le modèle en prod), pas à chaque étape intermédiaire.

### 7.1 Architecture

| Fichier | Rôle |
|---|---|
| `approval.py` | `PendingApproval` (dataclass avec RMSE actuel/candidat, raison du drift, période d'entraînement), `submit_for_approval()`, `record_decision()`, `list_pending_approvals()` |
| `model_versioning.py` (étendu) | `write_current_model_pointer()`, `load_current_model()`, `read_current_model_pointer()` |

**Principe central : aucune promotion automatique n'est structurellement possible** — vérifié par assertion de code que `attempt_retrain()` retourne toujours `promoted: False`, quelle que soit la branche exécutée. Un candidat qui passe la porte d'acceptation RMSE est mis en attente (`pending_approval`), jamais promu directement.

### 7.2 Validation — comportement vérifié

- **Aucune modification du modèle de production sans approbation explicite** : vérifié par comparaison MD5 du fichier `ec_xgboost_v1_final.json` avant/après soumission, avant/après approbation d'un autre candidat, avant/après rejet — identique à chaque étape.
- **Approbation** : le candidat est versionné (horodaté) dans le store, jamais en écrasant le JSON de production original.
- **Rejet** : le fichier candidat est supprimé, aucune trace dans le store de production ; le JSON d'audit reste archivé (`status: rejected`) pour traçabilité.

### 7.3 Problème découvert et corrigé — persistance après redémarrage

**Problème identifié** : `apply_approved_candidate()` mettait à jour l'état uniquement en mémoire. Un redémarrage du processus (nouveau déploiement, nouvelle exécution) aurait rechargé l'ancien JSON de production par défaut, perdant silencieusement la trace d'une approbation déjà effectuée — un vrai risque en usage réel, invisible dans une démo mono-session.

**Correction** : ajout d'un pointeur persistant (`models_store/current_model.json`) écrit atomiquement au moment de l'approbation (même transaction que le versioning), contenant le chemin du modèle approuvé, l'identifiant de l'approbation, et la date de promotion. Au démarrage, le système cherche ce pointeur en premier ; s'il n'existe pas (premier démarrage, aucune approbation encore effectuée), il retombe sur le JSON original.

**Test de redémarrage effectué** : simulation d'un nouveau processus (nouvelle instance de `RetrainManager`) après une approbation — confirmé que le pointeur est lu correctement et que la bonne version approuvée est chargée, sans jamais toucher au JSON de production original.

### 7.4 Limite assumée

Le mécanisme de validation humaine est actuellement **simulé** (fonction appelée directement avec un booléen), pas une vraie interface. Documenté explicitement comme tel dans le code — l'architecture prévoit le point de contrôle humain, une vraie UI (notification, dashboard IE) reste un travail futur, cohérent avec le contrat d'interface CS/IE en cours de discussion avec l'équipe.

---

## 7. Compteur de rejets consécutifs (trou identifié et comblé)

**Problème identifié** : sans garde-fou supplémentaire, un modèle qui échoue systématiquement à passer la porte d'acceptation (candidats rejetés cycle après cycle) le ferait silencieusement, indéfiniment — chaque rejet étant un événement isolé, jamais accumulé en signal d'alerte. Risque réel : un problème structurel (algorithme devenu inadapté à un nouveau régime, dérive non capturée par le critère RMSE actuel) resterait invisible tant que personne ne consulte manuellement l'historique des tentatives.

### 8.1 Mécanisme ajouté

- `model_versioning.py` : `write_rejection_counter()` / `read_rejection_counter()` — compteur persistant (survit à un redémarrage, même principe que le pointeur de modèle actuel)
- `retrain_manager.py` : `rejection_alert_threshold=3` (configurable) ; incrémenté et persisté à chaque rejet, remis à zéro à chaque acceptation

### 8.2 Décision sur la fréquence de re-notification — trois options comparées explicitement

| Option | Pattern (seuil=3) | Verdict |
|---|---|---|
| `>= threshold` | Alerte à 3, 4, 5, 6, 7… | Fatigue d'alerte — l'opérateur finit par l'ignorer |
| `== threshold` | Alerte à 3 seulement | Risque de silence permanent si la première alerte est manquée |
| **`% threshold == 0` (retenue)** | Alerte à 3, 6, 9… | Resurgit le signal sans noyer l'opérateur |

**Propriété intéressante du choix retenu** : l'espacement des rappels suit automatiquement la sensibilité configurée — un seuil bas (3) donne des rappels fréquents, un seuil haut (10) des rappels plus espacés, sans paramètre supplémentaire à gérer.

### 8.3 Validation (7 tests)

Confirmé : absence de compteur → 0 ; écriture/lecture ; chargement au démarrage d'un nouveau `RetrainManager` ; déclenchement exact au 3e rejet ; pas de plafond (le compteur continue au-delà) ; re-notification périodique confirmée en sortie réelle (silence à 4-5, alerte à 6, silence à 7) ; remise à zéro sur acceptation.

---

## 8. Fenêtre glissante d'historique (limite de long terme)

**Problème anticipé** : sans limite, l'historique cumulé grandirait indéfiniment (10-20+ ans en usage réel) — pas un problème de coût de calcul (XGBoost reste trivial même sur des milliers de lignes), mais un risque de dérive conceptuelle de très long terme : des données très anciennes pourraient représenter un régime devenu obsolète (évolution de l'activité minière, dérive climatique de long terme), diluant le signal récent plutôt que d'aider le modèle.

### 9.1 Décision retenue — hypothèse théorique, non testable empiriquement à ce stade

**Fenêtre glissante maximale : 5 ans**, poids uniforme sur toute la fenêtre (pas de pondération décroissante par ancienneté).

**Important — ce choix n'a pas pu être calibré empiriquement**, contrairement à toutes les autres décisions de ce module (`MIN_NEW_ROWS`, `rmse_ratio_threshold`, logique de drift) qui ont été testées sur des scénarios réels ou simulés représentatifs. Avec seulement ~1 an de données réelles disponibles (dataset C-1), impossible de comparer empiriquement différentes tailles de fenêtre. Le choix de 5 ans repose sur un raisonnement théorique : suffisamment long pour capturer plusieurs cycles saisonniers avec variabilité inter-annuelle (une seule année ne renseigne pas sur la variabilité d'une année à l'autre — certaines moussons plus intenses, certaines plus sèches), sans intégrer un historique trop ancien potentiellement obsolète.

**Pondération décroissante par ancienneté volontairement écartée** : théoriquement défendable si la dérive du système est progressive, mais introduirait un nouvel hyperparamètre (taux de décroissance) non calibrable avec l'historique actuel — casserait la discipline de calibration empirique maintenue sur tout le reste du module.

**À revalider explicitement une fois ≥ 3 ans de données réelles disponibles** — ne pas présenter cette valeur comme un résultat validé dans le papier, mais comme une hypothèse de conception assumée.

### 9.2 Validation (13 tests)

| Scénario | Historique | Comportement observé |
|---|---|---|
| Historique long (simulé, 8 ans, 2920 lignes) | > 5 ans | Fenêtre appliquée : split sur 1827 lignes (≈5×365.25), `window_applied=True` |
| Historique court (cas réel actuel, ~200-365 lignes) | < 5 ans | Aucune coupe : `window_applied=False`, toutes les lignes utilisées, comportement strictement identique à avant l'ajout de cette fenêtre |

**Confirmation pour EC (cas réel actuel)** : avec ~365 lignes journalières, la fenêtre de 5 ans (≈1826 jours) ne coupe rien — la fonction `_apply_history_window` détecte que tout l'historique est dans la fenêtre et retourne les données sans aucune modification, sans réappel de la fonction de features. Le comportement du pipeline EC reste inchangé.

---

## 9. Figures et schémas

| Visuel | Type | Contenu |
|---|---|---|
| Architecture du flux de réentraînement | Schéma (Mermaid) | data loader → volume gate → drift check → validation → versioning |
| Évolution du critère de drift | Schéma | OR (écartée) → AND (écartée, angle mort) → RMSE-seul (retenue), avec le problème identifié à chaque étape |
| `fig_12_ks_power_analysis` | Figure (données réelles, 300dpi) | Puissance statistique KS vs RMSE ratio en fonction de la magnitude de dégradation simulée (×1.05 à ×1.50), 200 répétitions par magnitude |

**Résultats clés de fig_12** — la preuve empirique de l'angle mort documenté en §3.2 :

| Magnitude (scaling résidus) | Détection KS | Détection RMSE ratio |
|---|---|---|
| ×1.10 (seuil retenu) | 5.2% (quasi aléatoire) | 48.6% (frontière, attendu au seuil) |
| ×1.20 | 10.2% (à peine au-dessus de α=5%) | 81.1% ✓ (franchit 80% de puissance) |
| ×1.50 | 41.0% (encore peu fiable) | 100.0% |

Le KS ne dépasse jamais 80% de puissance statistique sur toute la plage opérationnelle pertinente (×1.05-×1.50) — il reste essentiellement plat entre 4% et 41%. Le RMSE ratio atteint 80% de puissance dès ×1.20 et sature à 100% à partir de ×1.50, confirmant que le seuil `rmse_ratio_threshold=1.10` est bien calibré : suffisamment sensible pour capter une vraie dégradation, sans réagir au bruit normal.

---
-e 
*Document de synthèse — à déposer dans `reports/` du repo, à côté de rapport_ec_pipeline_complet.md.*
