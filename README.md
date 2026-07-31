![Python](https://img.shields.io/badge/python-3.12-3776AB)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)
![Airflow](https://img.shields.io/badge/orchestration-Airflow-017CEE)
![PostgreSQL](https://img.shields.io/badge/data-PostgreSQL-4169E1)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)
![Grafana](https://img.shields.io/badge/monitoring-Grafana-F46800)
![Docker](https://img.shields.io/badge/built%20with-Docker-2496ED)

# Prévision et pilotage du système électrique français

Application MLOps de prévision et d'analyse de la consommation et de la
production électrique en France (RTE) enrichie de données météo (Meteostat
et Open-Meteo, observées **et prévues**). Le projet couvre l'ensemble de la
chaîne : ingestion horaire, validation et détection de dérive, entraînement
avec sélection d'hyperparamètres, prévision multi-horizon avec intervalles
conformes, évaluation continue prévu/réalisé, promotion champion/challenger,
et supervision (Airflow + Grafana), le tout restitué dans un dashboard
Streamlit.

> **La prévision officielle RTE (J-1) est un étalon, jamais une entrée.**
> Elle est collectée et affichée pour situer la performance du modèle, mais
> n'entre ni dans les features, ni dans l'entraînement, ni dans une
> combinaison. La valeur à démontrer est celle d'un modèle **autonome**.

> **Périmètre couvert** : quatre filières de production (solaire, biomasse,
> éolien terrestre, nucléaire) + consommation totale + température. Ni
> l'hydraulique, ni le thermique fossile, ni les échanges transfrontaliers,
> ni les pertes réseau ne sont disponibles : les écarts
> production/consommation affichés dans l'application sont donc **partiels**
> et ne constituent pas le solde électrique national.

## Architecture

```
RTE (API conso + prod) ──────┐
Meteostat (temp. observée)   │
Open-Meteo (temp. observée   ├──► ingestion (scripts/ingest.py, DAG ingest_hourly @hourly)
  + PRÉVUE à échéance J-1)   │      upsert idempotent
RTE J-1 (prévision conso,    │
  étalon externe seulement) ─┘
                              │
                              ▼
                        PostgreSQL (observations, forecasts, forecast_metrics,
                                    pipeline_runs)
                              │
        ┌─────────────────────┼─────────────────────────┐
        ▼                     ▼                         ▼
 entraînement            prévision J+1             évaluation continue
 (pipeline_prevision/    (scripts/forecast.py,     (scripts/evaluate.py,
  components/*, main.py, DAG forecast_daily 06h)    DAG evaluate_daily 06h30)
  scripts/retrain.py)          │                         │
        │                     ▼                         ▼
        │              intervalles conformes      forecast_metrics
        ▼              dynamiques (backtest             │
  champion/challenger    interne récent)                 ▼
  (final_models/ vs                              Grafana (dashboards +
   candidate_models/)                              alerte de dégradation)
        │                                                 │
        ▼                                                 ▼
  MLflow (tracking + registre,                    alert-bridge ──► Airflow API
   alias `champion`/`challenger`)                  (DAG retrain_on_degradation,
        │                                           revalidé contre le seuil
        ▼                                           avant tout réentraînement)
  Streamlit (streamlit_app/, 6 pages :
   Accueil, Vue d'ensemble, Analyse
   Consommation, Analyse Production,
   Prévisions, Performance modèle)
```

Le suivi des exécutions Airflow et de la fraîcheur des données (« Pipelines »)
n'est pas dans l'app Streamlit mais dans Grafana (accès administrateur, voir
plus bas).

Le dashboard Streamlit et l'API FastAPI (`app.py`, déploiement autonome
optionnel) lisent uniquement des artefacts déjà produits — aucune logique
d'entraînement ne s'exécute dans l'interface.

## Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.12.4 (voir `.python-version`) |
| Données | pandas, numpy, PostgreSQL (SQLAlchemy + psycopg2) |
| Modèles | scikit-learn (splits, métriques), LightGBM, Optuna (TPE, 25 essais) |
| Sources de données | API RTE (consommation, production, prévision J-1), Meteostat (température), Open-Meteo (température observée et prévue, 17 sites pondérés) |
| Orchestration | Apache Airflow (LocalExecutor) |
| Suivi d'expériences | MLflow (registre de modèles, alias champion/challenger) |
| Dashboard | Streamlit + Plotly |
| Service d'inférence | FastAPI (déploiement autonome optionnel, `app.py`) |
| Supervision | Grafana (branché sur PostgreSQL) + service `alert-bridge` |
| Conteneurisation | Docker / Docker Compose |
| Qualité | pytest, ruff, GitHub Actions (CI) |

## Chemin des données

`RTE + Meteostat + Open-Meteo → ingestion (upsert PostgreSQL) → validation
(schéma, chronologie, dérive KS) → transformation (features causales : lags
et moyennes glissantes jusqu'à 336h, un jeu de features par cible) →
entraînement (LightGBM, un modèle direct par horizon 1 à 24h, sélection
bayésienne, comparaison à une double baseline de persistance) →
enregistrement (MLflow + artefacts hashés SHA-256) → prévision (intervalles
conformes dynamiques) → évaluation continue (prévu vs réalisé) → affichage
Streamlit → supervision Grafana`.

**Température : une seule source des deux côtés.** Les features thermiques
d'origine et la température prévue à l'heure cible viennent toutes deux
d'Open-Meteo (`temp_fr_om` / `temp_fr_prev`). Le biais entre grille
Open-Meteo et stations Meteostat n'est pas constant — il varie de 0,78 °C
selon l'heure et de 0,63 °C selon le niveau de température — donc un modèle
à arbres ne peut pas l'absorber : mélanger les sources ferait apprendre
l'écart grille/station comme de la physique de consommation. `temp` (une
station) et `temp_fr` (Meteostat pondéré) restent collectées pour pouvoir
rejouer la comparaison.

Le découpage entraînement/validation/test est **strictement chronologique**
(aucun mélange aléatoire), avec un embargo de 24h (l'horizon maximal) à
chaque frontière pour empêcher une cible de chevaucher la partition
suivante. Il n'y a ni imputer ni scaler ajustés globalement : les lignes
avec valeurs manquantes après calcul des features sont retirées
(`dropna`), de façon déterministe et identique entre partitions.

## Installation & lancement

### Prérequis

- Python 3.12
- Docker (recommandé pour l'environnement complet)
- Deux paires de clés API RTE (consommation + production) et un point
  géographique pour Meteostat, si vous relancez l'ingestion

### Configuration

Copier `.env.example` en `.env` et compléter les identifiants RTE/Meteostat
(non versionnés). Les valeurs PostgreSQL par défaut correspondent déjà au
`docker-compose.yml`.

### Stack complète (Docker Compose)

```bash
docker compose up -d --build
```

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| Airflow | http://localhost:8080 (admin / admin — à changer en dehors d'un usage local) |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin / admin — idem) |

Démarrer uniquement la base de données : `docker compose up -d postgres`.
Arrêter : `docker compose down` (les volumes persistent).

### Développement local (sans Docker)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux/macOS

pip install -r requirements-dev.txt   # requirements.txt + pytest + ruff

docker compose up -d postgres     # base de données requise par l'app et les scripts

streamlit run streamlit_app/app_main.py
```

### Pipeline d'entraînement

```bash
python main.py                     # ingestion -> validation -> transformation -> entraînement
python -m scripts.retrain          # entraîne un challenger et le promeut s'il bat le champion
```

### Scripts opérationnels (préfigurent les DAGs Airflow)

```bash
python -m scripts.seed_observations               # peuple `observations` depuis datasets/data.csv, puis complète via RTE
python -m scripts.ingest 2024-01-01 2024-01-07   # ingestion d'une période
python -m scripts.forecast 24                     # prévision J+1 (horizon en heures)
python -m scripts.evaluate                        # évaluation prévu vs réalisé
python -m scripts.backfill_forecasts 30 24         # rejoue des prévisions historiques
```

Rétro-alimentation (une migration additive laisse le passé à NULL : ces
scripts comblent l'historique après ajout d'une colonne) :

```bash
python -m scripts.build_temperature_france         # ajoute `temp_fr` à datasets/data.csv
python -m scripts.backfill_temperature_france      # comble `observations.temp_fr`
python -m scripts.backfill_prevision_temperature   # comble `temp_fr_om` / `temp_fr_prev`
python -m scripts.backfill_rte_forecast            # historique des prévisions RTE J-1 (étalon)
python -m scripts.extend_history                   # étend `observations` vers le passé
```

Analyse et validation — c'est ici que se juge un challenger :

```bash
python -m scripts.compare_models --start ... --end ...          # verdict par cible, IC95 par bootstrap à blocs
python -m scripts.evaluate_weather_regimes --start ... --end ... # backtest stratifié par bascule thermique
python -m scripts.rolling_origin_cv                              # validation à origine glissante
python -m scripts.analyse_rolling_cv                             # agrège les replis par catégorie de jour
python -m scripts.evaluate_features                              # apport de chaque jeu de features
python -m scripts.validate_bias_params                           # kill-test d'une correction de biais en ligne
```

`compare_models` moyenne sur les 24 heures d'origine ; `evaluate_weather_regimes`
filtre les origines réellement utilisées en production (21-23 h UTC). Les deux
peuvent diverger — voir « Limites connues ».

`seed_observations` est le point d'entrée du démarrage à froid : `datasets/data.csv`
est le seul historique versionné (et celui sur lequel le modèle en production a été
entraîné). Il inscrit la provenance dans la colonne `source` (`data.csv:<sha8>`), puis
contrôle continuité, bornes physiques et complétude. `--remplacer` réaligne la plage du
CSV sur son contenu exact — après archivage compressé systématique — quand la table
contient des valeurs aberrantes ou d'origine inconnue ; `--sans-rte` s'en tient au CSV.

### Qualité

```bash
pytest -q                                          # tests unitaires
RUN_STREAMLIT_INTEGRATION=1 pytest -q tests/test_streamlit_integration.py  # rendu des 6 pages (nécessite PostgreSQL peuplé)
ruff check .                                        # lint (règles syntaxiques : E9, F63, F7, F82)
docker compose config -q                            # valide docker-compose.yml
```

## Modèles et validation

> La formalisation mathématique complète de l'architecture — features,
> équation de prévision, intervalles conformes, protocole de validation et
> limites structurelles — est dans
> [`docs/ARCHITECTURE_MODELE.md`](docs/ARCHITECTURE_MODELE.md).

- **Cible** : production par filière suivie (solaire, biomasse, éolien
  terrestre, nucléaire) et consommation totale — cinq cibles, chacune avec
  son propre jeu de modèles. La température est une variable exogène
  (entrée uniquement, jamais prédite).
- **Architecture (`direct_multihorizon_residual`)** : pour chaque cible, 24
  modèles LightGBM indépendants sont entraînés, un par horizon (h = 1 à 24
  heures). Chacun apprend non pas la valeur future mais son **écart à la
  persistance** (`y_{t+h} − y_t`), ce qui retire la composante de niveau non
  stationnaire. La prévision publiée est une **agrégation convexe de deux
  experts** — persistance corrigée `y_t + α·f(x)` et persistance saisonnière
  `y_{t+h−24}` — dont les coefficients `(α, w)` sont estimés par cible et par
  horizon sur la validation. Il n'y a **pas** de réinjection autorégressive :
  le modèle de l'horizon 24 ne consomme pas la sortie de l'horizon 23, donc
  l'erreur ne se propage pas. Les intervalles conformes s'élargissent avec
  l'horizon.
- **Température prévue** : les features `temp_prev_h1..h24` donnent au
  modèle la température attendue à l'heure cible. L'entraînement consomme de
  **vraies prévisions passées** (archive des runs Open-Meteo), jamais
  l'observé à l'heure cible — s'entraîner sur une météo future parfaite
  produirait un modèle qui déçoit dès qu'il reçoit une prévision réelle.
- **Sélection de modèle** : LightGBM, hyperparamètres optimisés par
  recherche bayésienne (Optuna/TPE, 25 essais) via `TimeSeriesSplit` purgé
  sur le développement (train+valid), l'objectif étant un **ratio à la
  persistance** — comparable entre cibles d'échelles très différentes
  (nucléaire ~50 GW vs solaire 0–10 GW). Le nombre d'arbres par horizon est
  fixé par early stopping sur la validation, puis le modèle est réentraîné
  sur `train ∪ valid` avec ce nombre figé.
- **Baseline** : chaque modèle est comparé, par variable et par horizon, à
  une double baseline (persistance de la dernière valeur observée et
  persistance saisonnière) sur le test — le rapport `metadata.json` de
  chaque artefact conserve ce comparatif (gain % par rapport à la
  persistance).
- **Versionnement** : chaque modèle entraîné produit un `metadata.json`
  (hash SHA-256 du modèle et des CSV sources, commit Git, hyperparamètres,
  métriques de test par variable, versions runtime). MLflow conserve
  l'historique complet des runs et un registre de modèles avec les alias
  `challenger`/`champion`.
- **Promotion** : un challenger n'est promu en champion (`scripts/retrain.py`)
  que s'il bat le champion actuel sur un backtest récent — jamais
  automatiquement à la fin de l'entraînement.

## Limites connues

- Périmètre de production restreint à quatre filières (voir encart
  ci-dessus) : les écarts affichés ne sont pas le solde électrique
  national.
- Le lint CI (`ruff`) ne vérifie que des erreurs syntaxiques (E9, F63, F7,
  F82), pas un style complet.
- Les identifiants par défaut de `docker-compose.yml` (Airflow, Grafana,
  PostgreSQL) sont prévus pour un usage local uniquement et doivent être
  changés avant tout partage du déploiement.
- Le schéma de données (`data_schema/schema.yaml`) couvre les dix variables
  suivies (4 filières, production totale, consommation, et 4 colonnes de
  température) ; toute nouvelle source de données nécessite de l'étendre.

Limites du modèle lui-même, mesurées et documentées dans
[`docs/ARCHITECTURE_MODELE.md`](docs/ARCHITECTURE_MODELE.md) §11 :

- **Aucun jour férié ni vacances scolaires** dans le bloc calendaire. Sur la
  consommation nationale, c'est une source d'erreur systématique connue et
  bon marché à corriger.
- **Aucune prévision d'irradiance ni de vent.** La température prévue est
  branchée, mais SOLAR et WIND_ONSHORE sont physiquement pilotés par
  l'irradiance et le vent *futurs* : à h = 24, c'est la contrainte dominante
  qui reste sur ces deux filières.
- **Couverture des intervalles conformes : 92-94 % pour un nominal de 95 %**,
  la garantie conforme supposant une échangeabilité des résidus que les
  séries temporelles ne respectent pas. Les bornes sont donc légèrement
  optimistes.
- **Aucune adaptation en ligne.** `α` et `w` sont gelés entre deux
  réentraînements ; une correction de biais en ligne a existé et a été
  **retirée** après mesure (son backtest lisait le futur pour h ≥ 2, et tout
  son gain apparent venait de là — cf. §6 et `scripts/validate_bias_params.py`).
- **Verdicts divergents selon les origines évaluées.** `docs/model_comparison.json`
  moyenne sur les 24 heures d'origine et conclut « nul » sur la consommation,
  là où `docs/weather_regimes.json` mesure +15,4 % sur les seules origines
  21-23 h UTC utilisées en production. L'écart est à lever avant toute
  promotion du challenger.

## Structure du dépôt

```
pipeline_prevision/       # cœur ML : ingestion, validation, transformation, entraînement, DB, utils
  components/             # étapes du pipeline d'entraînement
  db/                      # accès PostgreSQL (observations, prévisions, métriques, runs)
  entity/                  # configs et artefacts typés
  utils/ml_utils/          # estimator, métriques, inférence locale (local_forecaster)
streamlit_app/            # dashboard (6 pages) + accès aux données
dags/                      # DAGs Airflow (ingestion, prévision, évaluation, réentraînement)
scripts/                   # exécutions unitaires préfigurant les DAGs
docker/                    # Dockerfiles par service (airflow, mlflow, streamlit, alert_bridge)
monitoring/grafana/        # dashboards et provisioning Grafana
data_schema/schema.yaml    # schéma et bornes plausibles des données
docs/                      # formalisation de l'architecture + rapports de mesure (JSON)
tests/                     # tests unitaires et d'intégration
app.py                     # service FastAPI autonome (déploiement optionnel du modèle seul)
main.py                    # exécution du pipeline d'entraînement complet
```

`notebooks/` n'est **pas versionné** : les notebooks portaient leurs sorties
et pesaient 27 Mo de blobs dans l'historique. Ce qui doit survivre à une
exploration est repris en script (`scripts/`) ou en note (`docs/`), qui se
relisent et se diffent.
