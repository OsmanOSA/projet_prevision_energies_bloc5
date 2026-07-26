![Python](https://img.shields.io/badge/python-3.12-3776AB)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)
![Airflow](https://img.shields.io/badge/orchestration-Airflow-017CEE)
![PostgreSQL](https://img.shields.io/badge/data-PostgreSQL-4169E1)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)
![Grafana](https://img.shields.io/badge/monitoring-Grafana-F46800)
![Docker](https://img.shields.io/badge/built%20with-Docker-2496ED)

# Prévision et pilotage du système électrique français

Application MLOps de prévision et d'analyse de la consommation et de la
production électrique en France (RTE) enrichie de données météo
(Meteostat). Le projet couvre l'ensemble de la chaîne : ingestion horaire,
validation et détection de dérive, entraînement avec sélection
d'hyperparamètres, prévision multi-horizon avec intervalles conformes,
évaluation continue prévu/réalisé, promotion champion/challenger, et
supervision (Airflow + Grafana), le tout restitué dans un dashboard
Streamlit.

> **Périmètre couvert** : quatre filières de production (solaire, biomasse,
> éolien terrestre, nucléaire) + consommation totale + température. Ni
> l'hydraulique, ni le thermique fossile, ni les échanges transfrontaliers,
> ni les pertes réseau ne sont disponibles : les écarts
> production/consommation affichés dans l'application sont donc **partiels**
> et ne constituent pas le solde électrique national.

## Architecture

```
RTE (API conso + prod) ─┐
Meteostat (température) ┴──► ingestion (scripts/ingest.py, DAG ingest_hourly @hourly)
                              │  upsert idempotent
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
  Streamlit (streamlit_app/, 7 pages :
   Accueil, Vue d'ensemble, Analyse
   Consommation, Analyse Production,
   Prévisions, Performance modèle, Pipelines)
```

Le dashboard Streamlit et l'API FastAPI (`app.py`, déploiement autonome
optionnel) lisent uniquement des artefacts déjà produits — aucune logique
d'entraînement ne s'exécute dans l'interface.

## Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.12.4 (voir `.python-version`) |
| Données | pandas, numpy, PostgreSQL (SQLAlchemy + psycopg2) |
| Modèles | scikit-learn, XGBoost, LightGBM, statsmodels, hyperopt (TPE) |
| Sources de données | API RTE (consommation, production), Meteostat (température) |
| Orchestration | Apache Airflow (LocalExecutor) |
| Suivi d'expériences | MLflow (registre de modèles, alias champion/challenger) |
| Dashboard | Streamlit + Plotly |
| Service d'inférence | FastAPI (déploiement autonome optionnel, `app.py`) |
| Supervision | Grafana (branché sur PostgreSQL) + service `alert-bridge` |
| Conteneurisation | Docker / Docker Compose |
| Qualité | pytest, ruff, GitHub Actions (CI) |

## Chemin des données

`RTE + Meteostat → ingestion (upsert PostgreSQL) → validation (schéma,
chronologie, dérive KS) → transformation (imputation médiane + MinMaxScaler,
fenêtres glissantes lookback=36h/horizon=1h) → entraînement (XGBoost /
LightGBM, sélection bayésienne, comparaison à une baseline de persistance) →
enregistrement (MLflow + artefacts hashés SHA-256) → prévision (intervalles
conformes dynamiques) → évaluation continue (prévu vs réalisé) → affichage
Streamlit → supervision Grafana`.

Le découpage entraînement/validation/test est **strictement chronologique**
(aucun mélange aléatoire) ; l'imputer et le scaler sont ajustés uniquement
sur le train ; le contexte de fenêtrage ajouté à la validation/au test ne
provient que du passé du split précédent.

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
python -m scripts.ingest 2024-01-01 2024-01-07   # ingestion d'une période
python -m scripts.forecast 24                     # prévision J+1 (horizon en heures)
python -m scripts.evaluate                        # évaluation prévu vs réalisé
python -m scripts.backfill_forecasts 30 24         # rejoue des prévisions historiques
```

### Qualité

```bash
pytest -q                                          # tests unitaires
RUN_STREAMLIT_INTEGRATION=1 pytest -q tests/test_streamlit_integration.py  # rendu des 7 pages (nécessite PostgreSQL peuplé)
ruff check .                                        # lint (règles syntaxiques : E9, F63, F7, F82)
docker compose config -q                            # valide docker-compose.yml
```

## Modèles et validation

- **Cible** : température, production par filière suivie (solaire, biomasse,
  éolien terrestre, nucléaire) et consommation totale — prévision
  multi-sortie (un seul modèle prédit les six variables).
- **Régime natif** : horizon 1 heure (`HORIZON=1`, `LOOKBACK=36`). Les
  horizons plus longs sont obtenus par autorégression (le modèle réinjecte
  ses propres prédictions), avec intervalles de confiance conformes qui
  s'élargissent avec l'horizon.
- **Sélection de modèle** : XGBoost et LightGBM (via `MultiOutputRegressor`),
  hyperparamètres optimisés par recherche bayésienne (hyperopt/TPE) sur la
  validation ; le modèle retenu est celui de MAE de validation la plus
  basse.
- **Baseline** : chaque modèle est comparé, par variable, à une baseline de
  persistance (dernière valeur observée) sur le test — le rapport
  `metadata.json` de chaque artefact conserve ce comparatif
  (`beats_persistence`).
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
- Le schéma de données (`data_schema/schema.yaml`) couvre les six variables
  du modèle ; toute nouvelle source de données nécessite de l'étendre.

## Structure du dépôt

```
pipeline_prevision/       # cœur ML : ingestion, validation, transformation, entraînement, DB, utils
  components/             # étapes du pipeline d'entraînement
  db/                      # accès PostgreSQL (observations, prévisions, métriques, runs)
  entity/                  # configs et artefacts typés
  utils/ml_utils/          # estimator, métriques, inférence locale (local_forecaster)
streamlit_app/            # dashboard (7 pages) + accès aux données
dags/                      # DAGs Airflow (ingestion, prévision, évaluation, réentraînement)
scripts/                   # exécutions unitaires préfigurant les DAGs
docker/                    # Dockerfiles par service (airflow, mlflow, streamlit, alert_bridge)
monitoring/grafana/        # dashboards et provisioning Grafana
data_schema/schema.yaml    # schéma et bornes plausibles des données
tests/                     # tests unitaires et d'intégration
app.py                     # service FastAPI autonome (déploiement optionnel du modèle seul)
main.py                    # exécution du pipeline d'entraînement complet
```
