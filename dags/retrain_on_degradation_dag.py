"""DAG `retrain_on_degradation` — réentraînement déclenché par la dégradation.

Ce DAG n'est **pas planifié** (`schedule=None`) : il est déclenché de l'extérieur
par l'alerte Grafana (seuil de MAE journalière) via le pont d'alerte, qui appelle
l'API REST d'Airflow.

Sécurité : le signal est **revalidé côté données** avant de lancer un
entraînement coûteux — une alerte parasite ne doit pas déclencher un retrain.
On peut forcer l'exécution avec la configuration `{"force": true}`.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Variable métier surveillée et seuil (surchargeable par l'environnement).
TARGET_VARIABLE = os.getenv("RETRAIN_TARGET_VARIABLE", "consommation_totale")
MAE_THRESHOLD = float(os.getenv("RETRAIN_MAE_THRESHOLD", "1500"))

# Marge d'amélioration relative exigée pour promouvoir le challenger.
PROMOTION_MARGIN = 0.0

default_args = {
    "owner": "energia",
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}


def _degradation_confirmee(**context):
    """Revalide la dégradation depuis `forecast_metrics` (garde-fou)."""
    from sqlalchemy import text
    from pipeline_prevision.db import get_engine

    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run is not None else {}
    if conf.get("force"):
        print("Exécution forcée (conf.force=true) : contrôle de seuil ignoré.")
        return True

    with get_engine().connect() as conn:
        row = conn.execute(text(
            "SELECT avg(mae) AS mae, avg(mse) AS mse FROM forecast_metrics "
            "WHERE variable = :var AND eval_ts > now() - interval '1 day'"
        ), {"var": TARGET_VARIABLE}).fetchone()

    mae = row[0] if row else None
    if mae is None:
        print("Aucune évaluation sur les dernières 24 h : pas de retrain.")
        return False

    degraded = mae > MAE_THRESHOLD
    print(f"MAE {TARGET_VARIABLE} sur 24 h = {mae:.1f} (seuil {MAE_THRESHOLD}) "
          f"-> {'DÉGRADATION confirmée' if degraded else 'performance nominale'}")
    return degraded


def _reentrainement(**_):
    """Entraîne un challenger et le promeut s'il bat le champion."""
    # Tag repris par MLflow pour tracer l'origine de l'entraînement.
    os.environ["RETRAIN_TRIGGER"] = "degradation"
    from scripts.retrain import run
    run(margin=PROMOTION_MARGIN)


with DAG(
    dag_id="retrain_on_degradation",
    description="Réentraînement déclenché par dégradation de performance (alerte Grafana)",
    default_args=default_args,
    schedule=None,                 # déclenché, jamais planifié
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["energia", "retrain", "mlops"],
) as dag:

    verifier = ShortCircuitOperator(
        task_id="verifier_degradation",
        python_callable=_degradation_confirmee,
    )

    reentrainer = PythonOperator(
        task_id="reentrainement_champion_challenger",
        python_callable=_reentrainement,
    )

    verifier >> reentrainer
