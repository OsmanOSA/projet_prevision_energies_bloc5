"""DAG `ingest_hourly` — collecte et persistance des séries réalisées.

Rejoue une fenêtre glissante des derniers jours (upsert idempotent) pour
rattraper les données arrivées tardivement côté RTE/Meteostat. Alimente la
table `observations` que consomme le dashboard.
"""

from datetime import datetime, timedelta, date

from airflow import DAG
from airflow.operators.python import PythonOperator #type: ignore

# Nombre de jours ré-ingérés à chaque exécution (rattrapage des retards).
ROLLING_DAYS = 3

default_args = {
    "owner": "energia",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def _ingest(**_):
    # Import différé : le code projet n'est résolu qu'à l'exécution de la tâche.
    from scripts.ingest import run
    from sqlalchemy import text
    from pipeline_prevision.db import get_engine

    # Fin de fenêtre : now() de PostgreSQL (source de temps unique et fiable),
    # et non date.today() du conteneur, dont l'horloge peut dériver (WSL2).
    with get_engine().connect() as conn:
        now_db = conn.execute(text("SELECT now()")).scalar()
    end = now_db.date() + timedelta(days=1)   # +1 j : marge pour le fuseau/latence
    start = end - timedelta(days=ROLLING_DAYS)
    run(start.isoformat(), end.isoformat())


with DAG(
    dag_id="ingest_hourly",
    description="Collecte RTE + Meteostat -> upsert PostgreSQL (fenêtre glissante)",
    default_args=default_args,
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["energia", "ingestion"],
) as dag:
    PythonOperator(
        task_id="collecte_et_persistance",
        python_callable=_ingest,
    )
