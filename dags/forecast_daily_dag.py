"""DAG `forecast_daily` — prévision énergétique J+1.

Lit les dernières séries observées, appelle le modèle de prévision et persiste
la prévision multi-horizon (par variable) dans la table `forecasts`. La table
sert ensuite au dashboard et au DAG d'évaluation (prévu vs réalisé).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

HORIZON_HOURS = 24

default_args = {
    "owner": "energia",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def _forecast(**_):
    from scripts.forecast import run
    run(HORIZON_HOURS)


with DAG(
    dag_id="forecast_daily",
    description="Prévision J+1 multi-source -> table forecasts",
    default_args=default_args,
    schedule="0 6 * * *",  # tous les jours à 06:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["energia", "prevision"],
) as dag:
    PythonOperator(
        task_id="prevision_j_plus_1",
        python_callable=_forecast,
    )
