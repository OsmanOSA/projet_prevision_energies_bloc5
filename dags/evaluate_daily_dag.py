"""DAG `evaluate_daily` — évaluation prévu vs réalisé.

Joint les prévisions passées au réalisé fraîchement collecté et calcule les
erreurs par variable/horizon dans la table `forecast_metrics`. C'est le moteur
du suivi de qualité du modèle (backtesting continu) affiché dans Grafana.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator #type: ignore

default_args = {
    "owner": "energia",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def _evaluate(**_):
    from scripts.evaluate import run
    run()


with DAG(
    dag_id="evaluate_daily",
    description="Prévu vs réalisé -> forecast_metrics (backtesting continu)",
    default_args=default_args,
    schedule="30 6 * * *",  # juste après ingestion + prévision du matin
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["energia", "evaluation"],
) as dag:
    PythonOperator(
        task_id="evaluation_prevu_vs_realise",
        python_callable=_evaluate,
    )
