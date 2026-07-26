"""DAG `rte_forecast_daily` — prévision RTE J-1 (repère de crédibilité).

Récupère la prévision officielle de consommation publiée par RTE la veille
au soir (API Consumption, type=D-1) pour le jour qui vient de commencer, et la
persiste dans `forecasts` sous `variable='consommation_totale_rte'`. Sert
uniquement de comparaison externe sur le dashboard — indépendante de notre
propre pipeline de prévision (cf. `forecast_daily`).

Planifiée à 00h10 heure de Paris : RTE publie sa prévision D-1 vers 23h58 la
veille (observé empiriquement) ; ce petit délai évite d'interroger l'API
avant que la donnée du jour ne soit disponible.
"""

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "energia",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def _fetch_rte_forecast(**_):
    from scripts.fetch_rte_forecast import run
    run()


with DAG(
    dag_id="rte_forecast_daily",
    description="Prévision RTE J-1 (consommation) -> table forecasts, repère de crédibilité",
    default_args=default_args,
    schedule="10 0 * * *",  # tous les jours à 00h10, heure de Paris
    start_date=pendulum.datetime(2024, 1, 1, tz="Europe/Paris"),
    catchup=False,
    max_active_runs=1,
    tags=["energia", "prevision", "rte"],
) as dag:
    PythonOperator(
        task_id="prevision_rte_j_moins_1",
        python_callable=_fetch_rte_forecast,
    )
