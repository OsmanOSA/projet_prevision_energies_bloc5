"""DAG `forecast_daily` — prévision énergétique J+1.

Lit les dernières séries observées, appelle le modèle de prévision et persiste
la prévision multi-horizon (par variable) dans la table `forecasts`. La table
sert ensuite au dashboard et au DAG d'évaluation (prévu vs réalisé).

Planifiée à 23h heure de Paris (pas UTC) : notre fenêtre glissante de 24h part
alors quasiment de minuit et couvre exactement la journée civile suivante,
pour s'aligner sur la prévision RTE J-1 (qui porte sur des jours calendaires
entiers, cf. `rte_forecast_daily` et `scripts/fetch_rte_forecast.py`) — sans
ça, une origine à 06h UTC ne couvrirait qu'une partie de n'importe quel jour
civil RTE, rendant la comparaison peu lisible.
"""

from datetime import timedelta

import pendulum
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
    schedule="10 0 * * *",  # tous les jours à 00h10, heure de Paris
    start_date=pendulum.datetime(2024, 1, 1, tz="Europe/Paris"),
    catchup=False,
    max_active_runs=1,
    tags=["energia", "prevision"],
) as dag:
    PythonOperator(
        task_id="prevision_j_plus_1",
        python_callable=_forecast,
    )
