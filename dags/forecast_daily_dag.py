"""DAG `forecast_daily` — prévision énergétique J+1.

Lit les dernières séries observées, appelle le modèle de prévision et persiste
la prévision multi-horizon (par variable) dans la table `forecasts`. La table
sert ensuite au dashboard et au DAG d'évaluation (prévu vs réalisé).

Planifiée à 00h10 heure de Paris (pas UTC) : notre fenêtre glissante de 24h
part alors de 23h la veille et couvre exactement la journée civile suivante,
pour s'aligner sur la prévision RTE J-1 (qui porte sur des jours calendaires
entiers, cf. `rte_forecast_daily` et `scripts/fetch_rte_forecast.py`) — sans
ça, une origine à 06h UTC ne couvrirait qu'une partie de n'importe quel jour
civil RTE, rendant la comparaison peu lisible.

Un sensor précède la prévision : RTE publie la consommation avec ~1h30 de
latence, contre moins de 30 min pour la production et la température. Sans
attente, l'origine de la consommation décrocherait d'une heure de celle des
autres cibles et la journée civile ne serait couverte qu'à 23/24. On patiente
donc jusqu'à 2 h que la ligne de 23h soit complète — puis on prévoit quand
même, avec l'origine réelle disponible (cf. `soft_fail` plus bas).
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

HORIZON_HOURS = 24
PARIS = ZoneInfo("Europe/Paris")

# Heure locale visée comme origine de la prévision (23h la veille).
ORIGIN_HOUR_PARIS = 23
# Colonnes qui doivent toutes être renseignées pour que les features de
# l'origine soient calculables. `production_total` est dérivée (somme des 4
# sources) : la contrôler n'ajouterait rien.
REQUIRED_COLUMNS = ("temp", "solar", "biomass", "wind_onshore", "nuclear",
                    "consommation_totale")

default_args = {
    "owner": "energia",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def _origine_visee(now_utc) -> datetime:
    """Horodatage UTC naïf (convention de la table) du 23h Paris de la veille.

    Calculé depuis `now()` de PostgreSQL et non depuis les dates logiques
    Airflow : le DAG peut démarrer entre 00h10 et 02h10 selon l'attente du
    sensor, et on veut toujours désigner la même heure d'origine.
    """
    now_paris = now_utc.astimezone(PARIS)
    veille = (now_paris - timedelta(days=1)).date()
    origine_locale = datetime.combine(veille, time(ORIGIN_HOUR_PARIS), tzinfo=PARIS)
    return origine_locale.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def _donnees_completes(**_) -> bool:
    """Vrai quand la ligne d'origine porte toutes ses variables.

    On teste la complétude, pas l'existence : la ligne apparaît dès que RTE
    publie la production, avec `consommation_totale` encore à NULL — c'est
    exactement le cas qui décalait les origines.
    """
    from sqlalchemy import text
    from pipeline_prevision.db import get_engine

    with get_engine().connect() as conn:
        now_utc = conn.execute(text("SELECT now()")).scalar()
        origine = _origine_visee(now_utc)
        manquantes = conn.execute(
            text(
                "SELECT " + ", ".join(f"({c} IS NULL) AS {c}" for c in REQUIRED_COLUMNS)
                + " FROM observations WHERE ts = :ts"
            ),
            {"ts": origine},
        ).mappings().one_or_none()

    if manquantes is None:
        print(f"Origine {origine} absente de `observations` — on patiente.")
        return False

    absentes = [c for c, est_nul in manquantes.items() if est_nul]
    if absentes:
        print(f"Origine {origine} incomplète : {', '.join(absentes)} — on patiente.")
        return False

    print(f"Origine {origine} complète sur {len(REQUIRED_COLUMNS)} variables.")
    return True


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
    attente = PythonSensor(
        task_id="attendre_ligne_23h_complete",
        python_callable=_donnees_completes,
        poke_interval=timedelta(minutes=10).total_seconds(),
        timeout=timedelta(hours=2).total_seconds(),
        # `reschedule` libère le worker entre deux vérifications : on attend
        # jusqu'à 2 h, autant ne pas immobiliser un slot pour ça.
        mode="reschedule",
        # Au bout des 2 h on ne renonce pas à la journée : le sensor passe en
        # SKIPPED et la prévision part quand même, en comblant l'heure
        # manquante par report de la valeur précédente (cf. REPORT_MAX_HOURS
        # dans scripts/forecast.py -- stratégie retenue sur mesure : elle bat
        # de 28,5 MW de MAE le fait de reculer l'origine d'une heure, p=0,035
        # sur 53 jours hors échantillon). Le filet reste MAX_ORIGIN_LAG, qui
        # refuse d'écrire au-delà de 3 h de retard.
        soft_fail=True,
    )

    prevision = PythonOperator(
        task_id="prevision_j_plus_1",
        python_callable=_forecast,
        # `all_done` : la prévision tourne que le sensor ait réussi ou expiré.
        trigger_rule="all_done",
    )

    attente >> prevision
