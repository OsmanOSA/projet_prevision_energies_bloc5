"""Prévision officielle RTE (J-1) pour la consommation — référence externe.

Persiste, pour un jour donné, la prévision de consommation publiée par RTE la
veille au soir (API Consumption, type=D-1) dans la table `forecasts`, sous
une variable dédiée (`consommation_totale_rte`) distincte de nos propres
prévisions. Sert uniquement de repère de crédibilité sur la page Prévisions
(comparaison à notre propre modèle) — jamais utilisée en entraînement ni en
évaluation de notre modèle.

Usage :
    python -m scripts.fetch_rte_forecast             # jour courant
    python -m scripts.fetch_rte_forecast 2026-07-25   # jour donné
"""

import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.db import save_forecasts, log_run
from pipeline_prevision.utils.main_utils.utils import extract_conso_forecast_rte

RTE_MODEL_VERSION = "RTE (J-1)"
RTE_VARIABLE = "consommation_totale_rte"


def run(target_date: str | None = None) -> int:
    t0 = time.time()
    # Jour civil Paris, pas la date du conteneur (UTC) : le DAG tourne à 00h10
    # heure de Paris = 22h10 UTC, encore la veille en UTC -> `date.today()` y
    # renverrait systématiquement le jour d'avant pendant cette fenêtre.
    target_date = target_date or datetime.now(ZoneInfo("Europe/Paris")).date().isoformat()
    try:
        df = extract_conso_forecast_rte(target_date)
        # "Origine" conventionnelle = la veille du jour couvert (moment où
        # RTE publie sa prévision D-1) ; sert seulement à calculer horizon_h,
        # la page Prévisions filtre par target_ts, pas par cette origine.
        origin_ts = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)

        pred_df = df.rename(columns={"y_pred": RTE_VARIABLE})
        n = save_forecasts(pred_df, origin_ts=origin_ts, model_version=RTE_MODEL_VERSION)

        duration = time.time() - t0
        log_run("fetch_rte_forecast", "success", rows=n, duration_s=duration,
                message=f"prévision RTE J-1 pour {target_date}")
        logging.info("Prévision RTE OK : %s points (%s) en %.1fs", n, target_date, duration)
        print(f"OK : {n} points de prévision RTE persistés pour {target_date}")
        return n

    except Exception as e:
        log_run("fetch_rte_forecast", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Récupération de la prévision RTE en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else None
    run(d)
