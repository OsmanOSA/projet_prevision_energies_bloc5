"""Génération et persistance de la prévision J+1.

Préfigure le futur DAG Airflow `forecast_daily` : lit les dernières séries
observées en base, appelle le modèle de prévision (API distante), et persiste
la prévision multi-horizon dans la table `forecasts`.

Usage :
    python -m scripts.forecast            # horizon 24 h par défaut
    python -m scripts.forecast 48
"""

import os
import sys
import time

import pandas as pd

# Garantir que la racine du projet est sur le path, quel que soit le mode de
# lancement (module, script direct, ou tâche Airflow).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.db import get_observations, save_forecasts, log_run
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
    dynamic_conformal_intervals,
    get_model_version,
)

FEATURES = ["temp", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale"]


def run(horizon: int = 24) -> int:
    t0 = time.time()

    try:

        df = get_observations()
        if df is None or df.empty:
            raise ValueError("Aucune observation en base : lancez l'ingestion d'abord")

        df = df.sort_index()
        origin_ts = pd.Timestamp(df.index[-1])

        # Inférence locale + intervalles conformes dynamiques (~95 %)
        y_pred, y_lower, y_upper, _y_test, _mae, _mse = dynamic_conformal_intervals(
            df, horizon, alpha=0.05)
        if y_pred is None or len(y_pred) == 0:
            raise ValueError("Le modèle n'a renvoyé aucune prévision")

        future_ts = pd.date_range(
            start=origin_ts + pd.Timedelta(1, unit="h"),
            periods=len(y_pred),
            freq="h",
        )
        pred_df = pd.DataFrame(y_pred, columns=FEATURES, index=future_ts)
        lower_df = pd.DataFrame(y_lower, columns=FEATURES, index=future_ts)
        upper_df = pd.DataFrame(y_upper, columns=FEATURES, index=future_ts)

        n = save_forecasts(pred_df, origin_ts=origin_ts, model_version=get_model_version(),
                           lower_df=lower_df, upper_df=upper_df)
        duration = time.time() - t0
        log_run("forecast_daily", "success", rows=n, duration_s=duration,
                message=f"origine {origin_ts} · horizon {horizon}h")
        logging.info("Prévision OK : %s points (origine %s, horizon %sh) en %.1fs",
                     n, origin_ts, horizon, duration)
        print(f"OK : {n} points de prévision persistés (origine {origin_ts}, horizon {horizon}h)")

        return n

    except Exception as e:
        log_run("forecast_daily", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Prévision en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    run(h)
