"""Backfill de prévisions historiques (origines glissantes).

Rejoue le modèle comme si `forecast_daily` avait tourné chaque jour : pour
chaque origine quotidienne des N derniers jours, produit une prévision à
`horizon` heures avec ses intervalles conformes et la persiste.

Objectif : disposer d'un historique de prévisions comparable au réalisé, pour
le backtesting visuel et pour que `evaluate_daily` produise des métriques
statistiquement solides (au lieu d'un point isolé).

Écriture idempotente : rejouer le script ne crée pas de doublons.

Usage :
    python -m scripts.backfill_forecasts            # 30 jours, horizon 24 h
    python -m scripts.backfill_forecasts 7 24
"""

import os
import sys
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.constant.training_pipeline import LOOKBACK
from pipeline_prevision.db import get_observations, save_forecasts, log_run, init_db
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
    dynamic_conformal_intervals,
    get_model_version,
)

FEATURES = ["temp", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale"]
ORIGIN_HOUR = 23          # une prévision par jour, émise à 23 h (couvre le lendemain)
CALIB_WINDOWS = 20        # calibration conforme allégée (le backfill est répétitif)


def run(days: int = 30, horizon: int = 24) -> int:
    t0 = time.time()
    try:
        init_db()
        obs = get_observations()
        if obs is None or obs.empty:
            raise ValueError("Aucune observation en base")
        obs = obs.sort_index()

        last_ts = obs.index.max()
        start_ts = last_ts - pd.Timedelta(days=days)

        # Origines quotidiennes présentes dans les observations.
        candidates = [ts for ts in obs.index
                      if ts >= start_ts and ts.hour == ORIGIN_HOUR]

        persisted = 0
        produced = 0
        for origin_ts in candidates:
            history = obs.loc[:origin_ts]
            # Il faut au moins la fenêtre de lookback + de quoi calibrer.
            if len(history) < LOOKBACK + horizon + 5:
                continue

            y_pred, y_lower, y_upper, _, _, _ = dynamic_conformal_intervals(
                history, horizon, alpha=0.05, n_calib=CALIB_WINDOWS)

            future_ts = pd.date_range(start=origin_ts + pd.Timedelta(hours=1),
                                      periods=len(y_pred), freq="h")
            pred_df = pd.DataFrame(y_pred, columns=FEATURES, index=future_ts)
            lower_df = pd.DataFrame(y_lower, columns=FEATURES, index=future_ts)
            upper_df = pd.DataFrame(y_upper, columns=FEATURES, index=future_ts)

            persisted += save_forecasts(pred_df, origin_ts=origin_ts,
                                        model_version=get_model_version(),
                                        lower_df=lower_df, upper_df=upper_df)
            produced += 1
            print(f"  origine {origin_ts} -> {len(y_pred)} pas persistés")

        duration = time.time() - t0
        log_run("backfill_forecasts", "success", rows=persisted, duration_s=duration,
                message=f"{produced} origines sur {days} j (horizon {horizon} h)")
        logging.info("Backfill prévisions : %s origines, %s lignes en %.1fs",
                     produced, persisted, duration)
        print(f"OK : {produced} origines, {persisted} lignes de prévision en {duration:.1f}s")
        return persisted

    except Exception as e:
        log_run("backfill_forecasts", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Backfill des prévisions en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    run(d, h)
