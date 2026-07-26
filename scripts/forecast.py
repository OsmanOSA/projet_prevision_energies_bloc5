"""Génération et persistance de la prévision multi-horizon (J+1 à J+horizon).

Préfigure le DAG Airflow `forecast_daily` : lit les dernières séries
observées en base, appelle le modèle de prévision **direct** par horizon
(un modèle dédié par heure d'anticipation, pas de rollout autorégressif),
et persiste la prévision dans la table `forecasts`.

Usage :
    python -m scripts.forecast            # horizon 24 h par défaut
    python -m scripts.forecast 12
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
    predict_with_conformal_intervals,
    get_model_version,
    derive_production_total,
    TARGETS,
)


def run(horizon: int = 24) -> int:
    t0 = time.time()

    try:

        df = get_observations()
        if df is None or df.empty:
            raise ValueError("Aucune observation en base : lancez l'ingestion d'abord")

        df = df.sort_index()
        origin_ts = pd.Timestamp(df.index[-1])
        horizons = range(1, horizon + 1)

        # Un jeu de 24 modèles dédiés par cible : chaque horizon est prédit
        # directement depuis l'origine, jamais nourri par une prédiction
        # antérieure (cf. local_forecaster.predict_with_conformal_intervals).
        per_target = {
            target: predict_with_conformal_intervals(df, target, horizons=horizons, alpha=0.05)
            for target in TARGETS
        }

        # production_total n'est plus une cible modélisée : reconstruite en
        # sommant les 4 sources, pour ne rien casser en aval (graphe du
        # déficit, habitudes du dashboard) -- cf. derive_production_total.
        derived = derive_production_total(per_target)
        pred_df = pd.concat({
            **{target: result["y_pred"] for target, result in per_target.items()},
            "production_total": derived["y_pred"],
        }, axis=1)
        lower_df = pd.concat({
            **{target: result["y_lower"] for target, result in per_target.items()},
            "production_total": derived["y_lower"],
        }, axis=1)
        upper_df = pd.concat({
            **{target: result["y_upper"] for target, result in per_target.items()},
            "production_total": derived["y_upper"],
        }, axis=1)

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
