"""Évaluation prévu vs réalisé (backtesting continu).

Préfigure le DAG `evaluate_daily` : joint les prévisions passées avec les
observations réellement collectées, calcule les erreurs par variable et par
horizon (MAE, RMSE, MAPE, biais, couverture des intervalles), et les persiste
dans la table `forecast_metrics` (source des panneaux Grafana de qualité modèle).

Usage :
    python -m scripts.evaluate
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sqlalchemy import text

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.db import get_observations, save_metrics, log_run, init_db
from pipeline_prevision.db.config import get_engine

_EPS = 1e-6  # seuil pour éviter la division par ~0 dans le MAPE (ex. SOLAR la nuit)


def run() -> int:
    t0 = time.time()
    try:
        init_db()  # garantit le schéma à jour (dont la colonne mse)
        obs = get_observations()
        if obs is None or obs.empty:
            raise ValueError("Aucune observation en base")

        # Prévisions déjà produites (avec bornes d'intervalle éventuelles).
        with get_engine().connect() as conn:
            result = conn.execute(text(
                "SELECT target_ts, horizon_h, variable, y_pred, y_lower, y_upper, model_version "
                "FROM forecasts"
            ))
            fc = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        if fc.empty:
            raise ValueError("Aucune prévision à évaluer")
        fc["target_ts"] = pd.to_datetime(fc["target_ts"])

        # Réalisé au format long (une ligne par horodatage × variable).
        obs_long = (obs.reset_index()
                    .melt(id_vars="timestamp", var_name="variable", value_name="y_true")
                    .rename(columns={"timestamp": "target_ts"}))
        obs_long["target_ts"] = pd.to_datetime(obs_long["target_ts"])

        # Jointure : ne garde que les prévisions dont la cible est déjà réalisée.
        merged = fc.merge(obs_long, on=["target_ts", "variable"], how="inner")
        if merged.empty:
            log_run("evaluate_daily", "success", rows=0, duration_s=time.time() - t0,
                    message="aucune cible réalisée à évaluer")
            print("Aucun réalisé disponible pour les prévisions -> 0 métrique")
            return 0

        rows = []
        # Keep model versions separated in every evaluation aggregate.
        for (variable, horizon, model_version), g in merged.groupby(
            ["variable", "horizon_h", "model_version"], dropna=False
        ):
            err = g["y_pred"].astype(float) - g["y_true"].astype(float)
            y_true = g["y_true"].astype(float)

            mae = float(err.abs().mean())
            mse = float((err ** 2).mean())
            rmse = float(np.sqrt(mse))
            bias = float(err.mean())

            mask = y_true.abs() > _EPS
            mape = float((err[mask].abs() / y_true[mask].abs()).mean() * 100) if mask.any() else None

            coverage = None
            if g["y_lower"].notna().any() and g["y_upper"].notna().any():
                inside = (y_true >= g["y_lower"].astype(float)) & (y_true <= g["y_upper"].astype(float))
                coverage = float(inside.mean() * 100)

            rows.append({
                "period_start": g["target_ts"].min().to_pydatetime(),
                "period_end": g["target_ts"].max().to_pydatetime(),
                "variable": variable,
                "horizon_h": int(horizon),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "bias": bias,
                "coverage": coverage,
                "n_points": int(len(g)),
                "model_version": None if pd.isna(model_version) else model_version,
            })

        n = save_metrics(rows)
        duration = time.time() - t0
        log_run("evaluate_daily", "success", rows=n, duration_s=duration,
                message=f"{merged.shape[0]} points évalués")
        logging.info("Évaluation OK : %s lignes de métriques (%s points) en %.1fs",
                     n, merged.shape[0], duration)
        print(f"OK : {n} lignes de métriques (variable × horizon), {merged.shape[0]} points évalués")
        return n

    except Exception as e:
        log_run("evaluate_daily", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Évaluation en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    run()
