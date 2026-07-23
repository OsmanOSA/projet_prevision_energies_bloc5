"""Accès aux données pour l'app Streamlit (lecture PostgreSQL)."""

import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

# Racine du projet importable (réutilise la couche db du pipeline).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline_prevision.db.config import get_engine
from pipeline_prevision.db import get_observations as _get_observations

# Les horodatages sont stockés en UTC (bonne pratique) ; l'exploitation d'un
# réseau français se lit en heure locale -> conversion à l'affichage seulement.
DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Europe/Paris")

FEATURES = ["temp", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale"]


def _to_local(values):
    """UTC (stockage) -> fuseau d'affichage, en horodatages naïfs.

    On repasse en naïf pour que les jointures et comparaisons en aval restent
    homogènes (mélanger tz-aware et naïf lèverait une erreur pandas).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(DISPLAY_TZ).tz_localize(None)


def _observations_local() -> pd.DataFrame:
    """Observations avec index converti dans le fuseau d'affichage."""
    obs = _get_observations()
    if obs is None or obs.empty:
        return obs
    obs = obs.copy()
    obs.index = _to_local(obs.index)
    obs.index.name = "timestamp"
    # Le passage heure d'été -> heure d'hiver répète l'heure locale 02h-03h une
    # fois par an : deux instants UTC distincts partagent alors le même
    # horodatage naïf. On ne garde que la première occurrence pour préserver
    # l'unicité de l'index (indispensable à .reindex/.loc en aval) ; l'écart
    # perdu est d'une heure sur ~8760/an, sans impact visible sur les courbes.
    obs = obs[~obs.index.duplicated(keep="first")]
    return obs
PRODUCTION_SOURCES = ["SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR"]
ENERGY_COLORS = {"SOLAR": "#f1c40f", "BIOMASS": "#27ae60", "WIND_ONSHORE": "#3498db", "NUCLEAR": "#e74c3c"}


def _query(sql: str) -> pd.DataFrame:
    """Exécute une requête et renvoie un DataFrame (robuste SQLAlchemy 1.4/2.0)."""
    with get_engine().connect() as conn:
        result = conn.execute(text(sql))
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def load_observations(start=None, end=None) -> pd.DataFrame:
    """Séries réalisées (index timestamp en heure locale, colonnes FEATURES)."""
    obs = _observations_local()
    if obs is None or obs.empty:
        return obs
    if start is not None:
        obs = obs[obs.index >= pd.Timestamp(start)]
    if end is not None:
        obs = obs[obs.index <= pd.Timestamp(end)]
    return obs


def load_kpis() -> dict:
    """Indicateurs de tête (une valeur chacun)."""
    row = _query(
        """
        SELECT
          EXTRACT(EPOCH FROM (now() - max(ts))) / 3600 AS freshness_h,
          count(*)                                     AS n_obs
        FROM observations
        """
    ).iloc[0]
    n_forecasts = _query("SELECT count(*) AS n FROM forecasts").iloc[0]["n"]
    runs_ok = _query(
        "SELECT count(*) AS n FROM pipeline_runs "
        "WHERE status='success' AND run_ts > now() - interval '24 hours'"
    ).iloc[0]["n"]
    active = _query(
        """
        SELECT model_version, origin_ts, run_ts
        FROM forecasts
        ORDER BY origin_ts DESC, run_ts DESC
        LIMIT 1
        """
    )
    return {
        "freshness_h": float(row["freshness_h"]) if row["freshness_h"] is not None else None,
        "n_obs": int(row["n_obs"]),
        "n_forecasts": int(n_forecasts),
        "runs_ok_24h": int(runs_ok),
        "model_version": (
            str(active.iloc[0]["model_version"]) if not active.empty else None
        ),
    }


def load_latest_forecast() -> pd.DataFrame:
    """Prévision **en cours de vérification**, avec ses bornes conformes.

    On retient la prévision la plus récente dont au moins une cible est déjà
    réalisée : c'est celle que l'on peut confronter au réel au fil de l'eau,
    l'ingestion horaire complétant la courbe « réalisé » à mesure que RTE
    publie (latence ~2 h).

    Une prévision toute fraîche (dont aucune cible n'est encore réalisée)
    n'apparaîtrait qu'en pointillés : elle sert de repli si aucune autre.

    NB : on sélectionne par `origin_ts` et non par `run_ts` — un backfill
    exécuté aujourd'hui porte des origines historiques.
    """
    fc = _query(
        """
        SELECT run_ts, origin_ts, target_ts, horizon_h, variable,
               y_pred, y_lower, y_upper, model_version
        FROM forecasts
        WHERE origin_ts = (SELECT max(origin_ts) FROM forecasts)
        ORDER BY target_ts
        """
    )
    if not fc.empty:
        for column in ("run_ts", "origin_ts", "target_ts"):
            fc[column] = _to_local(fc[column])
    return fc


def load_forecast_vs_actual(variable: str, horizon=None, days: int = 30) -> pd.DataFrame:
    """Prévisions passées alignées sur le réalisé (backtesting).

    horizon=None : toutes les échéances. Comme les origines sont quotidiennes et
    l'horizon de 24 h, les échéances H+1..H+24 pavent le temps en continu — on
    obtient une courbe prévue horaire sans recouvrement.
    horizon=<int> : ne garde qu'une échéance (ex. H+24) pour juger la précision
    à un délai d'anticipation donné.
    """
    if variable not in FEATURES:
        raise ValueError(f"Variable inconnue : {variable}")

    horizon_filter = f"AND horizon_h = {int(horizon)}" if horizon is not None else ""
    fc = _query(
        f"""
        SELECT target_ts, origin_ts, horizon_h, y_pred, y_lower, y_upper
        FROM forecasts
        WHERE variable = '{variable}'
          {horizon_filter}
          AND target_ts >= (SELECT max(ts) FROM observations) - INTERVAL '{int(days)} days'
        ORDER BY target_ts
        """
    )
    if fc.empty:
        return fc

    fc["target_ts"] = _to_local(fc["target_ts"])
    if "origin_ts" in fc.columns:
        fc["origin_ts"] = _to_local(fc["origin_ts"])

    obs = _observations_local()
    if obs is None or obs.empty or variable not in obs.columns:
        fc["y_true"] = pd.NA
        return fc

    actual = obs[variable].rename("y_true")
    fc = fc.merge(actual, left_on="target_ts", right_index=True, how="left")
    return fc


def _recent_window(days: int):
    """Fenêtre d'observations contiguë des `days` derniers jours (heure locale)."""
    obs = _observations_local()
    if obs is None or obs.empty:
        return None
    return obs[obs.index >= obs.index.max() - pd.Timedelta(days=days)].sort_index()


def _conformal_bounds(
    y_pred,
    y_true,
    alpha=0.05,
    step_index=None,
    min_calibration=10,
):
    """Bornes préquentielles : chaque point utilise seulement les résidus passés.

    Les premiers points restent sans borne tant que l'historique de calibration
    est insuffisant. La couverture affichée porte donc sur un vrai hors-calibration.
    """
    pred = np.asarray(y_pred, dtype=float)
    truth = np.asarray(y_true, dtype=float)
    residuals = np.abs(truth - pred)
    steps = (
        np.zeros(len(pred), dtype=int)
        if step_index is None
        else np.asarray(step_index)
    )
    lower = np.full(len(pred), np.nan)
    upper = np.full(len(pred), np.nan)

    history: dict[int, list[float]] = {}
    for index, step in enumerate(steps):
        key = int(step)
        past = history.setdefault(key, [])
        if len(past) >= min_calibration:
            quantile = float(np.quantile(past, 1 - alpha, method="higher"))
            lower[index] = pred[index] - quantile
            upper[index] = pred[index] + quantile
        if np.isfinite(residuals[index]):
            past.append(float(residuals[index]))
    return lower, upper


def load_singlestep_backtest(variable: str, days: int) -> pd.DataFrame:
    """Backtest H+1 **glissant** : le modèle dans son régime natif.

    À chaque heure, la valeur suivante est prédite à partir des observations
    réelles (aucune autorégression) — courbe continue, sans accumulation
    d'erreur. Calculé à la volée, rien n'est persisté.
    """
    if variable not in FEATURES:
        raise ValueError(f"Variable inconnue : {variable}")

    # Import tardif : la stack ML n'est pas requise pour le reste de l'app.
    from pipeline_prevision.utils.ml_utils.model.local_forecaster import predict_singlestep

    window = _recent_window(days)
    if window is None:
        return pd.DataFrame()

    targets, y_pred, y_true, _, _ = predict_singlestep(window)
    col = FEATURES.index(variable)
    pred, reel = y_pred[:, col], y_true[:, col]
    bas, haut = _conformal_bounds(pred, reel)

    return pd.DataFrame({
        "target_ts": pd.DatetimeIndex(targets),
        "y_pred": pred, "y_true": reel, "y_lower": bas, "y_upper": haut,
    })


def load_rolling_backtest(variable: str, horizon: int, days: int):
    """Backtest autorégressif glissant : une origine tous les `horizon` pas.

    Reproduit le mode d'exploitation « le modèle tourne toutes les `horizon`
    heures et prévoit `horizon` heures » : les blocs s'enchaînent en une courbe
    continue, et les origines marquent les remises à zéro de l'erreur.

    Retourne (DataFrame, origines) pour tracer les coutures.
    """
    if variable not in FEATURES:
        raise ValueError(f"Variable inconnue : {variable}")

    from pipeline_prevision.utils.ml_utils.model.local_forecaster import rolling_multistep

    window = _recent_window(days)
    if window is None:
        return pd.DataFrame(), []

    targets, y_pred, y_true, origins = rolling_multistep(window, horizon)
    col = FEATURES.index(variable)
    pred, reel = y_pred[:, col], y_true[:, col]

    # Position dans le bloc (1..horizon) : calibre l'intervalle par pas.
    pas = np.tile(np.arange(1, horizon + 1), len(origins))[:len(pred)]
    bas, haut = _conformal_bounds(pred, reel, step_index=pas)

    df = pd.DataFrame({
        "target_ts": pd.DatetimeIndex(targets),
        "y_pred": pred, "y_true": reel, "y_lower": bas, "y_upper": haut,
    })
    return df, list(origins)


def available_horizons() -> list:
    """Horizons disponibles dans la table des prévisions."""
    df = _query("SELECT DISTINCT horizon_h FROM forecasts ORDER BY horizon_h")
    return df["horizon_h"].tolist() if not df.empty else []


def load_metrics() -> pd.DataFrame:
    """Métriques de la dernière évaluation (prévu vs réalisé)."""
    return _query(
        """
        SELECT period_start, period_end, variable, horizon_h, mae, rmse,
               mape, bias, coverage, n_points, model_version
        FROM forecast_metrics
        WHERE eval_ts = (SELECT max(eval_ts) FROM forecast_metrics)
        ORDER BY variable, horizon_h
        """
    )


def load_runs(limit: int = 25) -> pd.DataFrame:
    """Dernières exécutions de pipelines (horodatages en heure locale)."""
    runs = _query(
        f"""
        SELECT run_ts, dag_id, status, rows, round(duration_s::numeric, 1) AS duration_s, message
        FROM pipeline_runs
        ORDER BY id DESC
        LIMIT {int(limit)}
        """
    )
    if not runs.empty:
        runs["run_ts"] = _to_local(runs["run_ts"])
    return runs
