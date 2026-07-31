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

# production_total reste sélectionnable (dérivé : somme des 4 sources,
# cf. local_forecaster.derive_production_total) ; les 4 sources sont
# désormais les cibles réellement modélisées (cf. TARGET_PREFIXES).
FEATURES = ["production_total", "consommation_totale", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR"]
PRODUCTION_SOURCES = ["SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR"]

# Jours de chauffe ajoutés avant la fenêtre affichée pour que la calibration
# conforme soit déjà mûre au premier point montré (~720 résidus à 30 j).
CALIBRATION_WARMUP_DAYS = 30

# Mémoire de la calibration conforme, en heures. 14 jours = deux cycles
# hebdomadaires complets (cf. `_conformal_bounds`).
CALIBRATION_ROLLING_HOURS = 336


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


ENERGY_COLORS = {
    "production_total": "#f1c40f",
    "SOLAR": "#f39c12",
    "BIOMASS": "#8e6c3a",
    "WIND_ONSHORE": "#3498db",
    "NUCLEAR": "#9b59b6",
}


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


def load_forecast_for_day(variable: str, start, end) -> pd.DataFrame:
    """Prévision couvrant toute une plage [start, end] (typiquement une
    journée civile 00h-23h), quitte à mélanger plusieurs origines : un batch
    ne peut jamais prédire une heure antérieure à sa propre origine, donc les
    premières heures d'aujourd'hui (déjà réalisées) ne sont couvertes que par
    le batch d'hier -- exactement comme la prévision RTE (J-1), qui elle
    aussi porte sur la journée entière indépendamment de notre notion
    d'origine/horizon.

    Pour chaque `target_ts`, on garde la prévision issue de l'origine la
    plus récente disponible (`DISTINCT ON` + tri décroissant sur origin_ts).
    """
    if variable not in FEATURES:
        raise ValueError(f"Variable inconnue : {variable}")

    start_utc = pd.Timestamp(start).tz_localize(DISPLAY_TZ).tz_convert("UTC").tz_localize(None)
    end_utc = pd.Timestamp(end).tz_localize(DISPLAY_TZ).tz_convert("UTC").tz_localize(None)

    fc = _query(
        f"""
        SELECT DISTINCT ON (target_ts)
               run_ts, origin_ts, target_ts, horizon_h, y_pred, y_lower, y_upper, model_version
        FROM forecasts
        WHERE variable = '{variable}'
          AND target_ts BETWEEN '{start_utc}' AND '{end_utc}'
        ORDER BY target_ts, origin_ts DESC
        """
    )
    if not fc.empty:
        for column in ("run_ts", "origin_ts", "target_ts"):
            fc[column] = _to_local(fc[column])
        fc = fc.sort_values("target_ts")
    return fc


def load_rte_consumption_forecast(start, end) -> pd.DataFrame:
    """Prévision officielle RTE J-1 de la consommation (repère de crédibilité,
    cf. scripts/fetch_rte_forecast.py), sur la fenêtre [start, end].

    Filtrée par `target_ts`, pas par origine : cette prévision externe ne
    partage pas la même logique d'origine/horizon que notre propre modèle,
    seule l'heure visée compte pour la superposer sur le même graphique.
    """
    start_local = pd.Timestamp(start)
    end_local = pd.Timestamp(end)
    # Les bornes sont en heure locale (comme le reste de l'app) ; la table
    # stocke en UTC -> on repasse en UTC naïf pour le filtre SQL.
    start_utc = start_local.tz_localize(DISPLAY_TZ).tz_convert("UTC").tz_localize(None)
    end_utc = end_local.tz_localize(DISPLAY_TZ).tz_convert("UTC").tz_localize(None)

    fc = _query(
        f"""
        SELECT target_ts, y_pred
        FROM forecasts
        WHERE variable = 'consommation_totale_rte'
          AND target_ts BETWEEN '{start_utc}' AND '{end_utc}'
        ORDER BY target_ts
        """
    )
    if not fc.empty:
        fc["target_ts"] = _to_local(fc["target_ts"])
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
    """Fenêtre d'observations contiguë des `days` derniers jours (heure locale).

    Réservé à l'AFFICHAGE. Ne jamais alimenter le modèle avec : cf.
    `_recent_window_utc`.
    """
    obs = _observations_local()
    if obs is None or obs.empty:
        return None
    return obs[obs.index >= obs.index.max() - pd.Timedelta(days=days)].sort_index()


def _recent_window_utc(days: int):
    """Fenêtre d'observations en UTC — la seule à donner au modèle.

    Le modèle est entraîné sur un index UTC naïf (`datasets/data.csv`) et ses
    features calendaires convertissent elles-mêmes en heure locale
    (`build_french_calendar`). Lui passer un index déjà converti provoque une
    DOUBLE conversion : le modèle croit qu'il est 23 h alors qu'il est 21 h.

    Mesuré avant correction : +201 % de MAE à h+1 et +231 % à h+6 sur le
    backtest affiché (355 -> 1 070 MW et 1 008 -> 3 335 MW), avec un profil en
    cloche caractéristique — l'erreur culminait vers h+6 puis retombait à h+24,
    où l'ancre saisonnière absorbe le décalage. La conversion en heure locale
    doit rester la toute dernière étape du rendu.
    """
    obs = _get_observations()
    if obs is None or obs.empty:
        return None
    obs = obs.sort_index()
    return obs[obs.index >= obs.index.max() - pd.Timedelta(days=days)]


def _min_calibration(alpha: float) -> int:
    """Plus petit échantillon permettant de VISER 1-alpha.

    L'indice conforme est le ⌈(n+1)(1-alpha)⌉-ième résidu trié ; il doit tenir
    dans n, d'où n >= (1-alpha)/alpha. Pour alpha=0,05 il faut 19 points : avec
    10, la couverture maximale atteignable est le maximum des résidus, soit
    10/11 = 90,9 % — impossible d'annoncer 95 %.
    """
    return int(np.ceil((1 - alpha) / alpha))


def _conformal_bounds(
    y_pred,
    y_true,
    alpha=0.05,
    step_index=None,
    min_calibration=None,
    fenetre=CALIBRATION_ROLLING_HOURS,
):
    """Bornes préquentielles : chaque point utilise seulement les résidus passés.

    Les premiers points restent sans borne tant que l'historique de calibration
    est insuffisant. La couverture affichée porte donc sur un vrai hors-calibration.

    `fenetre` borne la mémoire de calibration (None = expansive). Une fenêtre
    GLISSANTE est indispensable ici : la variance des résidus n'est pas
    stationnaire, et une calibration expansive converge vers une largeur moyenne
    qui ne convient à aucun régime. Mesuré sur 90 jours de backtest :

        calibration    h+1 (7j/90j)        h+24 (7j/90j)     largeur h+24
        expansive      88,8 % / 95,0 %     100 % / 97,2 %      6 933 MW
        glissante 14j  91,7 % / 95,1 %    99,4 % / 95,0 %      6 093 MW

    L'expansive sous-couvre le court terme à h+1 (semaine caniculaire, +18 % de
    volatilité) ET sur-couvre à h+24, avec un intervalle 14 % trop large. La
    fenêtre de 14 jours couvre deux cycles hebdomadaires complets : une fenêtre
    de 7 jours ne contient qu'un exemplaire de chaque jour de semaine, ce qui
    rend le quantile sensible au type de jour tombé dans la fenêtre.
    """
    if min_calibration is None:
        min_calibration = _min_calibration(alpha)
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
            # Correction d'échantillon fini : le quantile conforme est
            # ⌈(n+1)(1-alpha)⌉/n, pas (1-alpha). Sans elle l'intervalle est
            # légèrement anti-conservateur, d'autant plus que n est petit.
            niveau = min(1.0, np.ceil((len(past) + 1) * (1 - alpha)) / len(past))
            quantile = float(np.quantile(past, niveau, method="higher"))
            lower[index] = pred[index] - quantile
            upper[index] = pred[index] + quantile
        if np.isfinite(residuals[index]):
            past.append(float(residuals[index]))
            if fenetre is not None and len(past) > fenetre:
                del past[0]
    return lower, upper


def load_backtest(variable: str, horizon: int, days: int) -> pd.DataFrame:
    """Backtest à l'horizon `horizon` : chaque point est prédit directement
    depuis son origine (H-`horizon`) par le modèle dédié à cet horizon — plus
    de distinction autorégressif/glissant avec cette architecture (cf.
    local_forecaster.backtest_direct) : un horizon = un modèle direct,
    jamais nourri par ses propres prédictions passées.
    """
    if variable not in FEATURES:
        raise ValueError(f"Variable inconnue : {variable}")

    # Import tardif : la stack ML n'est pas requise pour le reste de l'app.
    from pipeline_prevision.utils.ml_utils.model.local_forecaster import backtest_direct

    # Contexte étendu, pour deux raisons distinctes :
    #  - les features (lags jusqu'à 336 h) ont besoin de bien plus d'historique
    #    que la fenêtre affichée ;
    #  - la calibration conforme est préquentielle : si elle redémarrait au
    #    début de la fenêtre, l'intervalle serait trop étroit sur les premiers
    #    points et la couverture affichée dépendrait du filtre choisi. Mesuré
    #    avant correction : les MÊMES 159 heures donnaient 91,8 % en fenêtre
    #    7 j et 96,2 % en fenêtre 90 j, avec un IC 13 % plus étroit. On calibre
    #    donc sur tout le contexte, puis on ne coupe QU'À L'AFFICHAGE.
    # Fenêtre en UTC : le modèle attend l'index dans lequel il a été entraîné
    # (cf. `_recent_window_utc`), la conversion d'affichage vient en sortie.
    window = _recent_window_utc(days + CALIBRATION_WARMUP_DAYS)
    if window is None:
        return pd.DataFrame()

    if variable == "production_total":
        # Plus une cible modélisée : reconstruite en sommant le backtest des
        # 4 sources (jointure stricte sur target_ts via `join="inner"` --
        # une heure ne compte que si les 4 sources ont produit un point).
        per_source = [backtest_direct(window, source, horizon) for source in PRODUCTION_SOURCES]
        if any(b.empty for b in per_source):
            return pd.DataFrame()
        preds = pd.concat([b["y_pred"] for b in per_source], axis=1, join="inner")
        trues = pd.concat([b["y_true"] for b in per_source], axis=1, join="inner")
        bt = pd.DataFrame({"y_pred": preds.sum(axis=1), "y_true": trues.sum(axis=1)})
    else:
        bt = backtest_direct(window, variable, horizon)

    if bt.empty:
        return pd.DataFrame()

    pred, reel = bt["y_pred"].to_numpy(), bt["y_true"].to_numpy()
    bas, haut = _conformal_bounds(pred, reel)

    resultat = pd.DataFrame({
        # Dernière étape seulement : les horodatages passent en heure
        # d'affichage une fois toutes les prédictions et bornes calculées.
        "target_ts": _to_local(bt.index),
        "y_pred": pred, "y_true": reel, "y_lower": bas, "y_upper": haut,
    })
    # Découpe d'affichage APRÈS calibration : la période de chauffe alimente les
    # bornes sans jamais peser dans les métriques montrées.
    debut = resultat["target_ts"].max() - pd.Timedelta(days=days)
    return resultat[resultat["target_ts"] >= debut].reset_index(drop=True)


def available_horizons() -> list:
    """Horizons disponibles dans la table des prévisions."""
    df = _query("SELECT DISTINCT horizon_h FROM forecasts ORDER BY horizon_h")
    return df["horizon_h"].tolist() if not df.empty else []


def load_metrics() -> pd.DataFrame:
    """Dernière évaluation disponible, par variable.

    Chaque variable est évaluée dès que son réalisé existe (asynchrone selon
    l'horizon et la date de mise en service de son modèle) : un unique
    `MAX(eval_ts)` global ferait disparaître silencieusement toute variable
    non couverte par le tout dernier lot (ex. une source de production tout
    juste réactivée dont le réalisé n'est pas encore connu pour tous les
    horizons). On prend donc le dernier lot propre à chaque variable.
    """
    return _query(
        """
        SELECT fm.period_start, fm.period_end, fm.variable, fm.horizon_h, fm.mae, fm.rmse,
               fm.mape, fm.bias, fm.coverage, fm.n_points, fm.model_version
        FROM forecast_metrics fm
        INNER JOIN (
            SELECT variable, max(eval_ts) AS eval_ts
            FROM forecast_metrics
            GROUP BY variable
        ) latest ON fm.variable = latest.variable AND fm.eval_ts = latest.eval_ts
        ORDER BY fm.variable, fm.horizon_h
        """
    )


