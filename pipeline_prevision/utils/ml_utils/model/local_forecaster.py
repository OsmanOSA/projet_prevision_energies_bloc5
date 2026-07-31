"""Inférence locale : prévision directe multi-horizon résiduelle.

Charge les 24 modèles par cible (SOLAR, BIOMASS, WIND_ONSHORE, NUCLEAR,
consommation_totale), entraînés par `model_trainer.py`, et calcule les
prévisions **directement** depuis l'origine (dernière heure observée) — pas
de boucle autorégressive : chaque horizon a son propre modèle, ancré sur des
features toutes réelles (jamais reconstruites à partir d'une prédiction
précédente). Cf. `feature_engineering.py` pour le détail des features et la
méthodologie (validée initialement sur production_total dans
notebooks/baseline_model.ipynb, +35 % de MAE vs persistance sur 24h ; la
décomposition par source vise à isoler l'éolien, seule composante volatile,
des composantes stables qui noyaient sa performance dans l'agrégat).

production_total n'est plus une cible modélisée : les scripts de prévision
(cf. `scripts/forecast.py`) le reconstruisent en sommant les 4 sources.

Le dossier des artefacts est `final_models/` par défaut, surchageable via la
variable d'environnement MODEL_DIR.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.utils.main_utils.utils import load_object
from pipeline_prevision.utils.main_utils.feature_engineering import (
    HORIZON_MAX, TARGET_PREFIXES, add_target_features, build_origin_feature_frame,
    build_series_by_target, seasonal_baseline,
    DEFAULT_ANCHOR, anchor_values, complementary_anchor, select_temperature,
    select_forecast_temperature,
)
from pipeline_prevision.exception.exception import ForecastingException

# Les 4 sources de production (SOLAR/BIOMASS/WIND_ONSHORE/NUCLEAR) + la
# consommation -- production_total est reconstruit en aval (somme des 4
# sources), plus jamais prédit directement (cf. scripts/forecast.py).
TARGETS = list(TARGET_PREFIXES)
PRODUCTION_SOURCES = [t for t in TARGETS if t != "consommation_totale"]


def derive_production_total(per_target: dict) -> dict:
    """Reconstruit production_total en sommant les prévisions des 4 sources
    (y_pred, et y_lower/y_upper si présentes -- bornes combinées par simple
    somme, comme pour le graphe du déficit : approximation conservatrice qui
    suppose les erreurs des 4 sources indépendantes).
    """
    result = {"y_pred": sum(per_target[s]["y_pred"] for s in PRODUCTION_SOURCES)}
    if all("y_lower" in per_target[s] for s in PRODUCTION_SOURCES):
        result["y_lower"] = sum(per_target[s]["y_lower"] for s in PRODUCTION_SOURCES)
        result["y_upper"] = sum(per_target[s]["y_upper"] for s in PRODUCTION_SOURCES)
    if all("y_decision" in per_target[s] for s in PRODUCTION_SOURCES):
        result["y_decision"] = sum(per_target[s]["y_decision"] for s in PRODUCTION_SOURCES)
    return result

# --- Pas de correction de niveau dynamique (retirée, mesurée perdante) ------
# Une correction de biais EWMA par horizon a existé ici (constantes BIAS_*,
# `_live_bias_correction` / `_walkforward_bias_correction`). Elle a été
# retirée après mesure -- ne pas la réintroduire sans rejouer
# `scripts/validate_bias_params.py`.
#
# 1. La variante walk-forward FUYAIT. Ses séries sont indexées par ORIGINE
#    (`series.shift(-horizon)`), donc son `.shift(1)` reculait d'une origine,
#    soit 1 h -- alors que l'erreur de l'origine `i-k` n'est observable qu'à
#    `i-k+h`. Il aurait fallu `.shift(horizon)`. À h=24, la correction lisait
#    un réalisé arrivant 23 h plus tard, ce qui gonflait `backtest_direct` et
#    les résidus de calibration conforme.
# 2. Une fois le décalage rendu causal, la correction perd sur les 5 cibles
#    (gain sur moitié de confirmation : SOLAR -4.5 %, BIOMASS -3.3 %,
#    WIND_ONSHORE -0.6 %, NUCLEAR -1.1 %, consommation -1.2 % ; taux de
#    correction nuisible ~50 %). Le grid-search causal converge vers
#    « ne rien corriger » sur les quatre paramètres à la fois.
# 3. Signature la plus nette : à h=1, seul horizon où l'ancien `.shift(1)`
#    était déjà causal, la correction n'apportait rien (-0.01 % à -0.87 %).
#
# Un biais de niveau qui persiste doit remonter via retrain_on_degradation,
# pas être rustiné en ligne.

# --- Trous de température ---------------------------------------------------
# `temp` est exogène (Meteostat) et publiée avec un retard variable. Un seul
# NaN suffit à invalider les 168 h de features SUIVANTES (temp_lag_168 et les
# rolling 168 h, cf. feature_engineering) : une heure météo manquante rendrait
# le modèle inexploitable pendant une semaine entière. À l'échelle horaire la
# température est lisse -> combler les trous COURTS et INTERNES par
# interpolation linéaire est légitime (et cohérent avec l'ingestion, cf.
# `concat_all_data`). `limit_area="inside"` laisse volontairement les NaN de
# fin de série : fabriquer l'heure la plus récente masquerait un retard
# d'ingestion. Dans ce cas l'origine recule simplement d'une heure ou deux, ce
# que `forecast_origin` rend explicite et que `scripts/forecast.py` contrôle.
TEMP_MAX_GAP_HOURS = 6

# --- Quantile de décision (§ 9 du notebook de comparaison) -------------------
# Une prévision ponctuelle minimisant la MAE vise la MÉDIANE. Or la décision
# adossée à cette prévision -- un engagement de capacité -- n'a pas un coût
# symétrique : sous-estimer la consommation oblige à un équilibrage d'urgence,
# surestimer n'immobilise que de la réserve. L'optimum n'est donc pas la
# médiane mais le quantile `c_sous / (c_sous + c_sur)`.
#
# Mesuré sur backtest à origine glissante (6 folds, calibration hors fold) :
# viser q0,762 au lieu de la médiane réduit le coût de déséquilibre de **19 %**
# (~10 400 €/h, soit ~91 M€/an aux coûts ci-dessous) -- **tout en DÉGRADANT la
# MAE de 21 %**. C'est le point : optimiser l'erreur et optimiser la décision
# ne sont pas le même problème.
#
# Les coûts sont des ordres de grandeur du marché d'ajustement français ; ils
# sont surchargeables par variable d'environnement. Tout changement de coût
# change le quantile optimal -- rejouer le § 9 du notebook avant d'y toucher.
COST_UNDER_FORECAST = float(os.getenv("COST_UNDER_FORECAST", "80"))  # €/MWh manquant
COST_OVER_FORECAST = float(os.getenv("COST_OVER_FORECAST", "25"))    # €/MWh excédentaire

# L'asymétrie ci-dessus décrit un ENGAGEMENT DE CAPACITÉ, donc la consommation.
# L'appliquer aux filières de production reviendrait à sur-annoncer
# systématiquement la production disponible -- exactement l'erreur inverse de
# celle qu'on cherche à éviter.
#
# ATTENTION -- les autres cibles n'ont PAS de quantile par défaut, et surtout
# pas 0,5. La médiane des résidus signés n'est pas nulle (mesuré : +836 MW sur
# NUCLEAR à h+24), donc viser q0,5 revient à appliquer une correction de niveau
# -- précisément celle qui a été retirée de ce module après mesure, parce
# qu'elle perdait sur les cinq cibles (cf. l'encadré ci-dessus). Sans règle de
# décision explicite, on ne retouche rien : `y_decision` vaut `y_pred`.
DECISION_QUANTILES = {
    "consommation_totale": COST_UNDER_FORECAST / (COST_UNDER_FORECAST + COST_OVER_FORECAST),
}


def decision_quantile(target: str) -> float | None:
    """Quantile de décision de la cible, ou None si aucune règle n'est définie.

    None signifie « ne pas retoucher la prévision », et non « viser la médiane ».
    """
    return DECISION_QUANTILES.get(target)

# Racine du projet : .../pipeline_prevision/utils/ml_utils/model/local_forecaster.py -> 5 niveaux
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

_model_cache: dict = {}


def get_model_dir() -> str:
    return os.getenv("MODEL_DIR", os.path.join(_ROOT, "final_models"))


def get_model_version() -> str:
    """Identifiant traçable : métadonnées d'entraînement ou hash de l'artefact."""
    model_dir = get_model_dir()
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as metadata_file:
                version = json.load(metadata_file).get("model_version")
            if version:
                return str(version)
        except (OSError, ValueError, TypeError):
            pass

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.isfile(model_path):
        return "indisponible"
    digest = hashlib.sha256()
    with open(model_path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:12]}"


def get_models() -> dict:
    """Charge (une seule fois) le dict composite {cible: {models, alphas, ...}}."""
    try:
        if "composite" not in _model_cache:
            model_path = os.path.join(get_model_dir(), "model.pkl")
            _model_cache["composite"] = load_object(model_path)
        return _model_cache["composite"]
    except Exception as e:
        raise ForecastingException(e, sys)


def _validate_target(target: str):
    if target not in TARGETS:
        raise ValueError(f"Cible inconnue : {target} (attendu : {TARGETS})")


def _temperature_prevue(observations: pd.DataFrame) -> pd.Series | None:
    """Température prévue à l'heure cible, pour l'inférence.

    Deux morceaux, et l'ordre de priorité compte :

    - le PASSÉ vient de `observations.temp_fr_prev`, l'archive à échéance J-1 sur
      laquelle le modèle a été entraîné ;
    - le FUTUR vient de l'API live, seule à couvrir t+1..t+24 — la table
      `observations` n'a aucune ligne au-delà de la dernière observation, donc la
      colonne archivée ne peut structurellement pas renseigner l'horizon utile.

    L'archive est prioritaire là où elle existe (`combine_first`) : laisser la
    prévision live, plus fraîche, écraser les heures passées mélangerait les
    millésimes entre l'entraînement et la calibration des intervalles conformes,
    qui se calent justement sur les origines récentes.

    En cas d'échec réseau on retourne l'archive seule. Les features de l'origine
    la plus récente restent alors NaN, et le forecaster s'ancre sur une origine
    plus ancienne mais complète (cf. `valid_features` / `forecast_origin`) : une
    prévision légèrement décalée plutôt qu'aucune prévision.
    """
    archive = select_forecast_temperature(observations)

    try:
        from pipeline_prevision.utils.main_utils.prevision_temperature_france import (
            prevision_france,
        )
        derniere = pd.Timestamp(observations.index.max())
        live = prevision_france(
            (derniere - pd.Timedelta(hours=6)).to_pydatetime(),
            (derniere + pd.Timedelta(hours=HORIZON_MAX + 1)).to_pydatetime(),
        )
    except Exception as e:
        logging.warning(
            "Prévision de température live indisponible (%s) : repli sur l'archive "
            "seule. L'origine la plus récente sera incomplète et le forecaster "
            "s'ancrera plus tôt.", e)
        return archive

    return archive.combine_first(live) if archive is not None else live


def _build_series(observations: pd.DataFrame):
    series_by_target = build_series_by_target(observations)
    temp = select_temperature(observations)
    temp = temp.interpolate(method="linear", limit=TEMP_MAX_GAP_HOURS, limit_area="inside")
    temp_prev = _temperature_prevue(observations)
    return series_by_target, temp, temp_prev


def forecast_origin(prediction: pd.DataFrame) -> pd.Timestamp:
    """Origine RÉELLEMENT utilisée par une prévision (`predict_direct` ou
    `predict_with_conformal_intervals`), reconstruite depuis `target_ts - h`.

    À ne pas confondre avec la dernière observation en base : le forecaster
    s'ancre sur la dernière origine dont TOUTES les features sont calculables
    (cf. `valid_features`), qui peut être plus ancienne si une série exogène
    a un trou. Étiqueter la prévision avec la dernière observation alors que
    le modèle s'est ancré ailleurs produit des horizons faux — négatifs si
    l'écart dépasse l'horizon — d'où ce point d'accès explicite.
    """
    if prediction.empty:
        raise ValueError("Prévision vide : aucune origine à en déduire")
    return prediction.index[0] - pd.Timedelta(hours=int(prediction["horizon_h"].iloc[0]))


def get_anchor(composite: dict, horizon: int) -> str:
    """Ancre retenue à l'entraînement pour cet horizon.

    Les artefacts entraînés avant la sélection d'ancre n'ont pas la clé
    `anchors` : ils sont tous ancrés sur la persistance, d'où ce repli qui
    laisse un `model.pkl` ancien produire exactement les mêmes prévisions
    qu'avant.
    """
    return composite.get("anchors", {}).get(horizon, DEFAULT_ANCHOR)


def _align_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """Restreint X aux colonnes que le modèle a réellement vues à l'entraînement.

    `add_target_features` construit les variables ancrées sur l'heure cible
    APRÈS la sélection par `feature_columns` : tout enrichissement de cette
    fonction change donc la largeur de X et casse les artefacts déjà entraînés
    (« number of features in data (182) is not the same as in training data
    (168) »). LightGBM conserve le nom de ses colonnes -- on s'y aligne, ce qui
    laisse un ancien champion prédire exactement comme avant tout en permettant
    aux modèles réentraînés d'exploiter les nouvelles variables.
    """
    attendues = getattr(model, "feature_name_", None)
    if not attendues or list(X.columns) == list(attendues):
        return X
    manquantes = [c for c in attendues if c not in X.columns]
    if manquantes:
        raise ValueError(
            f"features attendues par le modèle et absentes du jeu construit : "
            f"{manquantes[:5]}{'...' if len(manquantes) > 5 else ''}")
    return X[list(attendues)]


def _direct_prediction(model, alpha: float, seasonal_weight: float,
                       feature_row: pd.DataFrame, horizon: int, prefix: str, delta_col: str,
                       anchor: str = DEFAULT_ANCHOR):
    X = _align_to_model(model, add_target_features(feature_row, horizon, delta_col))
    residual_pred = model.predict(X)
    base = anchor_values(feature_row, horizon, prefix, anchor)
    blend = anchor_values(feature_row, horizon, prefix, complementary_anchor(anchor))
    direct = base + alpha * residual_pred
    blended = (1 - seasonal_weight) * direct + seasonal_weight * blend
    return np.maximum(blended, 0.0)


def predict_direct(observations: pd.DataFrame, target: str, horizons=None) -> pd.DataFrame:
    """Prévision directe pour `target`, aux horizons demandés (1..24 par défaut).

    `observations` : DataFrame indexé par timestamp, colonnes SOLAR/BIOMASS/
    WIND_ONSHORE/NUCLEAR/consommation_totale/temp, avec au moins ~337h
    d'historique propre (plus grand lag utilisé). Retourne un DataFrame
    indexé par `target_ts` avec `y_pred`.
    """
    try:
        _validate_target(target)
        horizons = list(horizons) if horizons is not None else list(range(1, HORIZON_MAX + 1))

        composite = get_models()[target]
        feature_columns = composite["feature_columns"]
        prefix = composite["prefix"]
        delta_col = f"{prefix}_delta_1"

        series_by_target, temp, temp_prev = _build_series(observations)
        features_df, _ = build_origin_feature_frame(series_by_target, temp, target,
                                                    temp_prev=temp_prev)
        features_df = features_df.dropna(subset=feature_columns)
        if features_df.empty:
            raise ValueError(f"Historique insuffisant pour calculer les features de {target}")

        origin_row = features_df.iloc[[-1]]
        origin_ts = origin_row.index[0]

        rows = []
        for horizon in horizons:
            model = composite["models"][horizon]
            alpha = composite["alphas"][horizon]
            seasonal_weight = composite["seasonal_weights"][horizon]
            y_pred = _direct_prediction(
                model, alpha, seasonal_weight, origin_row[feature_columns], horizon, prefix, delta_col,
                get_anchor(composite, horizon),
            )[0]
            rows.append({
                "target_ts": origin_ts + pd.Timedelta(hours=horizon),
                "horizon_h": horizon,
                "y_pred": float(y_pred),
            })

        return pd.DataFrame(rows).set_index("target_ts")

    except ForecastingException:
        raise
    except Exception as e:
        raise ForecastingException(e, sys)


def predict_with_conformal_intervals(observations: pd.DataFrame, target: str, horizons=None,
                                     alpha: float = 0.05, n_calib: int = 200) -> pd.DataFrame:
    """Prévision directe + intervalles conformes, par horizon.

    La prévision persistée est exactement la sortie du modèle (modèle + alpha
    + seasonal_weight) : plus aucune correction de niveau ne s'intercale ici
    (cf. l'encadré en tête de module).

    Calibration : pour chaque horizon, on compare aux `n_calib` dernières
    origines historiques dont la vraie valeur cible est déjà connue (jamais
    de rollout) -> résidus |réel - prévu|, demi-largeur = quantile(1-alpha).

    Trois sorties, à ne pas confondre :
      - `y_pred`     : la prévision du modèle, sans retouche (minimise la MAE) ;
      - `y_lower`/`y_upper` : intervalle conforme symétrique à 1-alpha ;
      - `y_decision` : la valeur à ENGAGER, décalée au quantile de décision
        propre à la cible (cf. `decision_quantile`). Elle est délibérément moins
        bonne en MAE et meilleure en coût -- c'est le § 9 du notebook.
    """
    try:
        _validate_target(target)
        horizons = list(horizons) if horizons is not None else list(range(1, HORIZON_MAX + 1))
        q_decision = decision_quantile(target)

        composite = get_models()[target]
        feature_columns = composite["feature_columns"]
        prefix = composite["prefix"]
        delta_col = f"{prefix}_delta_1"

        series_by_target, temp, temp_prev = _build_series(observations)
        features_df, _ = build_origin_feature_frame(series_by_target, temp, target,
                                                    temp_prev=temp_prev)
        series = series_by_target[target]

        valid_features = features_df[feature_columns].notna().all(axis=1)
        origin_row = features_df.loc[valid_features].iloc[[-1]]
        origin_ts = origin_row.index[0]

        rows = []
        for horizon in horizons:
            model = composite["models"][horizon]
            alpha_correction = composite["alphas"][horizon]
            seasonal_weight = composite["seasonal_weights"][horizon]

            actual_future = series.shift(-horizon)
            calib_mask = valid_features & actual_future.notna()
            calib_frame = features_df.loc[calib_mask]
            calib_actual = actual_future.loc[calib_mask]

            y_pred_live = _direct_prediction(
                model, alpha_correction, seasonal_weight, origin_row[feature_columns],
                horizon, prefix, delta_col, get_anchor(composite, horizon),
            )[0]

            half_width = np.nan
            decision_shift = np.nan
            if not calib_frame.empty:
                y_pred_calib = _direct_prediction(
                    model, alpha_correction, seasonal_weight, calib_frame[feature_columns],
                    horizon, prefix, delta_col, get_anchor(composite, horizon),
                )
                # Résidus SIGNÉS : l'intervalle conforme n'a besoin que de leur
                # valeur absolue, mais le quantile de décision a besoin du signe
                # -- c'est toute la différence entre « à quelle distance ? » et
                # « de quel côté se tromper ? ».
                signed = (calib_actual.to_numpy() - y_pred_calib)[-n_calib:]
                if len(signed):
                    half_width = float(np.quantile(np.abs(signed), 1 - alpha))
                    if q_decision is not None:
                        decision_shift = float(np.quantile(signed, q_decision))

            y_pred_final = max(y_pred_live, 0.0)
            y_decision = (max(y_pred_live + decision_shift, 0.0)
                          if pd.notna(decision_shift) else y_pred_final)

            y_lower = max(y_pred_final - half_width, 0.0) if pd.notna(half_width) else np.nan
            y_upper = y_pred_final + half_width if pd.notna(half_width) else np.nan

            rows.append({
                "target_ts": origin_ts + pd.Timedelta(hours=horizon),
                "horizon_h": horizon,
                "y_pred": float(y_pred_final),
                "y_lower": y_lower,
                "y_upper": y_upper,
                "y_decision": float(y_decision),
                # NaN plutôt que None : garde la colonne en float, sans quoi
                # elle devient de type objet et casse concat/persistance dès
                # qu'une cible sans règle de décision est dans le lot.
                "decision_q": float(q_decision) if q_decision is not None else np.nan,
            })

        return pd.DataFrame(rows).set_index("target_ts")

    except ForecastingException:
        raise
    except Exception as e:
        raise ForecastingException(e, sys)


def backtest_direct(observations: pd.DataFrame, target: str, horizon: int, days: int | None = None) -> pd.DataFrame:
    """Backtest : prévision directe à `horizon` vs réel, sur l'historique
    disponible (ou les derniers `days` jours). Remplace à la fois l'ancien
    `predict_singlestep` (h=1) et `rolling_multistep` (h=k) : avec cette
    architecture il n'y a plus de distinction autorégressif/glissant — un
    horizon = un modèle dédié, direct, jamais nourri par ses propres
    prédictions passées.
    """
    try:
        _validate_target(target)

        composite = get_models()[target]
        feature_columns = composite["feature_columns"]
        prefix = composite["prefix"]
        delta_col = f"{prefix}_delta_1"
        model = composite["models"][horizon]
        alpha = composite["alphas"][horizon]
        seasonal_weight = composite["seasonal_weights"][horizon]

        series_by_target, temp, temp_prev = _build_series(observations)
        features_df, _ = build_origin_feature_frame(series_by_target, temp, target,
                                                    temp_prev=temp_prev)
        series = series_by_target[target]
        actual_future = series.shift(-horizon)

        mask = features_df[feature_columns].notna().all(axis=1) & actual_future.notna()
        frame = features_df.loc[mask]
        y_true = actual_future.loc[mask]

        if frame.empty:
            return pd.DataFrame()

        # Prévision calculée sur tout l'historique disponible, puis découpée à
        # `days` seulement à la fin : le résultat d'une origine ne dépend que
        # de ses propres features, donc la fenêtre affichée ne change aucune
        # valeur -- mais garder le calcul global évite de refaire dépendre le
        # graphe de son point de départ si un état glissant réapparaissait ici.
        y_pred = _direct_prediction(model, alpha, seasonal_weight, frame[feature_columns], horizon, prefix,
                                    delta_col, get_anchor(composite, horizon))
        target_ts = frame.index + pd.Timedelta(hours=horizon)

        result = pd.DataFrame({
            "target_ts": target_ts,
            "y_pred": np.maximum(y_pred, 0.0),
            "y_true": y_true.to_numpy(),
        }).set_index("target_ts")

        if days is not None and not result.empty:
            cutoff = result.index.max() - pd.Timedelta(days=days)
            result = result.loc[result.index >= cutoff]

        return result

    except ForecastingException:
        raise
    except Exception as e:
        raise ForecastingException(e, sys)
