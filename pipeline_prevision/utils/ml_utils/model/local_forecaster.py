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

from pipeline_prevision.utils.main_utils.utils import load_object
from pipeline_prevision.utils.main_utils.feature_engineering import (
    HORIZON_MAX, TARGET_PREFIXES, add_target_features, build_origin_feature_frame,
    build_series_by_target, seasonal_baseline,
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
    return result

# --- Correction de niveau dynamique (biais EWMA par horizon) ----------------
# Les modèles montrent souvent un profil horaire correct mais un biais de
# niveau persistant (cf. carte « Biais moyen » du dashboard). Cette correction
# apprend ce biais sur les erreurs SIGNÉES de la sortie finale (modèle +
# alpha + seasonal_weight, cf. `_direct_prediction`) -- jamais sur le résidu
# brut du modèle avant cette correction, sinon on corrigerait deux fois la
# même chose. Walk-forward strict (chaque point ne voit que les erreurs
# passées, jamais la sienne) : pas de fuite, même rejoué sur tout un
# historique de backtest. Réduite (shrinkage) et plafonnée (% de la valeur
# réelle récente) plutôt qu'appliquée brute : une vraie dérive du modèle doit
# remonter via retrain_on_degradation, pas être rustinée indéfiniment ici.
BIAS_LAMBDA = 0.99        # EWMA à cadence horaire (origines) -> mémoire ~3 jours
BIAS_SHRINKAGE = 0.5      # n'applique que la moitié du biais estimé
BIAS_CAP_FRACTION = 0.05  # plafond : ±5 % de la valeur réelle récente (rolling 7 j)
BIAS_TYPICAL_WINDOW = 24 * 7

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


def _build_series(observations: pd.DataFrame):
    series_by_target = build_series_by_target(observations)
    temp = pd.to_numeric(observations["temp"], errors="coerce")
    return series_by_target, temp


def _direct_prediction(model, alpha: float, seasonal_weight: float,
                       feature_row: pd.DataFrame, horizon: int, prefix: str, delta_col: str):
    X = add_target_features(feature_row, horizon, delta_col)
    residual_pred = model.predict(X)
    persistence = feature_row[f"{prefix}_0"].to_numpy()
    seasonal_value = seasonal_baseline(feature_row, horizon, prefix)
    direct = persistence + alpha * residual_pred
    blended = (1 - seasonal_weight) * direct + seasonal_weight * seasonal_value
    return np.maximum(blended, 0.0)


def _walkforward_bias_correction(y_true: pd.Series, y_pred_raw: np.ndarray) -> np.ndarray:
    """Correction de niveau causale, point par point, sur une série ordonnée
    chronologiquement : à la position i, ne dépend que des erreurs signées
    aux positions < i (jamais de la sienne) -> utilisable telle quelle sur un
    historique de backtest sans fuite.
    """
    if len(y_true) == 0:
        return np.array([])

    error = pd.Series(y_true.to_numpy() - y_pred_raw, index=y_true.index)
    bias = error.shift(1).ewm(alpha=1 - BIAS_LAMBDA, adjust=False).mean().fillna(0.0)

    typical = y_true.shift(1).abs().rolling(BIAS_TYPICAL_WINDOW, min_periods=24).mean().fillna(0.0)
    cap = BIAS_CAP_FRACTION * typical

    return np.clip(BIAS_SHRINKAGE * bias.to_numpy(), -cap.to_numpy(), cap.to_numpy())


def _live_bias_correction(y_true: pd.Series, y_pred_raw: np.ndarray) -> float:
    """Correction pour une prévision live (pas encore réalisée) : même
    principe, mais utilise tout l'historique de calibration connu -- le point
    live n'a par définition pas encore de réalisé, donc aucune fuite.
    """
    if len(y_true) == 0:
        return 0.0

    error = pd.Series(y_true.to_numpy() - y_pred_raw, index=y_true.index)
    bias = float(error.ewm(alpha=1 - BIAS_LAMBDA, adjust=False).mean().iloc[-1])
    typical = float(y_true.tail(BIAS_TYPICAL_WINDOW).abs().mean())
    if not pd.notna(typical) or typical == 0:
        return 0.0
    cap = BIAS_CAP_FRACTION * typical
    return float(np.clip(BIAS_SHRINKAGE * bias, -cap, cap))


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

        series_by_target, temp = _build_series(observations)
        features_df, _ = build_origin_feature_frame(series_by_target, temp, target)
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
                model, alpha, seasonal_weight, origin_row[feature_columns], horizon, prefix, delta_col
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
    """Prévision directe + correction de biais + intervalles conformes,
    par horizon.

    Ordre strict (cf. `_walkforward_bias_correction`) :
      1. prévision brute finale = modèle + alpha + seasonal_weight ;
      2. + correction de niveau dynamique (biais EWMA récent, réduite et
         plafonnée) -> prévision opérationnelle, celle qui est persistée ;
      3. intervalle conforme calibré AUTOUR de la prévision opérationnelle
         (les résidus de calibration sont recalculés sur les prévisions
         historiques elles aussi corrigées, pour rester cohérents).

    Calibration : pour chaque horizon, on compare aux `n_calib` dernières
    origines historiques dont la vraie valeur cible est déjà connue (jamais
    de rollout) -> résidus |réel - prévu corrigé|, demi-largeur =
    quantile(1-alpha).
    """
    try:
        _validate_target(target)
        horizons = list(horizons) if horizons is not None else list(range(1, HORIZON_MAX + 1))

        composite = get_models()[target]
        feature_columns = composite["feature_columns"]
        prefix = composite["prefix"]
        delta_col = f"{prefix}_delta_1"

        series_by_target, temp = _build_series(observations)
        features_df, _ = build_origin_feature_frame(series_by_target, temp, target)
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
                horizon, prefix, delta_col,
            )[0]

            half_width = np.nan
            correction_live = 0.0
            if not calib_frame.empty:
                y_pred_calib = _direct_prediction(
                    model, alpha_correction, seasonal_weight, calib_frame[feature_columns],
                    horizon, prefix, delta_col,
                )
                # Biais appris sur l'erreur de la sortie finale non corrigée
                # (calib_actual - y_pred_calib) : jamais sur le résidu brut
                # d'avant alpha/seasonal_weight, pour ne pas corriger deux fois.
                correction_live = _live_bias_correction(calib_actual, y_pred_calib)
                y_pred_calib_corrected = y_pred_calib + _walkforward_bias_correction(calib_actual, y_pred_calib)
                residuals = np.abs(calib_actual.to_numpy() - y_pred_calib_corrected)
                recent_residuals = residuals[-n_calib:]
                if len(recent_residuals):
                    half_width = float(np.quantile(recent_residuals, 1 - alpha))

            y_pred_corrected = max(y_pred_live + correction_live, 0.0)

            y_lower = max(y_pred_corrected - half_width, 0.0) if pd.notna(half_width) else np.nan
            y_upper = y_pred_corrected + half_width if pd.notna(half_width) else np.nan

            rows.append({
                "target_ts": origin_ts + pd.Timedelta(hours=horizon),
                "horizon_h": horizon,
                "y_pred": float(y_pred_corrected),
                "y_lower": y_lower,
                "y_upper": y_upper,
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

        series_by_target, temp = _build_series(observations)
        features_df, _ = build_origin_feature_frame(series_by_target, temp, target)
        series = series_by_target[target]
        actual_future = series.shift(-horizon)

        mask = features_df[feature_columns].notna().all(axis=1) & actual_future.notna()
        frame = features_df.loc[mask]
        y_true = actual_future.loc[mask]

        if frame.empty:
            return pd.DataFrame()

        # Correction calculée sur tout l'historique dispo (jamais seulement la
        # fenêtre `days` affichée) : la bande walk-forward a besoin de son
        # propre passé pour "chauffer" avant le début de la période visible,
        # sinon elle repartirait de zéro à chaque rechargement du graphe.
        # Le découpage à `days` n'intervient qu'à la toute fin, sur le résultat.
        y_pred = _direct_prediction(model, alpha, seasonal_weight, frame[feature_columns], horizon, prefix, delta_col)
        correction = _walkforward_bias_correction(y_true, y_pred)
        y_pred_corrected = np.maximum(y_pred + correction, 0.0)
        target_ts = frame.index + pd.Timedelta(hours=horizon)

        result = pd.DataFrame({
            "target_ts": target_ts,
            "y_pred": y_pred_corrected,
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
