"""Inférence locale de prévision (remplace l'ancien appel API Heroku).

Charge le modèle et le préprocesseur depuis `final_models/` et effectue la
prévision multi-horizon en local, sans dépendance réseau. Même signature de
retour que l'ancien client HTTP : (y_pred, y_test, mae, mse), pour un
remplacement transparent dans le dashboard et les DAGs.

Le dossier des artefacts est `final_models/` par défaut, surchargeable via
la variable d'environnement MODEL_DIR.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from pipeline_prevision.constant.training_pipeline import LOOKBACK, HORIZON
from pipeline_prevision.utils.main_utils.utils import load_object
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel
from pipeline_prevision.utils.ml_utils.metric.forecasting_metric import get_forecast_score
from pipeline_prevision.exception.exception import ForecastingException

# Ordre des colonnes attendu par le modèle.
FEATURES = ["temp", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale"]

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


def get_forecast_model() -> ForecastModel:
    """Charge (une seule fois) le ForecastModel depuis final_models/."""
    try:
        if "model" not in _model_cache:
            model_dir = get_model_dir()
            preprocessor = load_object(os.path.join(model_dir, "preprocessor.pkl"))
            model = load_object(os.path.join(model_dir, "model.pkl"))
            _model_cache["model"] = ForecastModel(preprocessor=preprocessor, model=model)
        return _model_cache["model"]
    except Exception as e:
        raise ForecastingException(e, sys)

def predict_singlestep(data):
    """Prévision 1 pas glissante — régime natif du modèle (HORIZON = 1).

    `ForecastModel.predict` s'appuie sur `window_generator` : il produit une
    fenêtre par pas de temps et prédit chaque fois l'heure suivante à partir
    des observations **réelles**. On obtient donc une série **continue**, sans
    autorégression et sans accumulation d'erreur — contrairement à
    `predict_multistep`.

    /!\\ Ne pas tronquer l'entrée à LOOKBACK lignes : `window_generator` a
    besoin d'au moins LOOKBACK + HORIZON + 1 lignes pour produire une fenêtre,
    et c'est justement le nombre de lignes fourni qui détermine la longueur de
    la courbe. Passer une période **contiguë** (les fenêtres à cheval sur un
    trou de données produiraient des valeurs aberrantes).

    data : DataFrame indexé par horodatage, contenant les colonnes FEATURES.
    Retourne (targets, y_pred, y_true, mae, mse) où `targets` est l'index des
    horodatages **prédits** (la fenêtre k prédit index[LOOKBACK + k]).
    """
    try:
        model = get_forecast_model()

        # NB : on lève des exceptions standard ici ; ForecastingException lit
        # sys.exc_info() et exige donc d'être construite depuis un bloc except
        # (le wrapper plus bas s'en charge).
        if not isinstance(data, pd.DataFrame):
            raise TypeError("predict_singlestep attend un DataFrame indexé")

        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            raise ValueError(f"Features manquantes: {missing}")

        df = data.sort_index()
        minimum = LOOKBACK + HORIZON + 1
        if len(df) < minimum:
            raise ValueError(
                f"Historique insuffisant : {len(df)} lignes fournies, {minimum} minimum")

        y_pred, y_true = model.predict(x=df[FEATURES].values.tolist())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        # La fenêtre k prédit la ligne index[LOOKBACK + k].
        targets = df.index[LOOKBACK:LOOKBACK + len(y_pred)]

        score = get_forecast_score(y_true=y_true, y_pred=y_pred)
        return targets, y_pred, y_true, float(score.mae), float(score.mse)

    except ForecastingException:
        raise
    except Exception as e:
        raise ForecastingException(e, sys)


def rolling_multistep(data, horizon: int):
    """Prévisions autorégressives enchaînées, aplaties en une courbe continue.

    Une origine tous les `horizon` pas ; chaque origine produit `horizon`
    prédictions autorégressives à partir des observations **réelles** connues à
    cet instant. Les blocs se suivent sans recouvrement : on obtient une courbe
    continue dont les « coutures » (changements d'origine) sont renvoyées pour
    être matérialisées à l'écran.

    C'est le mode d'exploitation « le modèle tourne toutes les `horizon` heures
    et prévoit `horizon` heures » : entre deux coutures l'erreur s'accumule,
    puis elle est remise à zéro par de nouvelles observations.

    Retourne (targets, y_pred, y_true, origins).
    """
    try:
        model = get_forecast_model()

        if not isinstance(data, pd.DataFrame):
            raise TypeError("rolling_multistep attend un DataFrame indexé")
        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            raise ValueError(f"Features manquantes: {missing}")
        if horizon < 1:
            raise ValueError("horizon doit être >= 1")

        df = data.sort_index()
        feats = df[FEATURES].to_numpy(dtype=float)
        index = df.index
        minimum = LOOKBACK + horizon + 1
        if len(feats) < minimum:
            raise ValueError(
                f"Historique insuffisant : {len(feats)} lignes fournies, {minimum} minimum")

        targets, preds, actuals, origins = [], [], [], []
        i = LOOKBACK - 1  # dernier indice de la première fenêtre disponible
        while i + horizon < len(feats):
            window = feats[i - LOOKBACK + 1:i + 1].tolist()
            block, _ = model.predict_multistep(x=window, n_futur=horizon)

            preds.append(np.asarray(block))
            actuals.append(feats[i + 1:i + 1 + horizon])
            targets.extend(index[i + 1:i + 1 + horizon])
            origins.append(index[i])
            i += horizon

        if not preds:
            raise ValueError("Aucun bloc de prévision n'a pu être constitué")

        return (pd.DatetimeIndex(targets),
                np.vstack(preds),
                np.vstack(actuals),
                pd.DatetimeIndex(origins))

    except ForecastingException:
        raise
    except Exception as e:
        raise ForecastingException(e, sys)


def predict_multistep(data, n_future: int):
    """Prévision multi-horizon locale.

    data : DataFrame contenant au moins les colonnes FEATURES (les LOOKBACK
           dernières lignes sont utilisées) ou séquence déjà prête.
    Retourne (y_pred, y_test, mae, mse) — y_test/mae/mse à None hors validation.
    """
    try:
        model = get_forecast_model()

        if isinstance(data, pd.DataFrame):
            missing = [f for f in FEATURES if f not in data.columns]
            if missing:
                raise ForecastingException(f"Features manquantes: {missing}", sys)
            x = data[FEATURES].tail(LOOKBACK).values.tolist()
        else:
            x = data

        y_pred, y_test = model.predict_multistep(x=x, n_futur=n_future)
        y_pred = np.asarray(y_pred)

        mae = mse = None
        if y_test is not None:
            y_test = np.asarray(y_test)
            score = get_forecast_score(y_true=y_test, y_pred=y_pred)
            mae, mse = float(score.mae), float(score.mse)

        return y_pred, y_test, mae, mse
    except ForecastingException:
        raise
    except Exception as e:
        raise ForecastingException(e, sys)



def _clip_non_negative(y_lower):
    """Borne inférieure >= 0 pour toutes les variables sauf la température (idx 0)."""
    y_lower = y_lower.copy()
    if y_lower.shape[1] > 1:
        y_lower[:, 1:] = np.maximum(y_lower[:, 1:], 0.0)
    return y_lower


def _heuristic_intervals(y_pred):
    """Intervalles de repli : incertitude relative croissante avec l'horizon.

    Utilisé quand il n'y a pas assez d'historique pour calibrer des résidus.
    Va d'environ 5 % au premier pas à ~20 % au dernier.
    """
    n_future = y_pred.shape[0]
    denom = max(n_future - 1, 1)
    pct = 0.05 + 0.15 * (np.arange(n_future) / denom)
    width = np.abs(y_pred) * pct[:, None]
    # min(., y_pred) : le clip à 0 ne doit jamais remonter la borne au-dessus
    # de la prédiction (cas d'une prévision légèrement négative).
    y_lower = np.minimum(_clip_non_negative(y_pred - width), y_pred)
    return y_pred, y_lower, y_pred + width


def dynamic_conformal_intervals(data,
                                n_future: int,
                                alpha: float = 0.05,
                                n_calib: int = 40):
    """Prévision multi-horizon locale avec intervalles conformes dynamiques.

    Approche (split conformal adapté au multi-horizon autorégressif) :
      1. Prédiction centrale sur la dernière fenêtre (predict_multistep).
      2. Calibration : on backteste le modèle sur les `n_calib` fenêtres
         récentes de `data`, en collectant les résidus |réalisé - prévu| pour
         chaque pas d'horizon h et chaque variable.
      3. Demi-largeur = quantile(1 - alpha) des résidus, par (horizon, variable).
         L'intervalle s'élargit donc naturellement avec l'horizon (l'erreur
         autorégressive se cumule).

    data : DataFrame avec au moins les colonnes FEATURES.
    alpha : niveau de risque (0.05 -> intervalles ~95 %).
    n_calib : nombre de fenêtres de calibration (bornées par l'historique dispo).

    Retourne (y_pred, y_lower, y_upper, y_test, mae, mse). En mode prédiction
    réelle, y_test/mae/mse valent None.
    """

    try:
        
        # Prédiction centrale (réutilise la logique validée).
        y_pred, y_test, mae, mse = predict_multistep(data, n_future)

        # Sans DataFrame, impossible de backtester -> repli heuristique.
        if not isinstance(data, pd.DataFrame):
            y_pred, y_lower, y_upper = _heuristic_intervals(y_pred)
            return y_pred, y_lower, y_upper, y_test, mae, mse

        feats = data[FEATURES].to_numpy(dtype=float)
        needed = LOOKBACK + n_future
        n_origins = len(feats) - needed  # nombre d'origines de calibration possibles

        # Pas assez d'historique pour calibrer -> repli heuristique.
        if n_origins < 5:
            y_pred, y_lower, y_upper = _heuristic_intervals(y_pred)
            return y_pred, y_lower, y_upper, y_test, mae, mse

        model = get_forecast_model()

        # Résidus par (fenêtre, horizon, variable) sur les dernières origines.
        start = max(0, n_origins - n_calib)
        residuals = []
        for i in range(start, n_origins):
            x_cal = feats[i:i + LOOKBACK].tolist()
            pred_cal, _ = model.predict_multistep(x=x_cal, n_futur=n_future)
            pred_cal = np.asarray(pred_cal)
            actual = feats[i + LOOKBACK:i + LOOKBACK + n_future]
            residuals.append(np.abs(actual - pred_cal))

        residuals = np.asarray(residuals)              # (K, n_future, n_features)
        half_width = np.quantile(residuals, 1 - alpha, axis=0)  # (n_future, n_features)

        y_lower = np.minimum(_clip_non_negative(y_pred - half_width), y_pred)
        y_upper = y_pred + half_width

        return y_pred, y_lower, y_upper, y_test, mae, mse

    except ForecastingException:
        raise

    except Exception as e:
        raise ForecastingException(e, sys)