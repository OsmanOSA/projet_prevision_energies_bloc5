"""Features pour la prévision directe multi-horizon résiduelle.

Toutes les features sont ancrées sur l'origine t (l'instant de la dernière
observation réelle) : lags, rolling stats, deltas/trends/diffs de la cible
elle-même, plus les mêmes dérivées pour `temp` (exogène commune) et un
calendaire cyclique. `add_target_features` calcule en plus, par horizon h, le
calendaire de l'heure **cible** (t+h) — toujours connu à l'avance, donc sans
risque de contamination autorégressive : chaque horizon est prédit par un
modèle dédié à partir de ces seules features d'origine (cf. model_trainer.py).

Méthodologie validée par l'utilisateur en notebook (notebooks/baseline_model.ipynb) :
+35,6 % de MAE vs persistance sur production_total, moyenné sur 24h.
"""

import numpy as np
import pandas as pd

HORIZON_MAX = 24

# Recherche de la correction résiduelle (alpha) et du mélange avec la
# persistance saisonnière, par validation (cf. model_trainer.py).
ALPHA_GRID = np.arange(0.70, 1.401, 0.01)
SEASONAL_WEIGHT_GRID = np.arange(0.00, 0.501, 0.05)

_PRODUCTION_LAGS = list(range(1, 25)) + [25, 48, 72, 168, 336]
_TEMPERATURE_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
_ROLLING_WINDOWS = [3, 6, 12, 24, 48, 168]
_TREND_LAGS = [3, 6, 12, 24, 48, 168]
_TEMP_TREND_LAGS = [3, 6, 12, 24, 168]
_DISTANCE_WINDOWS = [6, 24, 48, 168]

TEMPERATURE_REFERENCE = 18.0


def build_origin_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Lags/rolling/deltas/trends/diffs de `series`, ancrés sur l'origine.

    `prefix` distingue la cible (`production_total` -> "prod",
    `consommation_totale` -> "conso") pour appliquer le même vocabulaire aux
    deux cibles sans collision de colonnes.
    """
    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_0"] = series

    for lag in _PRODUCTION_LAGS:
        out[f"{prefix}_lag_{lag}"] = series.shift(lag)

    for window in _ROLLING_WINDOWS:
        rolling = series.rolling(window=window, min_periods=window)
        out[f"{prefix}_mean_{window}"] = rolling.mean()
        out[f"{prefix}_std_{window}"] = rolling.std()
        out[f"{prefix}_min_{window}"] = rolling.min()
        out[f"{prefix}_max_{window}"] = rolling.max()
        out[f"{prefix}_median_{window}"] = rolling.median()

    out[f"{prefix}_delta_1"] = out[f"{prefix}_0"] - out[f"{prefix}_lag_1"]
    out[f"{prefix}_delta_2"] = out[f"{prefix}_lag_1"] - out[f"{prefix}_lag_2"]
    out[f"{prefix}_delta_3"] = out[f"{prefix}_lag_2"] - out[f"{prefix}_lag_3"]
    out[f"{prefix}_acceleration"] = out[f"{prefix}_delta_1"] - out[f"{prefix}_delta_2"]
    out[f"{prefix}_acceleration_2"] = out[f"{prefix}_delta_2"] - out[f"{prefix}_delta_3"]

    for lag in _TREND_LAGS:
        out[f"{prefix}_trend_{lag}"] = (out[f"{prefix}_0"] - out[f"{prefix}_lag_{lag}"]) / lag

    out[f"{prefix}_diff_0_24"] = out[f"{prefix}_0"] - out[f"{prefix}_lag_24"]
    out[f"{prefix}_diff_0_168"] = out[f"{prefix}_0"] - out[f"{prefix}_lag_168"]
    out[f"{prefix}_diff_24_48"] = out[f"{prefix}_lag_24"] - out[f"{prefix}_lag_48"]
    out[f"{prefix}_diff_168_336"] = out[f"{prefix}_lag_168"] - out[f"{prefix}_lag_336"]

    for window in _DISTANCE_WINDOWS:
        out[f"{prefix}_distance_mean_{window}"] = out[f"{prefix}_0"] - out[f"{prefix}_mean_{window}"]
        out[f"{prefix}_range_{window}"] = out[f"{prefix}_max_{window}"] - out[f"{prefix}_min_{window}"]

    out[f"{prefix}_relative_delta_1"] = out[f"{prefix}_delta_1"] / (out[f"{prefix}_lag_1"].abs() + 1)
    out[f"{prefix}_relative_trend_6"] = out[f"{prefix}_trend_6"] / (out[f"{prefix}_lag_6"].abs() + 1)
    out[f"{prefix}_relative_diff_24"] = out[f"{prefix}_diff_0_24"] / (out[f"{prefix}_lag_24"].abs() + 1)

    return out


def build_temperature_features(temp: pd.Series, origin_hour_sin: pd.Series,
                               origin_hour_cos: pd.Series, origin_is_weekend: pd.Series) -> pd.DataFrame:
    """Mêmes dérivées que `build_origin_features`, préfixe fixe "temp"
    (exogène partagée par les deux cibles), + effets non linéaires."""
    out = pd.DataFrame(index=temp.index)
    out["temp_0"] = temp

    for lag in _TEMPERATURE_LAGS:
        out[f"temp_lag_{lag}"] = temp.shift(lag)

    for window in _ROLLING_WINDOWS:
        rolling = temp.rolling(window=window, min_periods=window)
        out[f"temp_mean_{window}"] = rolling.mean()
        out[f"temp_std_{window}"] = rolling.std()
        out[f"temp_min_{window}"] = rolling.min()
        out[f"temp_max_{window}"] = rolling.max()

    out["temp_delta_1"] = out["temp_0"] - out["temp_lag_1"]
    out["temp_acceleration"] = (out["temp_0"] - out["temp_lag_1"]) - (out["temp_lag_1"] - out["temp_lag_2"])

    for lag in _TEMP_TREND_LAGS:
        out[f"temp_trend_{lag}"] = (out["temp_0"] - out[f"temp_lag_{lag}"]) / lag

    out["temp_diff_24"] = out["temp_0"] - out["temp_lag_24"]
    out["temp_diff_168"] = out["temp_0"] - out["temp_lag_168"]

    out["heating_degree"] = (TEMPERATURE_REFERENCE - out["temp_0"]).clip(lower=0)
    out["cooling_degree"] = (out["temp_0"] - TEMPERATURE_REFERENCE).clip(lower=0)
    out["temp_squared"] = out["temp_0"] ** 2
    out["temp_x_origin_hour_sin"] = out["temp_0"] * origin_hour_sin
    out["temp_x_origin_hour_cos"] = out["temp_0"] * origin_hour_cos
    out["temp_x_weekend"] = out["temp_0"] * origin_is_weekend

    return out


def build_wind_weather_features(wspd: pd.Series, wdir: pd.Series, cldc: pd.Series) -> pd.DataFrame:
    """Signal météo dédié à l'éolien (WIND_ONSHORE uniquement, cf.
    local_forecaster.WIND_EXTRA_TARGET) : la production éolienne suit une loi
    quasi cubique de la vitesse du vent (puissance ∝ v³), un signal physique
    direct qu'aucune feature calendaire ne peut reconstruire. WDIR encodé en
    cyclique (direction = angle, pas une grandeur linéaire) ; WPGT (rafales)
    non disponible sur notre station -- absent volontairement.
    """
    out = pd.DataFrame(index=wspd.index)
    out["wspd_0"] = wspd

    for lag in _TEMPERATURE_LAGS:  # même échelle de lags que temp (courts, jusqu'à 168h)
        out[f"wspd_lag_{lag}"] = wspd.shift(lag)

    for window in _ROLLING_WINDOWS:
        rolling = wspd.rolling(window=window, min_periods=window)
        out[f"wspd_mean_{window}"] = rolling.mean()
        out[f"wspd_std_{window}"] = rolling.std()
        out[f"wspd_min_{window}"] = rolling.min()
        out[f"wspd_max_{window}"] = rolling.max()

    out["wspd_delta_1"] = out["wspd_0"] - out["wspd_lag_1"]
    for lag in _TEMP_TREND_LAGS:
        out[f"wspd_trend_{lag}"] = (out["wspd_0"] - out[f"wspd_lag_{lag}"]) / lag

    # Loi physique puissance-vitesse (cubique), + version amortie (carré) --
    # laisse au modèle le choix de la meilleure combinaison plutôt que
    # d'imposer une seule forme fonctionnelle.
    out["wspd_squared"] = out["wspd_0"] ** 2
    out["wspd_cubed"] = out["wspd_0"] ** 3
    out["wspd_lag24_cubed"] = out["wspd_lag_24"] ** 3

    wdir_rad = np.deg2rad(wdir)
    out["wdir_sin"] = np.sin(wdir_rad)
    out["wdir_cos"] = np.cos(wdir_rad)

    out["cldc_0"] = cldc
    out["cldc_mean_24"] = cldc.rolling(window=24, min_periods=24).mean()

    return out


def build_origin_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendaire cyclique + régimes horaires, à l'heure d'origine t."""
    out = pd.DataFrame(index=index)
    hour = index.hour

    out["origin_hour"] = hour
    out["origin_dayofweek"] = index.dayofweek
    out["origin_month"] = index.month
    out["origin_dayofyear"] = index.dayofyear
    out["origin_hour_of_week"] = index.dayofweek * 24 + hour

    out["origin_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["origin_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["origin_dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    out["origin_dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    out["origin_year_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
    out["origin_year_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)

    out["origin_is_weekend"] = (index.dayofweek >= 5).astype(int)
    out["origin_night"] = ((hour <= 6) | (hour >= 22)).astype(int)
    out["origin_morning_ramp"] = ((hour >= 7) & (hour <= 11)).astype(int)
    out["origin_midday_peak"] = ((hour >= 12) & (hour <= 16)).astype(int)
    out["origin_evening_ramp"] = ((hour >= 17) & (hour <= 21)).astype(int)

    return out


def add_target_features(X: pd.DataFrame, horizon: int, delta_col: str) -> pd.DataFrame:
    """Calendaire + interactions à l'heure **cible** (t+h) — toujours connu à
    l'avance quel que soit h, donc sans risque autorégressif.

    `delta_col` : colonne de variation à 1 pas de la cible courante
    (`prod_delta_1` ou `conso_delta_1`), utilisée pour les interactions avec
    les régimes horaires cibles.
    """
    out = X.copy()
    target_index = out.index + pd.to_timedelta(horizon, unit="h")

    target_hour = target_index.hour
    target_dow = target_index.dayofweek
    target_dayofyear = target_index.dayofyear

    out["target_hour"] = target_hour
    out["target_dayofweek"] = target_dow
    out["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24)
    out["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24)
    out["target_dow_sin"] = np.sin(2 * np.pi * target_dow / 7)
    out["target_dow_cos"] = np.cos(2 * np.pi * target_dow / 7)
    out["target_year_sin"] = np.sin(2 * np.pi * target_dayofyear / 365.25)
    out["target_year_cos"] = np.cos(2 * np.pi * target_dayofyear / 365.25)
    out["target_is_weekend"] = (target_dow >= 5).astype(int)
    out["target_hour_of_week"] = target_dow * 24 + target_hour
    out["target_night"] = ((target_hour <= 6) | (target_hour >= 22)).astype(int)
    out["target_morning_ramp"] = ((target_hour >= 7) & (target_hour <= 11)).astype(int)
    out["target_midday_peak"] = ((target_hour >= 12) & (target_hour <= 16)).astype(int)
    out["target_evening_ramp"] = ((target_hour >= 17) & (target_hour <= 21)).astype(int)

    out["temp_x_target_hour_sin"] = out["temp_0"] * out["target_hour_sin"]
    out["temp_x_target_hour_cos"] = out["temp_0"] * out["target_hour_cos"]
    out["delta_x_morning_ramp"] = out[delta_col] * out["target_morning_ramp"]
    out["delta_x_evening_ramp"] = out[delta_col] * out["target_evening_ramp"]

    return out


def seasonal_baseline(frame: pd.DataFrame, horizon: int, prefix: str) -> np.ndarray:
    """Persistance saisonnière : valeur exactement 24h avant l'heure cible.

    `lag = 24 - horizon` place toujours la référence à 24h de la cible
    (t+horizon), qu'elle soit avant (lag>0) ou à l'origine même (lag=0,
    horizon=24 -> "même heure demain" = valeur d'aujourd'hui).
    """
    lag = 24 - horizon
    if lag == 0:
        return frame[f"{prefix}_0"].to_numpy()
    return frame[f"{prefix}_lag_{lag}"].to_numpy()


# Cibles directement modélisées : les 4 sources de production (plutôt que
# l'agrégat production_total, qui noyait l'éolien -- seule composante vraiment
# volatile -- sous des composantes stables/calendaires) + la consommation.
# production_total redevient une valeur dérivée (somme des 4), pas une cible
# entraînée (cf. scripts/forecast.py).
TARGET_PREFIXES = {
    "SOLAR": "solar",
    "BIOMASS": "biomass",
    "WIND_ONSHORE": "wind",
    "NUCLEAR": "nuclear",
    "consommation_totale": "conso",
}


def target_prefix(target_column: str) -> str:
    return TARGET_PREFIXES[target_column]


def build_series_by_target(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Extrait les séries numériques des cibles directement modélisées, par
    nom de colonne -- mêmes noms dans le CSV d'entraînement et dans
    `observations` (live), donc réutilisable dans les deux contextes."""
    return {name: pd.to_numeric(df[name], errors="coerce") for name in TARGET_PREFIXES}


def build_origin_feature_frame(series_by_target: dict[str, pd.Series], temp: pd.Series,
                               target_column: str) -> tuple[pd.DataFrame, str]:
    """Features d'origine pour une cible donnée (sans les colonnes cible —
    utilisé tel quel pour l'inférence ; `build_features_for_target` y ajoute
    les 24 colonnes `target_h*` pour l'entraînement)."""
    prefix = target_prefix(target_column)
    series = series_by_target[target_column]

    origin = build_origin_features(series, prefix)
    calendar = build_origin_calendar(series.index)
    temperature = build_temperature_features(
        temp, calendar["origin_hour_sin"], calendar["origin_hour_cos"], calendar["origin_is_weekend"]
    )

    features_df = pd.concat([origin, temperature, calendar], axis=1)
    return features_df, prefix


def build_features_for_target(series_by_target: dict[str, pd.Series], temp: pd.Series,
                              target_column: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Assemble les features d'origine pour une cible donnée + les 24 colonnes cible.

    Retourne (features_df, prefix, target_columns). `features_df` contient
    les colonnes features ET les colonnes `target_h1..target_h24`, avant
    dropna (fait par l'appelant, qui connaît aussi les colonnes de features).
    """
    series = series_by_target[target_column]

    features_df, prefix = build_origin_feature_frame(series_by_target, temp, target_column)

    target_columns = []
    for horizon in range(1, HORIZON_MAX + 1):
        target_name = f"target_h{horizon}"
        features_df[target_name] = series.shift(-horizon)
        target_columns.append(target_name)

    return features_df, prefix, target_columns
