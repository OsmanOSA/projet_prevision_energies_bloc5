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

from pipeline_prevision.logging.logger import logging

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


# --- Calendrier français : heure locale et jours fériés ----------------------
# L'index du pipeline est en UTC (vérifié par la physique : pic solaire à 11 h,
# pointe de consommation hivernale à 18 h UTC = 19 h locale). Les features
# calendaires historiques ci-dessous sont donc construites sur l'heure UTC, ce
# qui décale d'une heure tout le profil journalier entre l'hiver et l'été : le
# modèle doit apprendre deux profils au lieu d'un.
#
# Les fonctions qui suivent ajoutent le calendrier en HEURE LOCALE et les jours
# fériés français. Mesuré sur backtest à origine glissante (6 folds, cf.
# notebooks/comparison_time_series_models.ipynb) : **-9,7 % de MAE sur la
# consommation, à modèle et hyperparamètres strictement identiques**.
#
# Ces colonnes sont AJOUTÉES, jamais substituées : les modèles déjà entraînés
# référencent leurs colonnes par nom dans `metadata.json` et continuent de
# fonctionner à l'identique.

FUSEAU = "Europe/Paris"


def easter_sunday(year: int) -> pd.Timestamp:
    """Dimanche de Pâques — algorithme grégorien anonyme (Butcher / Meeus)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    n = h + ell - 7 * m + 114
    return pd.Timestamp(year, n // 31, (n % 31) + 1)


def french_holidays(years) -> set[pd.Timestamp]:
    """Les 11 jours fériés français, calculés (aucune dépendance externe).

    Un férié tombant en semaine retire ~12 % de consommation en moyenne (jusqu'à
    -28 % le 1er janvier), soit ≈ 6 GW — trois fois l'écart-type du résidu d'une
    décomposition saisonnière. C'est le signal calendaire le plus rentable.
    """
    out = set()
    for year in years:
        easter = easter_sunday(year)
        out |= {
            pd.Timestamp(year, 1, 1), pd.Timestamp(year, 5, 1),
            pd.Timestamp(year, 5, 8), pd.Timestamp(year, 7, 14),
            pd.Timestamp(year, 8, 15), pd.Timestamp(year, 11, 1),
            pd.Timestamp(year, 11, 11), pd.Timestamp(year, 12, 25),
            easter + pd.Timedelta(days=1),    # lundi de Pâques
            easter + pd.Timedelta(days=39),   # Ascension
            easter + pd.Timedelta(days=50),   # lundi de Pentecôte
        }
    return out


def _local_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convertit un index UTC naïf en heure locale française, naïf lui aussi."""
    return index.tz_localize("UTC").tz_convert(FUSEAU).tz_localize(None)


def build_french_calendar(index: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    """Calendrier en heure locale + fériés/ponts, préfixé pour éviter toute
    collision avec les colonnes calendaires UTC historiques.

    `prefix` vaut "origin" (ancré sur t) ou "target" (ancré sur t+h) ; dans les
    deux cas les variables sont déterministes et connues à l'avance, donc
    utilisables quel que soit l'horizon.
    """
    local = _local_index(index)
    dates = local.normalize()
    holidays = french_holidays(range(local.year.min() - 1, local.year.max() + 2))

    out = pd.DataFrame(index=index)
    hour, dow = local.hour, local.dayofweek

    out[f"{prefix}_loc_hour"] = hour
    out[f"{prefix}_loc_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out[f"{prefix}_loc_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # Harmonique 12 h : la double pointe midi/soir est le 3e pic du périodogramme.
    out[f"{prefix}_loc_semi_sin"] = np.sin(4 * np.pi * hour / 24)
    out[f"{prefix}_loc_semi_cos"] = np.cos(4 * np.pi * hour / 24)
    out[f"{prefix}_loc_hour_of_week"] = dow * 24 + hour
    out[f"{prefix}_loc_is_weekend"] = (dow >= 5).astype(int)

    is_holiday = pd.Series(dates.isin(holidays), index=index)
    out[f"{prefix}_holiday"] = is_holiday.astype(int)
    out[f"{prefix}_holiday_eve"] = pd.Series(
        (dates + pd.Timedelta(days=1)).isin(holidays), index=index).astype(int)
    out[f"{prefix}_holiday_next"] = pd.Series(
        (dates - pd.Timedelta(days=1)).isin(holidays), index=index).astype(int)
    # Pont : jour ouvré isolé entre un férié et un week-end.
    bridge = (((dow == 4) & (dates - pd.Timedelta(days=1)).isin(holidays))
              | ((dow == 0) & (dates + pd.Timedelta(days=1)).isin(holidays)))
    out[f"{prefix}_bridge"] = (pd.Series(bridge, index=index).astype(int)
                               * (1 - out[f"{prefix}_holiday"]))
    # Creux structurels de consommation, non couverts par les fériés.
    out[f"{prefix}_august"] = (local.month == 8).astype(int)
    out[f"{prefix}_xmas_break"] = ((local.month == 12) & (local.day >= 20)).astype(int)
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


def _add_forecast_temperature_features(out: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Température PRÉVUE à l'heure cible, pour l'horizon courant.

    C'est l'information qui manquait entièrement au modèle : toutes ses features
    thermiques étaient gelées à l'origine, si bien qu'il extrapolait le jour cible
    avec la météo de la veille au soir. Par météo stable cet handicap ne coûte
    rien et les features autorégressives dominent (+16 % de MAE contre RTE) ; dès
    que la température bascule, on extrapole à contresens (-15 %). Le signe du
    biais suivait celui de Δtemp : réchauffement -> sous-estimation, refroidissement
    -> surestimation.

    On retient la colonne de CET horizon parmi les 24 portées par le cadre
    d'origine, et on écarte les autres : sans ce filtrage, chacun des 24 modèles
    verrait 23 prévisions qui ne le concernent pas.

    Si les colonnes sont absentes (jeu de données antérieur au backfill, ou
    couverture insuffisante — cf. `select_forecast_temperature`), la fonction est
    un no-op : le cadre reste celui d'avant, et les modèles archivés se rejouent
    à l'identique.
    """
    colonnes = [c for c in out.columns if c.startswith(FORECAST_PREFIX)]
    if not colonnes:
        return out

    courante = f"{FORECAST_PREFIX}{horizon}"
    if courante not in out.columns:
        raise KeyError(
            f"`{courante}` manquante alors que d'autres colonnes `{FORECAST_PREFIX}*` "
            f"sont présentes ({len(colonnes)}) : cadre d'origine incohérent. "
            "Reconstruire les features au lieu de prédire sur un cadre partiel.")

    prevu = out[courante]
    out = out.drop(columns=colonnes)

    out["temp_prev_target"] = prevu
    # LE signal manquant : l'écart entre la température prévue à l'heure cible et
    # celle observée à l'origine. Les deux termes viennent de la même grille
    # Open-Meteo (cf. TEMPERATURE_PREFERENCES) : croiser les sources y injecterait
    # un biais non stationnaire de 0,63 à 0,78 °C à la place du signal.
    out["temp_prev_delta"] = prevu - out["temp_0"]
    # Thermosensibilité à l'heure cible. Les degrés-jours existaient déjà à
    # l'origine ; c'est leur valeur à l'heure CIBLE qui pilote la consommation de
    # cette heure-là, et le non-linéaire est ce qui rend la canicule coûteuse.
    out["heating_degree_target"] = (TEMPERATURE_REFERENCE - prevu).clip(lower=0)
    out["cooling_degree_target"] = (prevu - TEMPERATURE_REFERENCE).clip(lower=0)
    return out


def _add_daytype_anchor_features(out: pd.DataFrame, horizon: int,
                                 prefix: str) -> pd.DataFrame:
    """Retient la valeur du dernier jour comparable pour CET horizon, écarte les 23
    autres, et en dérive l'écart au niveau d'origine.

    Les 24 colonnes voyagent dans le cadre d'origine parce que `anchor_values` en
    a besoin AVANT cette fonction (elle lit `feature_row`, non transformé). Mais
    le modèle, lui, ne doit en voir qu'une : les 23 autres décrivent des heures
    qui ne le concernent pas.

    No-op si les colonnes sont absentes — un modèle antérieur à cette ancre se
    rejoue alors à l'identique.
    """
    colonnes = [c for c in out.columns if DAYTYPE_SUFFIX in c]
    if not colonnes:
        return out

    courante = f"{prefix}{DAYTYPE_SUFFIX}{horizon}"
    if courante not in out.columns:
        raise KeyError(
            f"`{courante}` manquante alors que {len(colonnes)} colonnes "
            f"`{DAYTYPE_SUFFIX}` sont présentes : cadre d'origine incohérent.")

    reference = out[courante]
    out = out.drop(columns=colonnes)
    out["daytype_ref"] = reference
    # Écart entre le dernier jour comparable et le niveau courant : dit au modèle
    # de combien la référence est décalée, ce qu'aucune des deux valeurs seules
    # n'exprime.
    out["daytype_vs_origin"] = reference - out[f"{prefix}_0"]
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

    out = _add_forecast_temperature_features(out, horizon)
    out = _add_daytype_anchor_features(out, horizon, delta_col.rsplit("_delta_1", 1)[0])

    # Calendrier français à l'heure CIBLE : c'est celui qui décide, puisque
    # chaque horizon a son propre modèle. `target_hour` ci-dessus est en UTC et
    # se décale d'une heure entre l'hiver et l'été ; `target_loc_hour` ne bouge
    # pas. Le jour férié de l'heure cible est également ce qui compte, pas celui
    # de l'origine.
    out = pd.concat([out, build_french_calendar(target_index, "target")
                     .set_index(out.index)], axis=1)
    out["temp_x_target_holiday"] = out["temp_0"] * out["target_holiday"]

    return out


# --- Ancre « dernier jour comparable » --------------------------------------
# Suffixe des 24 colonnes portées par le cadre d'origine, une par horizon.
DAYTYPE_SUFFIX = "_daytype_h"

# Typage des jours, convention usuelle en prévision de charge française.
JOUR_OUVRE, JOUR_SAMEDI, JOUR_CHOME = 0, 1, 2


def day_types(index: pd.DatetimeIndex) -> np.ndarray:
    """0 = ouvré, 1 = samedi, 2 = dimanche ou férié (en heure LOCALE).

    Le calendrier qui gouverne la consommation est local, pas UTC : un jour
    férié commence à minuit à Paris.
    """
    local = _local_index(index)
    feries = french_holidays(range(local.year.min() - 1, local.year.max() + 2))
    dow = local.dayofweek
    types = np.where(dow == 5, JOUR_SAMEDI, np.where(dow == 6, JOUR_CHOME, JOUR_OUVRE))
    return np.where(local.normalize().isin(feries), JOUR_CHOME, types)


def build_daytype_anchor_columns(series: pd.Series, prefix: str) -> pd.DataFrame:
    """24 colonnes : valeur à la même heure du dernier jour COMPARABLE à la cible.

    Pourquoi cette ancre
    --------------------
    L'ancre `seasonal_24` lit « la même heure hier ». Un lundi, « hier » est un
    dimanche : la base est fausse de **5 766 MW** en moyenne sur 3,5 ans, là où le
    dernier jour ouvré donne 2 630 MW (-54 %). Comme le modèle apprend un RÉSIDU
    par rapport à l'ancre, il part d'un trou qu'il ne comble pas — d'où
    l'effondrement mesuré face à RTE les lundis, les jours fériés (-37 %) et les
    lendemains de férié (-34 %).

    Règle de comparabilité, arbitrée sur mesure (MAE de l'ancre, 30 600 points) :

    | cible          | candidats retenus      | ancre « hier » | cette ancre |
    |----------------|------------------------|----------------|-------------|
    | ouvré          | derniers jours ouvrés  | 1 529 / 5 766* | 1 454 / 2 630* |
    | samedi         | derniers samedis       | 4 192          | 3 462       |
    | dimanche/férié | derniers jours chômés  | 2 104 / 3 399* | 2 104 / 2 402* |

    (*) second chiffre = lundi, resp. jour férié.

    Le dimanche fait exception et c'est délibéré : y viser « le dimanche
    précédent » serait PIRE que « hier » (3 349 contre 2 104), le samedi étant
    plus proche en météo et en saison. On retient donc le dernier jour **chômé**,
    qui pour un dimanche est le samedi de la veille.

    Causalité
    ---------
    Seuls les horodatages **antérieurs ou égaux à l'origine** sont candidats
    (`searchsorted` borné par l'origine). Une ancre qui lirait au-delà ferait
    fuiter l'avenir dans la cible résiduelle elle-même — la fuite la plus
    difficile à détecter, puisqu'elle n'apparaît dans aucune feature.
    """
    index = series.index
    types_index = day_types(index)
    heures_index = index.hour
    valeurs = series.to_numpy()

    # Positions candidates par (heure, type) et par (heure, chômé) — pré-groupées
    # une fois, puis interrogées par recherche dichotomique : une boucle par
    # origine coûterait ~30 000 x 24 recherches.
    par_type, par_chome = {}, {}
    for heure in range(24):
        meme_heure = heures_index == heure
        for t in (JOUR_OUVRE, JOUR_SAMEDI, JOUR_CHOME):
            par_type[(heure, t)] = np.flatnonzero(meme_heure & (types_index == t))
        par_chome[heure] = np.flatnonzero(meme_heure & (types_index != JOUR_OUVRE))

    colonnes = {}
    for horizon in range(1, HORIZON_MAX + 1):
        cibles = index + pd.Timedelta(hours=horizon)
        types_cible, heures_cible = day_types(cibles), cibles.hour
        sortie = np.full(len(index), np.nan)

        for heure in range(24):
            for t in (JOUR_OUVRE, JOUR_SAMEDI, JOUR_CHOME):
                lignes = np.flatnonzero((heures_cible == heure) & (types_cible == t))
                if lignes.size == 0:
                    continue
                # Un dimanche/férié se compare au dernier jour chômé (souvent la
                # veille) ; les autres types à leur propre type.
                candidats = par_chome[heure] if t == JOUR_CHOME else par_type[(heure, t)]
                if candidats.size == 0:
                    continue
                # `lignes` indexe des ORIGINES : le candidat doit être <= origine.
                pos = np.searchsorted(candidats, lignes, side="right") - 1
                valide = pos >= 0
                trouve = np.full(lignes.size, np.nan)
                trouve[valide] = valeurs[candidats[pos[valide]]]
                sortie[lignes] = trouve

        colonnes[f"{prefix}{DAYTYPE_SUFFIX}{horizon}"] = sortie

    return pd.DataFrame(colonnes, index=index)


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


# --- Ancres de la cible résiduelle ------------------------------------------
# Le modèle n'apprend jamais y(t+h) mais son écart à une ANCRE causale. Ce
# choix décide de ce que les arbres ont à apprendre : ancrer sur la
# persistance laisse tout le cycle diurne dans le résidu, ancrer sur la veille
# à la même heure n'y laisse que l'anomalie. C'est donc un hyperparamètre à
# part entière, sélectionné par cible ET par horizon sur la validation (cf.
# `model_trainer._train_horizons`) -- l'ancre gagnante varie fortement :
# `seasonal_24` l'emporte sur la consommation h1..h17 (+6,8 % de MAE en
# moyenne, jusqu'à +22 % à h2) et sur SOLAR h3..h23 (+1,6 %), tandis que la
# persistance reste meilleure sur NUCLEAR, WIND_ONSHORE et BIOMASS.
# `seasonal_daytype` s'ajoute aux deux précédentes plutôt que de remplacer
# `seasonal_24` : la sélection par validation tranche, cible par cible et horizon
# par horizon. Si la nouvelle ancre dégradait quoi que ce soit, elle serait
# simplement écartée — le dispositif existant rend l'ajout sans risque.
ANCHOR_NAMES = ("persistence", "seasonal_24", "seasonal_daytype")
DEFAULT_ANCHOR = "persistence"


def anchor_values(frame: pd.DataFrame, horizon: int, prefix: str, anchor: str) -> np.ndarray:
    """Valeurs de l'ancre `anchor` pour `horizon`. Toutes sont causales :
    `persistence` lit l'origine t, `seasonal_24` lit t+horizon-24 (antérieur ou
    égal à t pour horizon <= 24), `seasonal_daytype` lit le dernier jour
    comparable, borné à t par construction (cf. `build_daytype_anchor_columns`)."""
    if anchor == "persistence":
        return frame[f"{prefix}_0"].to_numpy()
    if anchor == "seasonal_24":
        return seasonal_baseline(frame, horizon, prefix)
    if anchor == "seasonal_daytype":
        colonne = f"{prefix}{DAYTYPE_SUFFIX}{horizon}"
        if colonne not in frame.columns:
            raise KeyError(
                f"`{colonne}` absente : le cadre d'origine a été construit sans les "
                "colonnes d'ancre par type de jour. Reconstruire les features "
                "plutôt que de rabattre l'ancre en silence sur une autre.")
        return frame[colonne].to_numpy()
    raise ValueError(f"Ancre inconnue : {anchor} (attendu : {ANCHOR_NAMES})")


# Ancre de mélange associée à chaque ancre principale (poids `seasonal_weight`).
# Avec trois candidates, la bascule binaire d'origine ne suffit plus : on
# explicite le couple. Les deux ancres saisonnières se mélangent à la
# persistance, qui est le signal le plus différent des deux — les mélanger entre
# elles combinerait deux références très corrélées, sans diversification.
_COMPLEMENTARY = {
    "persistence": "seasonal_24",
    "seasonal_24": "persistence",
    "seasonal_daytype": "persistence",
}


def complementary_anchor(anchor: str) -> str:
    """Ancre servant de terme de mélange (poids `seasonal_weight`).

    À horizon 24 `persistence` et `seasonal_24` coïncident — `seasonal_baseline`
    y retourne `{prefix}_0` — donc leur mélange dégénère : la prédiction se
    réduit à `ancre + (1-w)·alpha·résidu`, où w ne fait que rééchelonner alpha.
    Sans effet sur la qualité, laissé tel quel plutôt que traité en cas
    particulier (vérifié : gain strictement nul à h24 sur les 5 cibles).
    `seasonal_daytype` ne dégénère pas, elle : à h24 elle désigne le dernier jour
    comparable, qui n'est l'origine que si la veille était du même type.
    """
    return _COMPLEMENTARY[anchor]


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


TEMPERATURE_FALLBACK = "temp"
# Préférence décroissante pour la température exogène d'ORIGINE. `temp_fr_om`
# passe devant `temp_fr` non pas parce qu'elle serait plus exacte (elle suit
# `temp_fr` à +0,15 °C de biais et 0,50 °C de RMSE, corrélation 0,997) mais parce
# qu'elle vient de la MÊME grille Open-Meteo que la température prévue à l'heure
# cible. Le biais grille/station varie de 0,78 °C selon l'heure et de 0,63 °C
# selon le niveau de température : franchir la frontière de source dans l'écart
# `temp_prev_target - temp_0` y injecterait ce décalage à la place du signal.
TEMPERATURE_PREFERENCES = ("temp_fr_om", "temp_fr")
TEMPERATURE_COLUMN = TEMPERATURE_PREFERENCES[-1]   # compat. rétro (imports existants)
# Couverture minimale d'une colonne candidate, relativement à celle de `temp`,
# en dessous de laquelle on considère la migration incomplète et on se replie.
MIN_TEMPERATURE_COVERAGE = 0.95

# Température PRÉVUE à l'heure cible. Colonne de la table `observations`
# (vintage J-1 figé, cf. models.py) et préfixe des 24 colonnes portées par le
# cadre d'origine, `temp_prev_h1..h24`.
FORECAST_TEMPERATURE_COLUMN = "temp_fr_prev"
FORECAST_PREFIX = "temp_prev_h"


def select_temperature(df: pd.DataFrame) -> pd.Series:
    """Série de température exogène d'origine : première colonne de
    `TEMPERATURE_PREFERENCES` suffisamment couverte, sinon `temp`.

    `temp_fr` est la moyenne pondérée par la population régionale de 17 stations
    couvrant les 12 régions métropolitaines (cf. `temperature_france.py`), là où
    `temp` n'est qu'une station (Paris). Mesuré par backtesting à origine
    glissante : **-2,1 % de MAE sur la consommation, gain tenu sur 5 folds
    sur 6** (`python -m scripts.evaluate_features`).

    `temp_fr_om` est le même indice pondéré sur la grille Open-Meteo. Elle est
    préférée pour la cohérence de source avec la prévision (cf.
    `TEMPERATURE_PREFERENCES`), pas pour une meilleure exactitude.

    Le repli en cascade garde lisibles les jeux de données antérieurs à chaque
    migration (archives, sauvegardes de `data.csv`), sans lesquels toute
    relecture d'historique échouerait.
    """
    repli = pd.to_numeric(df[TEMPERATURE_FALLBACK], errors="coerce")
    attendu = max(repli.notna().sum(), 1)

    for colonne in TEMPERATURE_PREFERENCES:
        if colonne not in df.columns:
            continue
        serie = pd.to_numeric(df[colonne], errors="coerce")
        # Une colonne PRÉSENTE MAIS QUASI VIDE est le cas dangereux : il survient
        # pendant une migration (colonne ajoutée en base, historique pas encore
        # rétro-alimenté). Tester `notna().sum() == 0` ne suffit pas -- il a
        # laissé passer une base à 70 valeurs sur 31 341, ce qui aurait vidé
        # toutes les features de température (lags et rolling jusqu'à 168 h) sans
        # qu'aucune erreur ne se déclenche. On exige donc une couverture
        # comparable à celle du repli, et on trace la bascule : un repli
        # silencieux ferait diverger l'entraînement (sur CSV, complet) de
        # l'évaluation (sur base, incomplète).
        if serie.notna().sum() >= MIN_TEMPERATURE_COVERAGE * attendu:
            return serie
        logging.warning(
            "`%s` couvre %d lignes sur %d attendues : passage au repli suivant. "
            "Rétro-alimenter avec `python -m scripts.backfill_prevision_temperature` "
            "(temp_fr_om/temp_fr_prev) ou `python -m scripts.backfill_temperature_france` "
            "(temp_fr).", colonne, serie.notna().sum(), attendu)

    logging.warning("Aucune température pondérée exploitable : repli sur `%s` "
                    "(station unique).", TEMPERATURE_FALLBACK)
    return repli


def select_forecast_temperature(df: pd.DataFrame) -> pd.Series | None:
    """Série de température PRÉVUE à l'heure cible, ou None si indisponible.

    Retourner None plutôt qu'une série vide est délibéré : les features de
    température cible sont alors simplement absentes du cadre, et les modèles
    entraînés sans elles continuent de fonctionner à l'identique. Une série de
    NaN, à l'inverse, ferait tomber toutes les lignes au `dropna` en aval — le
    pipeline « marcherait » en n'apprenant plus rien.
    """
    if FORECAST_TEMPERATURE_COLUMN not in df.columns:
        return None

    serie = pd.to_numeric(df[FORECAST_TEMPERATURE_COLUMN], errors="coerce")
    attendu = max(pd.to_numeric(df[TEMPERATURE_FALLBACK], errors="coerce").notna().sum(), 1)
    if serie.notna().sum() < MIN_TEMPERATURE_COVERAGE * attendu:
        logging.warning(
            "`%s` couvre %d lignes sur %d attendues : features de température "
            "cible désactivées (le modèle reste aveugle à la météo future). "
            "Rétro-alimenter avec `python -m scripts.backfill_prevision_temperature`.",
            FORECAST_TEMPERATURE_COLUMN, serie.notna().sum(), attendu)
        return None
    return serie


def build_series_by_target(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Extrait les séries numériques des cibles directement modélisées, par
    nom de colonne -- mêmes noms dans le CSV d'entraînement et dans
    `observations` (live), donc réutilisable dans les deux contextes."""
    return {name: pd.to_numeric(df[name], errors="coerce") for name in TARGET_PREFIXES}


def build_forecast_temperature_columns(temp_prev: pd.Series,
                                       index: pd.DatetimeIndex) -> pd.DataFrame:
    """Les 24 colonnes `temp_prev_h1..h24` : température prévue à t+h, indexée
    sur l'origine t.

    Elles voyagent dans le cadre d'origine (donc dans `feature_columns`) plutôt
    que d'être passées en argument à `add_target_features` : c'est ce qui évite
    de modifier la signature de cette fonction, appelée depuis six endroits
    (entraînement, inférence, évaluation). `add_target_features` retient ensuite
    la colonne de SON horizon et écarte les 23 autres, de sorte qu'aucun modèle
    ne voit la prévision d'un horizon qui n'est pas le sien.
    """
    # Lecture directe à `index + h`, et NON un `reindex(index).shift(-h)` : ce
    # dernier confine la série à l'index des observations et perd donc toute
    # valeur postérieure à la dernière observation -- exactement celles dont
    # l'inférence a besoin, puisqu'à l'origine t elle prédit t+1..t+24.
    colonnes = {}
    for h in range(1, HORIZON_MAX + 1):
        valeurs = temp_prev.reindex(index + pd.Timedelta(hours=h)).to_numpy()
        colonnes[f"{FORECAST_PREFIX}{h}"] = pd.Series(valeurs, index=index)
    return pd.DataFrame(colonnes, index=index)


def build_origin_feature_frame(series_by_target: dict[str, pd.Series], temp: pd.Series,
                               target_column: str,
                               temp_prev: pd.Series | None = None
                               ) -> tuple[pd.DataFrame, str]:
    """Features d'origine pour une cible donnée (sans les colonnes cible —
    utilisé tel quel pour l'inférence ; `build_features_for_target` y ajoute
    les 24 colonnes `target_h*` pour l'entraînement).

    `temp_prev` (température prévue, cf. `select_forecast_temperature`) est
    facultative : à None, les 24 colonnes `temp_prev_h*` sont absentes et le
    cadre est exactement celui d'avant leur introduction — les modèles archivés
    restent donc rejouables à l'identique.
    """
    prefix = target_prefix(target_column)
    series = series_by_target[target_column]

    origin = build_origin_features(series, prefix)
    calendar = build_origin_calendar(series.index)
    calendar_fr = build_french_calendar(series.index, "origin")
    temperature = build_temperature_features(
        temp, calendar["origin_hour_sin"], calendar["origin_hour_cos"], calendar["origin_is_weekend"]
    )

    morceaux = [origin, temperature, calendar, calendar_fr,
                # Support de l'ancre `seasonal_daytype`. Toujours construites :
                # l'ancre est choisie par validation et doit pouvoir être servie
                # quel que soit le modèle rechargé. Coût : 24 colonnes par cible.
                build_daytype_anchor_columns(series, prefix)]
    if temp_prev is not None:
        morceaux.append(build_forecast_temperature_columns(temp_prev, series.index))

    features_df = pd.concat(morceaux, axis=1)
    return features_df, prefix


def build_features_for_target(series_by_target: dict[str, pd.Series], temp: pd.Series,
                              target_column: str, temp_prev: pd.Series | None = None
                              ) -> tuple[pd.DataFrame, str, list[str]]:
    """Assemble les features d'origine pour une cible donnée + les 24 colonnes cible.

    Retourne (features_df, prefix, target_columns). `features_df` contient
    les colonnes features ET les colonnes `target_h1..target_h24`, avant
    dropna (fait par l'appelant, qui connaît aussi les colonnes de features).
    """
    series = series_by_target[target_column]

    features_df, prefix = build_origin_feature_frame(series_by_target, temp, target_column,
                                                     temp_prev=temp_prev)

    target_columns = []
    for horizon in range(1, HORIZON_MAX + 1):
        target_name = f"target_h{horizon}"
        features_df[target_name] = series.shift(-horizon)
        target_columns.append(target_name)

    return features_df, prefix, target_columns
