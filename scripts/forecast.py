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
    forecast_origin,
    get_model_version,
    derive_production_total,
    PRODUCTION_SOURCES,
    TARGETS,
)

# Écart maximal toléré entre la dernière observation et l'origine réellement
# utilisée par le modèle. Au-delà, on refuse d'écrire : une origine périmée
# donne une prévision qui porte sur des heures déjà passées (horizons
# négatifs), invisible dans le dashboard autrement que par une courbe
# inexplicablement décalée. Mieux vaut un DAG en échec, qui alerte, qu'une
# prévision fausse persistée en silence. Une ou deux heures de retard restent
# normales (publication Meteostat/RTE), d'où une tolérance non nulle.
MAX_ORIGIN_LAG = pd.Timedelta(hours=3)

# Report de la dernière valeur connue sur les toutes dernières heures, quand
# une source publie en retard (RTE sort la consommation avec ~1h30 de latence,
# contre <30 min pour la production et la température).
#
# Arbitré par mesure, pas par principe : sur 53 jours hors échantillon, en
# comparant l'erreur de prévision réelle sur les 23 h communes, le report bat
# l'alternative « reculer l'origine d'une heure » de 28,5 MW de MAE en moyenne
# (Wilcoxon apparié, p = 0,035). Reculer l'origine évite certes de dégrader
# `conso_delta_1`, mais oblige à prévoir chaque heure un cran plus loin, et ce
# surcoût d'horizon l'emporte. Moyenne ou médiane sur 2-3 h font nettement
# moins bien (la moyenne 3 h est significativement la pire, p = 0,025) : elles
# lissent à travers la marche « heures creuses » de 23 h.
#
# Bornée à 2 h : au-delà ce n'est plus un retard de publication mais une panne,
# que MAX_ORIGIN_LAG doit faire remonter. Et strictement EN MÉMOIRE — la valeur
# reportée ne retourne jamais dans `observations`, qui reste la vérité terrain
# de l'évaluation (cf. `evaluate_daily`).
REPORT_MAX_HOURS = 2


def _combler_fin_de_serie(df: pd.DataFrame, max_heures: int = REPORT_MAX_HOURS) -> tuple:
    """Reporte la dernière valeur connue sur les NaN de FIN de série.

    Uniquement en fin de série : les trous internes sont déjà comblés par
    interpolation à l'ingestion (cf. `concat_all_data`), et une interpolation
    y est meilleure qu'un report puisqu'elle connaît les deux bords.

    Retourne (dataframe comblé, {colonne: [horodatages comblés]}).
    """
    out = df.copy()
    comble = {}
    for colonne in out.columns:
        serie = out[colonne]
        dernier_valide = serie.last_valid_index()
        if dernier_valide is None or dernier_valide == serie.index[-1]:
            continue
        # Tout ce qui suit `last_valid_index` est NaN par définition.
        cibles = serie.loc[dernier_valide:].index[1:1 + max_heures]
        if len(cibles):
            out.loc[cibles, colonne] = serie.loc[dernier_valide]
            comble[colonne] = list(cibles)
    return out, comble


def origins_by_target(per_target: dict, expected_origin: pd.Timestamp) -> dict:
    """Origine réellement utilisée par chaque cible, validée contre l'origine
    attendue (dernière observation en prévision live, origine rejouée en
    backfill).

    Les cibles n'ont pas forcément la même origine : RTE publie la
    consommation avec un peu plus de latence que la production, donc la
    dernière heure aux features complètes peut différer d'une heure d'une
    cible à l'autre. C'est normal, et chaque cible garde son origine propre
    (persistée telle quelle) plutôt que d'être recalée de force sur une
    origine commune, ce qui décalerait ses horizons d'une heure.

    En revanche une origine nettement périmée n'est jamais normale : elle
    signale un trou dans une série exogène qui invalide les features récentes
    (cf. temp_lag_168) et produirait une prévision portant sur des heures déjà
    passées. On refuse alors d'écrire — un DAG en échec alerte, une prévision
    fausse persistée en silence non.
    """
    origins = {target: forecast_origin(result) for target, result in per_target.items()}

    perimees = {t: o for t, o in origins.items() if expected_origin - o > MAX_ORIGIN_LAG}
    if perimees:
        raise ValueError(
            f"Origine périmée pour {sorted(perimees)} : ancrage sur {perimees}, "
            f"alors que l'origine attendue est {expected_origin} "
            f"(tolérance {MAX_ORIGIN_LAG}). Une série exogène a probablement un "
            "trou qui invalide les features récentes (cf. temp_lag_168) — "
            "vérifier les NULL de `observations` et relancer l'ingestion sur "
            "une fenêtre couvrant le trou."
        )
    return origins


def frames_for_origin(per_target: dict, targets: list, with_production_total: bool) -> tuple:
    """(pred, lower, upper) pour un groupe de cibles partageant une origine.

    `production_total` n'est reconstruit que si les 4 sources sont dans ce
    groupe : les sommer à travers des origines différentes ne ferait que
    produire des NaN sur l'intersection vide des `target_ts`, soit un total
    silencieusement tronqué.
    """
    columns = {key: {t: per_target[t][key] for t in targets}
               for key in ("y_pred", "y_lower", "y_upper")}

    if with_production_total:
        derived = derive_production_total(per_target)
        for key in columns:
            columns[key]["production_total"] = derived[key]

    return tuple(pd.concat(columns[key], axis=1) for key in ("y_pred", "y_lower", "y_upper"))


def run(horizon: int = 24) -> int:
    t0 = time.time()

    try:

        df = get_observations()
        if df is None or df.empty:
            raise ValueError("Aucune observation en base : lancez l'ingestion d'abord")

        df = df.sort_index()
        last_observation = pd.Timestamp(df.index[-1])
        horizons = range(1, horizon + 1)

        # Report des retards de publication (cf. REPORT_MAX_HOURS). Sans effet
        # quand toutes les sources sont à jour -- c'est le cas nominal, le DAG
        # attendant la ligne complète avant de déclencher.
        df, comble = _combler_fin_de_serie(df)
        for colonne, horodatages in comble.items():
            logging.warning("Origine comblée par report : %s à %s (valeur de %s)",
                            colonne, ", ".join(str(h) for h in horodatages),
                            horodatages[0] - pd.Timedelta(hours=1))

        # Un jeu de 24 modèles dédiés par cible : chaque horizon est prédit
        # directement depuis l'origine, jamais nourri par une prédiction
        # antérieure (cf. local_forecaster.predict_with_conformal_intervals).
        per_target = {
            target: predict_with_conformal_intervals(df, target, horizons=horizons, alpha=0.05)
            for target in TARGETS
        }

        # L'origine est celle que le modèle a réellement utilisée, jamais la
        # dernière observation supposée : les deux divergent dès qu'une série
        # exogène a un trou (cf. `origins_by_target`).
        origins = origins_by_target(per_target, last_observation)

        # Un lot d'écriture par origine distincte : `save_forecasts` dérive
        # l'horizon de `target_ts - origin_ts`, donc une seule origine par
        # appel, sous peine d'horizons faux pour les cibles décalées.
        # production_total n'est plus une cible modélisée : reconstruite en
        # sommant les 4 sources, pour ne rien casser en aval (graphe du
        # déficit, habitudes du dashboard) -- cf. derive_production_total.
        model_version = get_model_version()
        n = 0
        for origin_ts in sorted(set(origins.values())):
            targets = sorted(t for t, o in origins.items() if o == origin_ts)
            pred_df, lower_df, upper_df = frames_for_origin(
                per_target, targets,
                with_production_total=set(PRODUCTION_SOURCES).issubset(targets),
            )
            n += save_forecasts(pred_df, origin_ts=origin_ts, model_version=model_version,
                                lower_df=lower_df, upper_df=upper_df)

        resume = " · ".join(f"{t}@{o:%Y-%m-%d %H:%M}" for t, o in sorted(origins.items()))
        # Le report est consigné dans `pipeline_runs` : une prévision dont
        # l'ancre a été comblée doit rester identifiable a posteriori, sans
        # quoi on ne pourrait plus distinguer ses métriques des autres.
        mention = f" · report sur {', '.join(sorted(comble))}" if comble else ""
        duration = time.time() - t0
        log_run("forecast_daily", "success", rows=n, duration_s=duration,
                message=f"horizon {horizon}h · origines {resume}{mention}")
        logging.info("Prévision OK : %s points (horizon %sh, origines %s) en %.1fs",
                     n, horizon, resume, duration)
        print(f"OK : {n} points de prévision persistés (horizon {horizon}h, origines {resume})")

        return n

    except Exception as e:
        log_run("forecast_daily", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Prévision en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    run(h)
