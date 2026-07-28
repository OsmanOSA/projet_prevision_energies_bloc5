"""Étude : que faire quand la consommation de 23h n'est pas encore publiée ?

RTE publie la consommation réalisée avec ~1h30 de latence, contre moins de
30 min pour la production et la température. Quand `forecast_daily` se
déclenche à 00h10, l'heure d'origine (23h la veille) peut donc être connue
pour toutes les variables sauf la consommation. Quatre stratégies possibles,
plus une référence :

  reference_23h  RTE a publié à l'heure -- borne haute, aucune imputation
  origine_22h    reculer l'origine d'une heure (que du réel, mais chaque
                 cible est alors prévue un cran plus loin : horizons 2->24)
  report_22h     persistance naïve : y(23h) := y(22h)
  moyenne_2h     y(23h) := moyenne(y(21h), y(22h))
  moyenne_3h     y(23h) := moyenne(y(20h), y(21h), y(22h))
  mediane_3h     y(23h) := médiane(y(20h), y(21h), y(22h))

Le protocole mesure l'erreur de prévision EN BOUT DE CHAÎNE, pas l'erreur sur
la valeur imputée : c'est la seule métrique qui décide. Pour chaque jour de
test, la prévision porte sur les 24 h de la journée civile et l'évaluation ne
retient que les 23 heures COMMUNES à toutes les stratégies (00:00 -> 22:00),
`origine_22h` n'ayant par construction rien à dire sur 23:00 -- son coût
propre, compté à part plutôt que dilué.

RÉSULTAT (exécution du 2026-07-28, 53 jours hors échantillon, cible
consommation_totale) :

    reference_23h   MAE  927,1    (p = 0,002 vs origine_22h)
    report_22h      MAE  948,6    (p = 0,035  -> significativement meilleur)
    mediane_3h      MAE  956,3    (p = 0,148  -> non significatif)
    moyenne_2h      MAE  961,1    (p = 0,177  -> non significatif)
    origine_22h     MAE  977,0    (référence de comparaison)
    moyenne_3h      MAE 1016,6    (p = 0,025  -> significativement pire)

Conclusion retenue et implémentée dans `scripts/forecast.py`
(`_combler_fin_de_serie`, `REPORT_MAX_HOURS`) : persistance naïve. Reculer
l'origine évite de dégrader `conso_delta_1`, mais le surcoût d'un horizon
supplémentaire l'emporte. Moyenne et médiane lissent à travers la marche
« heures creuses » de 23h et font moins bien.

Réserve : 5 tests appariés sans correction de multiplicité, un seuil de
Bonferroni (0,01) ne retiendrait pas p = 0,035. L'effet est réel mais modeste
(3 % du MAE) sur 53 jours.

Non mesuré : le bornage à 2 h du report (`REPORT_MAX_HOURS`) relève du
jugement, pas de la mesure. Pour le trancher, étendre `strategies()` à des
scénarios où deux heures consécutives manquent.

Usage (depuis un conteneur ayant accès à la base) :
    python -m analyses.etude_imputation_origine
"""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pipeline_prevision.db import get_observations
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
    predict_direct, forecast_origin,
)

PARIS = ZoneInfo("Europe/Paris")
CIBLE = "consommation_totale"
# Historique fourni au modèle. Au-delà de ~720 h le résultat ne bouge plus
# (écart max 1,6 MW sur ~48 000, soit 0,003 %), pour un coût 11x moindre que
# de rejouer tout l'historique à chaque appel.
HISTORIQUE = 720
RESULTATS = Path(__file__).with_name("resultats_imputation_origine.csv")


def utc_naif(jour: pd.Timestamp, heure: int) -> pd.Timestamp:
    """Heure locale Paris -> horodatage UTC naïf (convention de la base)."""
    local = pd.Timestamp(jour.year, jour.month, jour.day, heure, tz=PARIS)
    return local.tz_convert("UTC").tz_localize(None)


def prevoir(histoire: pd.DataFrame, origine_attendue: pd.Timestamp) -> pd.Series:
    """Prévision 24 h, avec contrôle que le modèle s'ancre bien où on croit.

    Le contrôle n'est pas décoratif : le forecaster s'ancre sur la dernière
    ligne aux features complètes, qui peut différer de la dernière ligne tout
    court. Sans cette assertion, une stratégie pourrait être évaluée sur une
    origine qui n'est pas la sienne.
    """
    pred = predict_direct(histoire.tail(HISTORIQUE), CIBLE, horizons=range(1, 25))
    reelle = forecast_origin(pred)
    if reelle != origine_attendue:
        raise AssertionError(f"origine {reelle} au lieu de {origine_attendue}")
    return pred["y_pred"]


def strategies(serie: pd.Series, o22: pd.Timestamp) -> dict:
    """Valeur imputée pour 23:00 selon chaque stratégie (23:00 est manquant)."""
    v = [serie.loc[o22 - pd.Timedelta(hours=k)] for k in range(3)]  # 22h, 21h, 20h
    return {
        "report_22h": v[0],
        "moyenne_2h": np.mean(v[:2]),
        "moyenne_3h": np.mean(v),
        "mediane_3h": np.median(v),
    }


def etudier(obs: pd.DataFrame, jours: list) -> pd.DataFrame:
    lignes = []
    for jour in jours:
        o23 = utc_naif(jour - pd.Timedelta(days=1), 23)
        o22 = o23 - pd.Timedelta(hours=1)
        communes = pd.date_range(o23 + pd.Timedelta(hours=1), periods=23, freq="h")
        derniere = o23 + pd.Timedelta(hours=24)   # hors de portée de origine_22h

        besoin = obs.loc[:derniere]
        if len(besoin) < HISTORIQUE + 24 or besoin.index[-1] != derniere:
            continue
        if obs.loc[o23 - pd.Timedelta(hours=2):derniere, CIBLE].isna().any():
            continue

        reel = obs.loc[communes, CIBLE]
        try:
            preds = {
                "reference_23h": prevoir(obs.loc[:o23], o23),
                "origine_22h": prevoir(obs.loc[:o22], o22),
            }
            for nom, valeur in strategies(obs[CIBLE], o22).items():
                truque = obs.loc[:o23].copy()
                truque.loc[o23, CIBLE] = valeur
                preds[nom] = prevoir(truque, o23)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {jour.date()} ignoré : {exc}", file=sys.stderr)
            continue

        for nom, p in preds.items():
            err = (p.reindex(communes) - reel).dropna()
            if len(err) != len(communes):
                continue
            lignes.append({"jour": jour.date(), "strategie": nom,
                           "mae": err.abs().mean(), "biais": err.mean()})
    return pd.DataFrame(lignes)


def rapport(res: pd.DataFrame, titre: str):
    if res.empty:
        print(f"\n### {titre} : aucun jour exploitable")
        return
    tab = res.groupby("strategie").agg(MAE=("mae", "mean"), biais=("biais", "mean"))
    tab["surcout_vs_reference"] = tab["MAE"] - tab.loc["reference_23h", "MAE"]
    ordre = ["reference_23h", "origine_22h", "report_22h",
             "moyenne_2h", "moyenne_3h", "mediane_3h"]
    tab = tab.reindex([o for o in ordre if o in tab.index])

    print(f"\n### {titre} — {res['jour'].nunique()} jours, 23 heures par jour")
    print(tab.round(1).to_string())

    # Les écarts moyens sont faibles devant la dispersion jour à jour : un test
    # apparié est indispensable avant d'en tirer une décision.
    pivot = res.pivot(index="jour", columns="strategie", values="mae")
    if "origine_22h" not in pivot:
        return
    print("\n  Wilcoxon apparié, chaque stratégie vs `origine_22h` :")
    for s in [o for o in ordre if o != "origine_22h" and o in pivot]:
        d = pivot[[s, "origine_22h"]].dropna()
        _, p = wilcoxon(d[s], d["origine_22h"])
        gagnes = int((d[s] < d["origine_22h"]).sum())
        print(f"    {s:14s} écart moyen {(d[s] - d['origine_22h']).mean():+8.1f} MW"
              f"   p = {p:.3f}   {gagnes:3d}/{len(d)} jours gagnés"
              f"   -> {'significatif' if p < 0.05 else 'NON significatif'}")


if __name__ == "__main__":
    obs = get_observations().sort_index()

    # Hors échantillon uniquement : sur la période d'entraînement le modèle a
    # mémorisé, et le classement s'y inverse artificiellement en faveur des
    # stratégies qui ne dégradent aucune feature.
    jours = pd.date_range("2026-06-05", "2026-07-27", freq="D")

    res = etudier(obs, list(jours))
    res.to_csv(RESULTATS, index=False)
    rapport(res, "HORS ÉCHANTILLON (postérieur au jeu d'entraînement)")
    print(f"\nDétail par jour : {RESULTATS}")
