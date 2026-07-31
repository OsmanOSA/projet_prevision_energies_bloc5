"""Agrège les replis de `rolling_origin_cv` en un verdict par catégorie de jour.

C'est l'étape qui donne son intérêt à la validation glissante : au lieu des 6
jours fériés et 24 lundis du holdout unique, on agrège **~35 fériés et ~165
lundis** de prédictions toutes hors-échantillon.

Ce que le script rend
---------------------
1. MAE par catégorie de jour (mardi-vendredi, lundi, samedi, dimanche, veille /
   jour / lendemain de férié), contre RTE.
2. Significativité par bootstrap à blocs mobiles de 7 jours sur la différence
   quotidienne appariée — les erreurs horaires sont trop autocorrélées pour un
   test i.i.d.
3. Comparaison de DEUX variantes si les deux dossiers existent (avec / sans
   features de température prévue), sur les origines communes.

Le repère RTE ne sert QU'À se comparer : il n'entre jamais dans un modèle
(cf. `scripts/fetch_rte_forecast.py`).

Usage :
    python -m scripts.analyse_rolling_cv
    python -m scripts.analyse_rolling_cv --variante sans_prevision
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.db.config import get_engine  # noqa: E402
from pipeline_prevision.utils.main_utils.feature_engineering import (  # noqa: E402
    french_holidays,
)
from scripts.compare_models import block_bootstrap_ci  # noqa: E402

RACINE = os.path.join("docs", "rolling_cv")
ORDRE = ["mardi-vendredi", "lundi", "samedi", "dimanche",
         "veille ferie", "ferie", "lendemain ferie"]


def charger(dossier: str) -> pd.DataFrame:
    fichiers = sorted(glob.glob(os.path.join(dossier, "repli_*.parquet")))
    if not fichiers:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in fichiers], ignore_index=True)
    print(f"  {os.path.basename(dossier)} : {len(fichiers)} replis, "
          f"{len(df):,} points, {df['origine'].nunique()} origines".replace(",", " "))
    return df


def categoriser(cibles: pd.Series) -> pd.Series:
    annees = range(cibles.dt.year.min() - 1, cibles.dt.year.max() + 2)
    feries = french_holidays(annees)

    def cat(ts):
        ts = pd.Timestamp(ts)
        if ts.normalize() in feries:
            return "ferie"
        if (ts - pd.Timedelta(days=1)).normalize() in feries:
            return "lendemain ferie"
        if (ts + pd.Timedelta(days=1)).normalize() in feries:
            return "veille ferie"
        return ["lundi", "mardi-vendredi", "mardi-vendredi", "mardi-vendredi",
                "mardi-vendredi", "samedi", "dimanche"][ts.dayofweek]

    return cibles.map(cat)


def tableau(d: pd.DataFrame, colonnes: dict, titre: str) -> None:
    """colonnes : {etiquette affichee -> colonne d'erreur absolue}."""
    print("\n" + "=" * 92)
    print(titre)
    print("=" * 92)
    entetes = "".join(f"{e:>13}" for e in colonnes)
    print(f"{'categorie':18}{'jours':>7}{'points':>8}{entetes}{'meilleur':>18}")
    print("-" * 92)
    for cat in ORDRE:
        g = d[d["cat"] == cat]
        if g.empty:
            continue
        moy = {e: g[c].mean() for e, c in colonnes.items()}
        best = min(moy, key=moy.get)
        vals = "".join(f"{moy[e]:>13.0f}" for e in colonnes)
        print(f"{cat:18}{g['origine'].nunique():>7}{len(g):>8}{vals}{best:>18}")
    print("-" * 92)
    moy = {e: d[c].mean() for e, c in colonnes.items()}
    best = min(moy, key=moy.get)
    vals = "".join(f"{moy[e]:>13.0f}" for e in colonnes)
    print(f"{'TOUTES':18}{d['origine'].nunique():>7}{len(d):>8}{vals}{best:>18}")


def significativite(d: pd.DataFrame, paires: list[tuple[str, str, str]]) -> None:
    print("\n" + "=" * 92)
    print("SIGNIFICATIVITE (bootstrap a blocs de 7 j, MAE quotidienne appariee)")
    print("=" * 92)
    par_jour = d.groupby("origine")[[c for _, a, b in paires for c in (a, b)]].mean()
    for label, a, b in paires:
        diff = (par_jour[a] - par_jour[b]).to_numpy()
        bas, haut = block_bootstrap_ci(diff)
        if np.isnan(bas):
            verdict = "echantillon trop court"
        elif bas < 0 and haut < 0:
            verdict = "SIGNIFICATIF (favorable)"
        elif bas > 0 and haut > 0:
            verdict = "SIGNIFICATIF (defavorable)"
        else:
            verdict = "non concluant"
        print(f"  {label:34} {diff.mean():+8.1f} MW   IC95 [{bas:+7.1f}, {haut:+7.1f}]   {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cible", default="consommation_totale")
    p.add_argument("--racine", default=RACINE)
    args = p.parse_args()

    print("Chargement des replis :")
    avec = charger(os.path.join(args.racine, f"{args.cible}_avec_prevision"))
    sans = charger(os.path.join(args.racine, f"{args.cible}_sans_prevision"))
    if avec.empty and sans.empty:
        sys.exit(f"Aucun repli dans {args.racine}. Lancer `rolling_origin_cv` d'abord.")

    base = avec if not avec.empty else sans
    d = base.rename(columns={"y_pred": "pred_avec"})[
        ["origine", "horizon", "cible_ts", "y_true", "pred_avec", "ancre"]].copy()
    if not avec.empty and not sans.empty:
        d = d.merge(sans.rename(columns={"y_pred": "pred_sans"})[
            ["origine", "horizon", "pred_sans"]], on=["origine", "horizon"], how="inner")

    rte = pd.read_sql(
        "SELECT target_ts, y_pred AS pred_rte FROM forecasts "
        "WHERE variable = 'consommation_totale_rte'",
        get_engine(), parse_dates=["target_ts"])
    d = d.merge(rte, left_on="cible_ts", right_on="target_ts", how="left").drop(
        columns=["target_ts"])

    colonnes = {"modele": "pred_avec"}
    if "pred_sans" in d.columns:
        colonnes = {"avec prev.": "pred_avec", "sans prev.": "pred_sans"}
    if d["pred_rte"].notna().any():
        colonnes["RTE (J-1)"] = "pred_rte"
    # On n'evalue que les points ou TOUTES les series comparees existent :
    # comparer des echantillons differents ne voudrait rien dire.
    d = d.dropna(subset=list(colonnes.values()) + ["y_true"])
    erreurs = {}
    for etiquette, col in colonnes.items():
        e = f"err_{col}"
        d[e] = (d[col] - d["y_true"]).abs()
        erreurs[etiquette] = e

    d["cat"] = categoriser(d["cible_ts"])
    annees = f"{d['cible_ts'].min():%Y-%m} -> {d['cible_ts'].max():%Y-%m}"
    print(f"\n{len(d):,} points evalues, {d['origine'].nunique()} origines, {annees}"
          .replace(",", " "))
    feries = d[d["cat"] == "ferie"]["origine"].nunique()
    lundis = d[d["cat"] == "lundi"]["origine"].nunique()
    print(f"Jours rares couverts : {feries} feries, {lundis} lundis "
          f"(holdout unique : 6 et 24)")

    tableau(d, erreurs, "MAE PAR CATEGORIE DE JOUR (MW) — tout hors-echantillon")

    paires = []
    if "pred_sans" in d.columns:
        paires.append(("avec prevision - sans prevision", "err_pred_avec", "err_pred_sans"))
    if "pred_rte" in d.columns and d["pred_rte"].notna().any():
        paires.append(("modele - RTE", "err_pred_avec", "err_pred_rte"))
    if paires:
        significativite(d, paires)

    if "pred_rte" in d.columns:
        pj = d.groupby("origine")[["err_pred_avec", "err_pred_rte"]].mean()
        gagnes = int((pj["err_pred_avec"] < pj["err_pred_rte"]).sum())
        print(f"\n  jours gagnes contre RTE : {gagnes}/{len(pj)} ({100*gagnes/len(pj):.0f} %)")

    print("\n" + "=" * 92)
    print("ANCRE RETENUE PAR REPLI ET PAR HORIZON")
    print("=" * 92)
    if "ancre" in base.columns:
        pivot = base.pivot_table(index="horizon", columns="ancre", values="y_pred",
                                 aggfunc="count").fillna(0).astype(int)
        print(pivot.to_string())


if __name__ == "__main__":
    main()
