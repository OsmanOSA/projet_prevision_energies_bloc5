"""Mesure l'apport de chaque jeu de features par backtesting à origine glissante.

Applique le protocole validé dans `notebooks/comparison_time_series_models.ipynb`
(§ 5) : origines roulantes, embargo de 24 h, entraînement strictement antérieur,
comparaison sur les mêmes origines. Aucune décision de bascule ne doit être prise
sans passer par ici — un `train_test_split` ou un `KFold` sous-estime l'erreur
réelle de plus de 40 % sur ces données.

Échelle de variantes testées (chacune ajoute une brique à la précédente) :

    utc          calendrier UTC seul, température station unique  (état d'avant)
    calendrier   + heure locale, jours fériés, ponts              (§ 3 du notebook)
    temp_fr      + température France pondérée sur 17 stations

Usage :
    python -m scripts.evaluate_features
    python -m scripts.evaluate_features --cible WIND_ONSHORE --folds 3
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from pipeline_prevision.utils.main_utils.feature_engineering import (  # noqa: E402
    HORIZON_MAX, add_target_features, build_features_for_target, build_series_by_target,
)

# Marqueurs des colonnes ajoutées par le levier « calendrier français ».
MARQUEURS_CAL = ("_loc_", "_holiday", "_bridge", "_august", "_xmas_break")

PARAMS = dict(n_estimators=200, learning_rate=0.06, num_leaves=31,
              min_child_samples=30, colsample_bytree=0.7, max_bin=127,
              verbose=-1, n_jobs=4, random_state=42)


def mae(y, p):
    return float(np.nanmean(np.abs(y - p)))


def wape(y, p):
    return float(np.nansum(np.abs(y - p)) / np.nansum(np.abs(y)))


def cout_metier(y, p, cout_sous=80.0, cout_sur=25.0):
    """Coût de déséquilibre asymétrique (€/h). Cf. § 4 du notebook."""
    return float(np.nanmean(cout_sous * np.maximum(y - p, 0)
                            + cout_sur * np.maximum(p - y, 0)))


def construire_folds(index, n_folds, jours_test, pas_origine):
    heure = pd.Timedelta(1, unit="h")
    jour = pd.Timedelta(1, unit="D")
    fin_donnees = index.max() - HORIZON_MAX * heure
    folds = []
    for k in range(n_folds):
        fin_test = fin_donnees - (jours_test * (n_folds - 1 - k)) * jour
        debut_test = fin_test - jours_test * jour + heure
        folds.append({
            "n": k + 1,
            # embargo = horizon maximal : sans lui, la cible target_h24 de la
            # dernière ligne d'entraînement chevauche la fenêtre de test.
            "fin_train": debut_test - (HORIZON_MAX + 1) * heure,
            "debut_test": debut_test, "fin_test": fin_test,
            "origines": pd.date_range(debut_test, fin_test, freq=f"{pas_origine}h"),
        })
    return folds


def backtest(features, colonnes, folds, verite, prefix, garder_fr):
    """Un modèle LightGBM par horizon et par fold — chemin identique au pipeline.

    `add_target_features` est appliqué par horizon, exactement comme
    `model_trainer._train_horizons` : c'est là que sont calculées les variables
    calendaires ancrées sur l'heure CIBLE, celles qui pèsent le plus puisque
    chaque horizon a son propre modèle. Les mesurer suppose donc de reproduire
    ce chemin, pas seulement les features d'origine.

    `garder_fr=False` retire les colonnes du levier calendrier français pour
    reconstituer l'état antérieur du pipeline.
    """
    delta_col = f"{prefix}_delta_1"
    sorties = []
    for f in folds:
        dispo = features.index.intersection(f["origines"])
        origines = list(dispo[features.loc[dispo, colonnes].notna().all(axis=1)])
        if not origines:
            continue

        app = features.loc[:f["fin_train"]]
        app = app[app[colonnes].notna().all(axis=1)]
        base_app = app[colonnes]
        base_org = features.loc[origines, colonnes]

        preds = np.empty((len(origines), HORIZON_MAX))
        for h in range(1, HORIZON_MAX + 1):
            # Une seule copie vivante à la fois : convertie en float32 puis
            # libérée. La version naïve gardait 24 DataFrames et saturait la RAM.
            Xa_df = add_target_features(base_app, h, delta_col)
            cols_h = [c for c in Xa_df.columns
                      if garder_fr or not any(m in c for m in MARQUEURS_CAL)]
            Xa = Xa_df[cols_h].to_numpy(dtype=np.float32)
            del Xa_df
            Xo = add_target_features(base_org, h, delta_col)[cols_h].to_numpy(np.float32)

            ya = app[f"target_h{h}"].to_numpy(dtype=np.float32)
            m = np.isfinite(ya)
            modele = lgb.LGBMRegressor(**PARAMS).fit(Xa[m], ya[m])
            preds[:, h - 1] = modele.predict(Xo)
            del modele, Xa, Xo
        del base_app, base_org, app
        gc.collect()

        for i, org in enumerate(origines):
            idx = org + pd.to_timedelta(np.arange(1, HORIZON_MAX + 1), unit="h")
            sorties.append(pd.DataFrame({
                "fold": f["n"], "origine": org,
                "horizon": np.arange(1, HORIZON_MAX + 1), "cible_ts": idx,
                "y": verite.reindex(idx).to_numpy(), "pred": preds[i]}))
    return pd.concat(sorties, ignore_index=True).dropna(subset=["y", "pred"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join("datasets", "data.csv"))
    p.add_argument("--cible", default="consommation_totale")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--jours-test", type=int, default=28)
    p.add_argument("--pas-origine", type=int, default=6)
    args = p.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"],
                     index_col="timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].asfreq("h")
    series = build_series_by_target(df)
    verite = pd.to_numeric(df[args.cible], errors="coerce")

    if "temp_fr" not in df.columns:
        sys.exit("Colonne `temp_fr` absente : lancer d'abord "
                 "`python -m scripts.build_temperature_france`")

    folds = construire_folds(df.index, args.folds, args.jours_test, args.pas_origine)
    print(f"Cible : {args.cible}")
    print(f"Protocole : {len(folds)} folds x {args.jours_test} j, origine toutes "
          f"les {args.pas_origine} h, horizons 1..{HORIZON_MAX}, embargo {HORIZON_MAX} h")
    print(f"Test du fold 1 : {folds[0]['debut_test']:%Y-%m-%d} -> "
          f"{folds[-1]['fin_test']:%Y-%m-%d} au total\n")

    # Les trois jeux de features. `utc` retire les colonnes du levier calendrier
    # pour reconstituer exactement l'état antérieur du pipeline.
    feat_station, prefix, cibles = build_features_for_target(
        series, pd.to_numeric(df["temp"], errors="coerce"), args.cible)
    feat_france, _, _ = build_features_for_target(
        series, pd.to_numeric(df["temp_fr"], errors="coerce"), args.cible)

    toutes = [c for c in feat_station.columns if c not in cibles]
    sans_cal = [c for c in toutes if not any(m in c for m in MARQUEURS_CAL)]

    variantes = [
        ("utc (état d'avant)", feat_station, sans_cal, False),
        ("+ calendrier FR", feat_station, toutes, True),
        ("+ temp_fr (17 stations)", feat_france, toutes, True),
    ]
    print(f"features d'origine : {len(sans_cal)} sans calendrier FR, "
          f"{len(toutes)} avec (+ variables ancrées cible par horizon)\n")

    resultats, brut = [], {}
    for nom, feats, cols, garder_fr in variantes:
        t0 = time.time()
        res = backtest(feats, cols, folds, verite, prefix, garder_fr)
        brut[nom] = res
        y, pr = res["y"].to_numpy(), res["pred"].to_numpy()
        resultats.append({
            "variante": nom, "features": len(cols), "MAE": mae(y, pr),
            "WAPE": wape(y, pr), "biais": float(np.mean(pr - y)),
            "coût €/h": cout_metier(y, pr), "n": len(res),
            "secondes": round(time.time() - t0),
        })
        print(f"  {nom:26s} MAE {resultats[-1]['MAE']:8.1f} MW "
              f"({resultats[-1]['secondes']:3d} s)")

    tab = pd.DataFrame(resultats)
    ref = tab["MAE"].iloc[0]
    tab["gain cumulé %"] = 100 * (1 - tab["MAE"] / ref)
    tab["gain de l'étape %"] = 100 * (1 - tab["MAE"] / tab["MAE"].shift().fillna(ref))

    print("\n" + "=" * 100)
    print(tab.to_string(index=False,
                        float_format=lambda v: f"{v:,.2f}".replace(",", " ")))

    print("\n--- Régularité : MAE par fold (un gain qui ne tient pas sur tous les "
          "folds n'est pas un gain) ---")
    par_fold = pd.DataFrame({
        nom: res.groupby("fold").apply(lambda g: mae(g["y"], g["pred"]),
                                       include_groups=False)
        for nom, res in brut.items()})
    par_fold.index = [f"fold {i} ({folds[i-1]['debut_test']:%b %Y})"
                      for i in par_fold.index]
    print(par_fold.to_string(float_format=lambda v: f"{v:,.1f}".replace(",", " ")))

    noms = list(brut)
    gagne = (par_fold[noms[-1]] < par_fold[noms[0]]).sum()
    print(f"\n« {noms[-1]} » bat « {noms[0]} » sur {gagne}/{len(par_fold)} folds")

    print("\n--- MAE par horizon ---")
    par_h = pd.DataFrame({
        nom: res.groupby("horizon").apply(lambda g: mae(g["y"], g["pred"]),
                                          include_groups=False)
        for nom, res in brut.items()})
    print(par_h.loc[[1, 6, 12, 18, 24]].to_string(
        float_format=lambda v: f"{v:,.1f}".replace(",", " ")))


if __name__ == "__main__":
    main()
