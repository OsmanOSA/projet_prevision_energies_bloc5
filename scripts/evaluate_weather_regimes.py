"""Backtest stratifié par RÉGIME MÉTÉO : le test qui juge la prévision de température.

`scripts/compare_models.py` rend le verdict global, cible par cible, avec IC95.
Il ne répond pas à la question qui a motivé ces features : notre modèle battait
RTE (J-1) par météo stable et perdait par bascule thermique, parce que toutes ses
features thermiques étaient gelées à l'origine. Une MAE moyennée sur six mois
mélange les deux régimes et peut masquer complètement le gain.

Ce script stratifie donc les origines par l'ampleur de la bascule :

    delta_temp = moyenne(temp_fr_om sur t+1..t+24) - temp_fr_om(t)

C'est la même définition que le diagnostic initial, qui avait mesuré, sur les
origines >= 2026-07-23 : +16 % de MAE contre RTE par |delta_temp| < 2 °C, mais -15 %
au-delà. Le succès attendu ici n'est pas « une MAE globale plus basse » mais
**« l'écart entre les deux régimes se resserre »**.

Le signe compte autant que l'ampleur : à l'origine, le biais suivait celui de
delta_temp (réchauffement -> sous-estimation, refroidissement -> surestimation). Si les
features fonctionnent, cette corrélation doit s'effondrer.

Usage :
    python -m scripts.evaluate_weather_regimes --start 2026-02-01 --end 2026-07-27
    python -m scripts.evaluate_weather_regimes --start ... --end ... --cible SOLAR
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.utils.main_utils.utils import load_object  # noqa: E402
from pipeline_prevision.utils.main_utils.feature_engineering import (  # noqa: E402
    HORIZON_MAX, build_origin_feature_frame, build_series_by_target,
    select_temperature, select_forecast_temperature,
)
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (  # noqa: E402
    _direct_prediction, get_anchor,
)
from scripts.compare_models import block_bootstrap_ci  # noqa: E402

# Bornes des régimes, en °C de bascule sur les 24 h suivant l'origine. 2 °C est
# le seuil qui séparait le mieux gain et perte dans le diagnostic initial ; 4 °C
# isole les épisodes francs (la canicule du 27/07/2026 était à +5,5 °C).
SEUILS = (2.0, 4.0)

# Heures d'origine retenues par défaut (UTC).
#
# ATTENTION -- ce filtre n'est pas cosmétique, il conditionne la validité de
# toute la stratification. delta_temp = moyenne(t+1..t+24) - temp(t) mélange DEUX
# choses : le cycle diurne et le changement de régime météo. À une origine de
# 03 h UTC, la moyenne des 24 h suivantes est mécaniquement plus chaude que
# l'instant d'origine — tous les jours, quel que soit le temps qu'il fait.
# Stratifier sur toutes les origines horaires revient donc à trier les origines
# par heure de la journée en croyant les trier par régime météo (mesuré : le
# champion y paraît MEILLEUR en « bascule » qu'en « stable », artefact pur).
#
# À heure d'origine fixée, la composante diurne devient une constante et delta_temp
# mesure ce qu'on veut : l'écart de température d'un jour à l'autre. On retient
# les origines réellement utilisées en production — le DAG `forecast_daily`
# tourne vers 22 h UTC et s'ancre sur la dernière heure observée.
HEURES_ORIGINE_DEFAUT = (21, 22, 23)


def mae_par_origine(composite: dict, series_by_target: dict, temp: pd.Series,
                    target: str, start, end, temp_prev=None) -> pd.DataFrame:
    """MAE et biais moyennés sur les 24 horizons, UNE LIGNE PAR ORIGINE.

    Agréger par origine (et non par jour cible comme `compare_models.daily_series`)
    est ce qui permet la stratification : delta_temp est une propriété de l'origine,
    puisqu'il décrit ce que le modèle ne pouvait pas savoir à cet instant.
    """
    entry = composite[target]
    feature_columns = entry["feature_columns"]
    prefix = entry["prefix"]
    delta_col = f"{prefix}_delta_1"
    series = series_by_target[target]

    features_df, _ = build_origin_feature_frame(series_by_target, temp, target,
                                                temp_prev=temp_prev)

    erreurs, biais = {}, {}
    for horizon in range(1, HORIZON_MAX + 1):
        futur = series.shift(-horizon)
        mask = (features_df[feature_columns].notna().all(axis=1)
                & futur.notna()
                & (features_df.index >= start) & (features_df.index <= end))
        frame = features_df.loc[mask]
        if frame.empty:
            continue

        y_pred = _direct_prediction(
            entry["models"][horizon], entry["alphas"][horizon],
            entry["seasonal_weights"][horizon], frame[feature_columns],
            horizon, prefix, delta_col, get_anchor(entry, horizon),
        )
        ecart = y_pred - futur.loc[mask].to_numpy()
        erreurs[horizon] = pd.Series(np.abs(ecart), index=frame.index)
        biais[horizon] = pd.Series(ecart, index=frame.index)

    if not erreurs:
        return pd.DataFrame(columns=["mae", "biais", "n_horizons"])

    mae = pd.concat(erreurs, axis=1)
    return pd.DataFrame({
        "mae": mae.mean(axis=1),
        "biais": pd.concat(biais, axis=1).mean(axis=1),
        # Une origine dont tous les horizons ne sont pas évaluables n'est pas
        # comparable aux autres : on la trace pour pouvoir l'exclure.
        "n_horizons": mae.notna().sum(axis=1),
    }).dropna(subset=["mae"])


def bascule_thermique(temp_om: pd.Series, origines: pd.DatetimeIndex) -> pd.Series:
    """delta_temp : moyenne de la température sur t+1..t+24 moins celle à l'origine.

    C'est exactement l'information dont le modèle était privé : à l'origine il ne
    connaissait que `temp_0`, et extrapolait les 24 h suivantes avec.
    """
    cumul = pd.Series(0.0, index=origines)
    compte = pd.Series(0, index=origines)
    for h in range(1, HORIZON_MAX + 1):
        valeurs = temp_om.reindex(origines + pd.Timedelta(hours=h))
        valeurs.index = origines
        cumul = cumul.add(valeurs.fillna(0.0))
        compte = compte.add(valeurs.notna().astype(int))
    moyenne_cible = (cumul / compte).where(compte > 0)
    return moyenne_cible - temp_om.reindex(origines)


def etiqueter(delta: pd.Series) -> pd.Series:
    bas, haut = SEUILS
    return pd.Series(
        np.select(
            [delta.abs() < bas, delta.abs() < haut],
            [f"A. stable (<{bas:g} C)", f"B. transition ({bas:g}-{haut:g} C)"],
            default=f"C. bascule (>{haut:g} C)"),
        index=delta.index)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion", default="final_models")
    p.add_argument("--challenger", default="candidate_models")
    p.add_argument("--data", default=os.path.join("datasets", "data.csv"))
    p.add_argument("--cible", default="consommation_totale")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--heures-origine", default=",".join(map(str, HEURES_ORIGINE_DEFAUT)),
                   help="heures UTC d'origine retenues, séparées par des virgules. "
                        "`toutes` désactive le filtre — mais la stratification "
                        "devient alors confondue par le cycle diurne (cf. code).")
    p.add_argument("--out", default=os.path.join("docs", "weather_regimes.json"))
    args = p.parse_args()

    if args.heures_origine.strip().lower() == "toutes":
        heures = None
    else:
        heures = tuple(int(h) for h in args.heures_origine.split(","))

    full = pd.read_csv(args.data, sep=None, engine="python",
                       parse_dates=["timestamp"], index_col="timestamp").sort_index()
    full = full[~full.index.duplicated(keep="last")].asfreq("h")

    series_by_target = build_series_by_target(full)
    temp = select_temperature(full)
    temp_prev = select_forecast_temperature(full)
    if temp_prev is None:
        sys.exit("temp_fr_prev absente du jeu : lancer d'abord "
                 "`python -m scripts.backfill_prevision_temperature --csv`.")
    # La stratification se fait sur l'observé Open-Meteo, jamais sur la prévision :
    # on veut mesurer la performance face à la bascule RÉELLE, pas face à celle
    # que le modèle croyait voir.
    temp_om = pd.to_numeric(full["temp_fr_om"], errors="coerce")

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"Cible   : {args.cible}")
    print(f"Fenêtre : {start:%Y-%m-%d} -> {end:%Y-%m-%d}\n")

    resultats = {}
    for label, chemin, prev in (("champion", args.champion, None),
                                ("challenger", args.challenger, temp_prev)):
        composite = load_object(os.path.join(chemin, "model.pkl"))
        # Le champion précède ces features : lui passer temp_prev élargirait son X
        # et `_align_to_model` le ramènerait de toute façon à ses colonnes. On lui
        # passe None pour que la comparaison soit explicite, pas implicite.
        besoin = any("temp_prev" in c for c in composite[args.cible]["feature_columns"])
        resultats[label] = mae_par_origine(
            composite, series_by_target, temp, args.cible, start, end,
            temp_prev=temp_prev if besoin else prev)
        print(f"{label:11} : {len(resultats[label]):5} origines évaluées"
              f"  (features de prévision : {'OUI' if besoin else 'non'})")
        del composite

    communes = resultats["champion"].index.intersection(resultats["challenger"].index)
    if len(communes) == 0:
        sys.exit("Aucune origine commune aux deux modèles sur cette fenêtre.")

    tableau = pd.DataFrame({
        "mae_champion": resultats["champion"].loc[communes, "mae"],
        "mae_challenger": resultats["challenger"].loc[communes, "mae"],
        "biais_champion": resultats["champion"].loc[communes, "biais"],
        "biais_challenger": resultats["challenger"].loc[communes, "biais"],
    })
    tableau["delta_temp"] = bascule_thermique(temp_om, communes)
    tableau = tableau.dropna(subset=["delta_temp"])

    if heures is not None:
        avant = len(tableau)
        tableau = tableau[tableau.index.hour.isin(heures)]
        print(f"\nFiltre d'heure d'origine {heures} : {avant} -> {len(tableau)} origines")
        print("  (à heure fixée, delta_temp mesure le changement de régime météo et non "
              "le cycle diurne)")
    else:
        print("\n/!\\ Filtre d'heure désactivé : delta_temp mélange cycle diurne et régime "
              "météo, la stratification ci-dessous n'est PAS interprétable comme telle.")
    if tableau.empty:
        sys.exit("Aucune origine après filtrage.")
    tableau["regime"] = etiqueter(tableau["delta_temp"])
    rapport_heures = list(heures) if heures is not None else "toutes"

    print(f"\n{len(tableau)} origines communes retenues\n")
    print("=" * 92)
    print("MAE PAR RÉGIME MÉTÉO")
    print("=" * 92)
    entete = (f"{'régime':26} {'origines':>9} {'champion':>10} {'challenger':>11} "
              f"{'écart':>9} {'IC95 de l\'écart':>22}")
    print(entete)
    print("-" * len(entete))

    rapport = {"fenetre": [str(start), str(end)], "cible": args.cible,
           "heures_origine": rapport_heures, "regimes": {}}
    for regime, groupe in tableau.groupby("regime"):
        mc, mch = groupe["mae_champion"].mean(), groupe["mae_challenger"].mean()
        diff = (groupe["mae_challenger"] - groupe["mae_champion"]).to_numpy()
        bas, haut = block_bootstrap_ci(diff)
        gain = 100 * (mc - mch) / mc if mc else float("nan")
        print(f"{regime:26} {len(groupe):>9} {mc:>10.1f} {mch:>11.1f} "
              f"{gain:>+8.1f}% {f'[{bas:+.1f}, {haut:+.1f}] MW':>22}")
        rapport["regimes"][regime] = {
            "origines": int(len(groupe)), "mae_champion": round(float(mc), 2),
            "mae_challenger": round(float(mch), 2), "gain_pct": round(float(gain), 2),
            "ic95_ecart_mw": [None if np.isnan(bas) else round(float(bas), 2),
                              None if np.isnan(haut) else round(float(haut), 2)],
        }

    glob_c, glob_ch = tableau["mae_champion"].mean(), tableau["mae_challenger"].mean()
    bas, haut = block_bootstrap_ci((tableau["mae_challenger"] - tableau["mae_champion"]).to_numpy())
    gain = 100 * (glob_c - glob_ch) / glob_c
    print("-" * len(entete))
    print(f"{'TOUS RÉGIMES':26} {len(tableau):>9} {glob_c:>10.1f} {glob_ch:>11.1f} "
          f"{gain:>+8.1f}% {f'[{bas:+.1f}, {haut:+.1f}] MW':>22}")
    rapport["global"] = {"origines": int(len(tableau)), "gain_pct": round(float(gain), 2),
                         "ic95_ecart_mw": [None if np.isnan(bas) else round(float(bas), 2),
                                           None if np.isnan(haut) else round(float(haut), 2)]}

    print("\n" + "=" * 92)
    print("CE QUI EST VRAIMENT EN JEU : l'écart de performance entre régimes")
    print("=" * 92)
    par_regime = tableau.groupby("regime")[["mae_champion", "mae_challenger"]].mean()
    if len(par_regime) >= 2:
        for label in ("champion", "challenger"):
            col = f"mae_{label}"
            ecart = par_regime[col].max() - par_regime[col].min()
            print(f"  {label:11} : MAE de {par_regime[col].min():.1f} (meilleur régime) "
                  f"à {par_regime[col].max():.1f} (pire) -> amplitude {ecart:.1f} MW")
        rapport["amplitude_inter_regimes"] = {
            label: round(float(par_regime[f"mae_{label}"].max()
                               - par_regime[f"mae_{label}"].min()), 2)
            for label in ("champion", "challenger")}

    print("\n" + "=" * 92)
    print("CORRÉLATION BIAIS / BASCULE THERMIQUE (doit s'effondrer si ça marche)")
    print("=" * 92)
    for label in ("champion", "challenger"):
        c = tableau["delta_temp"].corr(tableau[f"biais_{label}"])
        print(f"  {label:11} : corr(delta_temp, biais) = {c:+.3f}   "
              f"biais moyen {tableau[f'biais_{label}'].mean():+8.1f} MW")
        rapport.setdefault("corr_delta_biais", {})[label] = round(float(c), 3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rapport, handle, indent=2, ensure_ascii=False)
    print(f"\nRapport : {args.out}")


if __name__ == "__main__":
    main()
