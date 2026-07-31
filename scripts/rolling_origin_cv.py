"""Validation à origine glissante sur tout l'historique.

Le problème qu'elle résout
--------------------------
Le holdout unique (15 % final) ne contient que **6 à 10 jours fériés et 24 à 41
lundis**. À ces effectifs, aucune conclusion par catégorie de jour n'est
atteignable — et c'est exactement le mur sur lequel trois expériences
successives se sont arrêtées, toutes concluant « direction favorable, IC95 qui
croise zéro » :

    features de température prévue  IC95 [-200,8 ; -112,3] MW   (celle-là a conclu)
    ancre par type de jour          IC95 [ -25,3 ;  +39,8] MW   non concluant
    historique étendu à 5,35 ans    IC95 [ -52,9 ;  +10,9] MW   non concluant

En repliant l'évaluation sur toute la profondeur, on passe à **~49 fériés et
~233 lundis** — de quoi trancher.

Méthode
-------
Fenêtre EXPANSIVE : pour chaque repli, on entraîne sur tout ce qui précède et on
évalue sur le repli suivant. Un embargo de `HORIZON_MAX` sépare les deux, sinon
la cible `target_h24` d'une ligne d'entraînement déborderait dans l'évaluation.
Les prédictions de tous les replis, concaténées, forment une série
hors-échantillon couvrant la quasi-totalité de l'historique.

Trois choix assumés, parce qu'ils conditionnent la validité
-----------------------------------------------------------
1. **Hyperparamètres FIXES**, repris d'un entraînement déjà fait, au lieu d'être
   re-cherchés à chaque repli. C'est ce qui rend l'exercice réalisable : Optuna
   représente >80 % du coût (~375 ajustements contre ~50 pour la phase finale).
   Le biais introduit est réel — ces hyperparamètres ont vu des données que
   certains replis évaluent — mais il est IDENTIQUE pour tous les modèles
   comparés, donc il ne fausse pas une comparaison entre variantes. Il
   interdirait en revanche d'annoncer une performance absolue.

2. **Ancre, alpha et poids saisonnier choisis DANS chaque repli**, sur sa propre
   tranche de validation. Ce sont les paramètres les plus sensibles au régime de
   données (mesuré : `seasonal_24` retenue sur 0 horizon à 3,58 ans, sur 13 à
   5,35 ans) — les figer aurait transporté un choix d'une époque à une autre.

3. **Origines de production uniquement** (23 h locales). Évaluer toutes les
   heures mélangerait l'effet d'échéance et l'effet d'heure de la journée : à
   une origine de 03 h, l'horizon 12 vise 15 h, à une origine de 23 h il vise
   11 h. Les MAE par horizon deviendraient ininterprétables.

Reprise sur incident
--------------------
Chaque repli écrit ses prédictions sur disque et les replis déjà présents sont
sautés. Les entraînements de ce dépôt ont échoué plusieurs fois en `MemoryError`
sur cette machine ; perdre 5 heures de calcul pour la dernière allocation d'un
repli serait inacceptable.

Usage :
    python -m scripts.rolling_origin_cv --verifier
    python -m scripts.rolling_origin_cv --replis 6
    python -m scripts.rolling_origin_cv --sans-prevision   # variante à comparer
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, time as heure, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightgbm import LGBMRegressor, early_stopping, log_evaluation  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

from pipeline_prevision.db import get_observations  # noqa: E402
from pipeline_prevision.logging.logger import logging  # noqa: E402
from pipeline_prevision.utils.main_utils.feature_engineering import (  # noqa: E402
    ALPHA_GRID, ANCHOR_NAMES, HORIZON_MAX, SEASONAL_WEIGHT_GRID, anchor_values,
    build_features_for_target, build_series_by_target, complementary_anchor,
    select_forecast_temperature, select_temperature, target_prefix,
)

PARIS = ZoneInfo("Europe/Paris")
ORIGIN_HOUR_PARIS = 23          # = dags/forecast_daily_dag.py
CIBLE_DEFAUT = "consommation_totale"
SORTIE_DEFAUT = os.path.join("docs", "rolling_cv")

# Part de la fin de chaque bloc d'entraînement réservée au choix de l'ancre,
# d'alpha et du poids saisonnier. Contiguë et postérieure au train : mêmes
# conditions que la validation du pipeline principal.
PART_VALIDATION = 0.15
EARLY_STOPPING = 100


def origines_production(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Horodatages UTC correspondant à 23 h heure de Paris.

    Calculé par conversion et non par `hour == 21` : l'heure UTC de l'origine
    change entre l'été (21 h) et l'hiver (22 h), et un filtre figé perdrait la
    moitié de l'année.
    """
    local = index.tz_localize("UTC").tz_convert(PARIS)
    return index[local.hour == ORIGIN_HOUR_PARIS]


def hyperparametres(chemin: str | None, cible: str) -> dict:
    """Hyperparamètres repris d'un `metadata.json` d'entraînement."""
    if chemin and os.path.isfile(chemin):
        meta = json.load(open(chemin, encoding="utf-8"))
        h = meta["per_target"][cible]["hyperparameters"]
        h = json.loads(h) if isinstance(h, str) else h
        print(f"Hyperparamètres repris de {chemin}")
    else:
        # Valeurs mesurées sur `consommation_totale` (run 5,35 ans) : sert de
        # repli explicite plutôt que de laisser LightGBM à ses défauts, qui
        # n'ont rien à voir avec ce problème.
        h = {"learning_rate": 0.0386, "num_leaves": 31, "max_depth": 7,
             "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
             "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0}
        print("Hyperparamètres par défaut (aucun metadata fourni)")
    h.update({"objective": "regression_l1", "boosting_type": "gbdt",
              "n_estimators": 2000, "random_state": 42, "n_jobs": -1, "verbosity": -1})
    return h


def entrainer_repli(cadre: pd.DataFrame, colonnes: list[str], cibles: list[str],
                    prefix: str, params: dict,
                    fin_train: pd.Timestamp, debut_eval: pd.Timestamp,
                    fin_eval: pd.Timestamp, origines: pd.DatetimeIndex) -> pd.DataFrame:
    """Entraîne sur tout ce qui précède `fin_train`, prédit sur [debut_eval, fin_eval].

    L'embargo est porté par l'appelant (`debut_eval > fin_train + HORIZON_MAX`).
    """
    delta_col = f"{prefix}_delta_1"
    bloc = cadre.loc[:fin_train]
    coupe = int(len(bloc) * (1 - PART_VALIDATION))
    # Embargo interne aussi : la validation ne doit pas voir les cibles du train.
    train, valid = bloc.iloc[:coupe - HORIZON_MAX], bloc.iloc[coupe:]
    if min(len(train), len(valid)) < 500:
        raise ValueError(f"bloc trop court (train={len(train)}, valid={len(valid)})")

    eval_idx = origines[(origines >= debut_eval) & (origines <= fin_eval)]
    eval_idx = eval_idx.intersection(cadre.index)
    if len(eval_idx) == 0:
        return pd.DataFrame()

    from pipeline_prevision.utils.main_utils.feature_engineering import add_target_features
    lignes = []
    for horizon in range(1, HORIZON_MAX + 1):
        cible_h = f"target_h{horizon}"
        # Construites UNE fois par horizon, hors de la boucle d'ancres : le
        # cadre de features ne dépend pas de l'ancre, seule la cible résiduelle
        # en dépend.
        X_tr = add_target_features(train[colonnes], horizon, delta_col)
        X_va = add_target_features(valid[colonnes], horizon, delta_col)
        y_va = valid[cible_h].to_numpy()

        meilleur = (np.inf, "persistence", 1.0, 0.0, None)
        for ancre in ANCHOR_NAMES:
            base_tr = anchor_values(train, horizon, prefix, ancre)
            base_va = anchor_values(valid, horizon, prefix, ancre)
            melange_va = anchor_values(valid, horizon, prefix, complementary_anchor(ancre))

            modele = LGBMRegressor(**params)
            modele.fit(X_tr, train[cible_h].to_numpy() - base_tr,
                       eval_set=[(X_va, y_va - base_va)], eval_metric="mae",
                       callbacks=[early_stopping(EARLY_STOPPING, first_metric_only=True,
                                                 verbose=False), log_evaluation(period=0)])
            it = modele.best_iteration_ or params["n_estimators"]
            res_va = modele.predict(X_va, num_iteration=it)
            for a in ALPHA_GRID:
                direct = base_va + a * res_va
                for w in SEASONAL_WEIGHT_GRID:
                    score = mean_absolute_error(y_va, (1 - w) * direct + w * melange_va)
                    if score < meilleur[0]:
                        meilleur = (score, ancre, float(a), float(w), int(it))
        _, ancre, alpha, poids, it = meilleur

        # Réentraînement sur train+valid, comme le pipeline principal.
        # X_all est obtenu en EMPILANT X_tr et X_va plutôt qu'en rappelant
        # `add_target_features` : la fonction ne fait que dériver des colonnes
        # ligne à ligne, donc l'empilement donne le même résultat pour une
        # fraction du coût — et surtout sans allouer une troisième copie du
        # cadre complet, ce qui faisait échouer le premier essai.
        X_all = pd.concat([X_tr, X_va])
        y_all = np.concatenate([train[cible_h].to_numpy(), valid[cible_h].to_numpy()])
        base_all = np.concatenate([
            anchor_values(train, horizon, prefix, ancre),
            anchor_values(valid, horizon, prefix, ancre)])
        final = LGBMRegressor(**{**params, "n_estimators": it})
        final.fit(X_all, y_all - base_all)

        eval_frame = cadre.loc[eval_idx]
        X_ev = add_target_features(eval_frame[colonnes], horizon, delta_col)
        base_ev = anchor_values(eval_frame, horizon, prefix, ancre)
        melange_ev = anchor_values(eval_frame, horizon, prefix, complementary_anchor(ancre))
        direct = base_ev + alpha * final.predict(X_ev)
        y_pred = np.maximum((1 - poids) * direct + poids * melange_ev, 0.0)

        lignes.append(pd.DataFrame({
            "origine": eval_idx, "horizon": horizon,
            "cible_ts": eval_idx + pd.Timedelta(hours=horizon),
            "y_pred": y_pred, "y_true": eval_frame[cible_h].to_numpy(),
            "ancre": ancre, "alpha": alpha}))
        del final, X_tr, X_va, X_all, X_ev
        gc.collect()

    return pd.concat(lignes, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cible", default=CIBLE_DEFAUT)
    p.add_argument("--replis", type=int, default=6)
    p.add_argument("--part-initiale", type=float, default=0.40,
                   help="fraction de l'historique réservée au premier entraînement")
    p.add_argument("--params", help="metadata.json d'où reprendre les hyperparamètres")
    p.add_argument("--sortie", default=SORTIE_DEFAUT)
    p.add_argument("--sans-prevision", action="store_true",
                   help="variante SANS les features de température prévue (comparaison)")
    p.add_argument("--repli", type=int,
                   help="n'exécute QUE ce repli (1-based) puis sort. Permet de "
                        "piloter un processus par repli : Python ne rend jamais "
                        "totalement la mémoire au système, et sur une machine "
                        "contrainte l'accumulation entre replis finit par tuer le "
                        "run. Un processus par repli garantit un état propre.")
    p.add_argument("--data",
                   help="lit les observations depuis ce CSV au lieu de la base. "
                        "Le contenu est identique (datasets/data.csv est régénéré "
                        "depuis `observations`), mais s'en passer permet d'éteindre "
                        "PostgreSQL — et donc la VM WSL, qui retient ~3 Go que "
                        "l'arrêt des seuls conteneurs ne rend pas.")
    p.add_argument("--verifier", action="store_true")
    args = p.parse_args()

    if args.data:
        obs = pd.read_csv(args.data, sep=None, engine="python",
                          parse_dates=["timestamp"], index_col="timestamp").sort_index()
        obs = obs[~obs.index.duplicated(keep="last")].asfreq("h")
        print(f"Source     : {args.data} (hors base)")
    else:
        obs = get_observations().sort_index()
    temp = select_temperature(obs)
    temp_prev = None if args.sans_prevision else select_forecast_temperature(obs)
    prefix = target_prefix(args.cible)

    cadre, _, cibles = build_features_for_target(
        build_series_by_target(obs), temp, args.cible, temp_prev=temp_prev)
    colonnes = [c for c in cadre.columns if c not in cibles]
    cadre = cadre.replace([np.inf, -np.inf], np.nan).dropna(subset=colonnes + cibles)
    # float32 : divise par deux l'empreinte du cadre ET de chaque copie produite
    # par `add_target_features` (une par horizon). LightGBM ramène de toute façon
    # ses entrées en float32 en interne, et sur des puissances en MW la précision
    # reste de l'ordre du centième de MW — sans effet sur une MAE de ~800 MW.
    # C'est ce qui fait tenir la validation glissante sur cette machine.
    cadre = cadre.astype(np.float32)

    origines = origines_production(cadre.index)
    n = len(cadre)
    debut_eval_global = cadre.index[int(n * args.part_initiale)]
    bornes = pd.date_range(debut_eval_global, cadre.index[-1], periods=args.replis + 1)

    variante = "sans_prevision" if args.sans_prevision else "avec_prevision"
    dossier = os.path.join(args.sortie, f"{args.cible}_{variante}")
    print(f"Jeu        : {n:,} lignes, {cadre.index.min():%Y-%m-%d} -> "
          f"{cadre.index.max():%Y-%m-%d}".replace(",", " "))
    print(f"Variante   : {variante} ({len(colonnes)} features)")
    print(f"Origines   : {len(origines)} à 23 h locales")
    print(f"Replis     : {args.replis}, évaluation à partir du {debut_eval_global:%Y-%m-%d}")
    print(f"Sortie     : {dossier}")
    for i in range(args.replis):
        deb, fin = bornes[i], bornes[i + 1]
        nb = len(origines[(origines >= deb) & (origines <= fin)])
        print(f"  repli {i+1} : évalue {deb:%Y-%m-%d} -> {fin:%Y-%m-%d}  ({nb} origines)")
    if args.verifier:
        print("\n--verifier : aucun entraînement.")
        return

    os.makedirs(dossier, exist_ok=True)
    params = hyperparametres(args.params, args.cible)
    t0 = time.time()
    a_faire = [args.repli - 1] if args.repli else range(args.replis)
    for i in a_faire:
        chemin = os.path.join(dossier, f"repli_{i+1:02d}.parquet")
        if os.path.isfile(chemin):
            print(f"  repli {i+1}/{args.replis} déjà présent — sauté")
            continue
        deb, fin = bornes[i], bornes[i + 1]
        fin_train = deb - pd.Timedelta(hours=HORIZON_MAX)   # embargo
        t = time.time()
        pred = entrainer_repli(cadre, colonnes, cibles, prefix, params,
                               fin_train, deb, fin, origines)
        if pred.empty:
            print(f"  repli {i+1}/{args.replis} : aucune origine — ignoré")
            continue
        pred.to_parquet(chemin)
        mae = float((pred["y_pred"] - pred["y_true"]).abs().mean())
        print(f"  repli {i+1}/{args.replis} : {len(pred):5} points, MAE {mae:7.1f} MW, "
              f"{time.time()-t:.0f}s  -> {os.path.basename(chemin)}")
        gc.collect()

    print(f"\nTerminé en {time.time()-t0:.0f} s. "
          f"Agréger avec `python -m scripts.analyse_rolling_cv`.")


if __name__ == "__main__":
    main()
