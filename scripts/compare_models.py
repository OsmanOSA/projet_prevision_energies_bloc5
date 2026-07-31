"""Comparaison champion / challenger sur une fenêtre commune hors-échantillon.

Complète `scripts/retrain._backtest_mae`, qui décide la promotion sur les 30
dernières origines en agrégeant des MAE en MW bruts — donc dominées par
NUCLEAR (~46 GW) et la consommation (~54 GW), pendant que SOLAR et BIOMASS
ne pèsent presque rien. Un challenger qui abîme la biomasse et gratte un peu
sur le nucléaire y passe pour meilleur.

Ce script corrige trois points :

1. **Fenêtre commune explicite.** Deux modèles entraînés à des dates
   différentes ont des partitions test différentes ; comparer leurs
   `metadata.json` n'a aucun sens. Il faut une fenêtre postérieure aux données
   d'entraînement du champion ET incluse dans la partition test du challenger.
   `--start` / `--end` la fixent, et le script vérifie qu'elle est bien
   hors-échantillon pour le champion.
2. **Par cible, et normalisé.** Le verdict est rendu cible par cible, avec le
   gain vs persistance (sans unité) à côté de la MAE.
3. **Significativité.** Bootstrap à blocs mobiles de 7 j sur la différence
   quotidienne appariée de MAE — les erreurs horaires sont trop autocorrélées
   pour un test i.i.d.

Usage :
    python -m scripts.compare_models --start 2026-06-01 --end 2026-07-26
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
    HORIZON_MAX, TARGET_PREFIXES, build_origin_feature_frame, build_series_by_target,
    select_temperature, select_forecast_temperature,
)
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (  # noqa: E402
    _direct_prediction, get_anchor,
)

BLOCK_DAYS = 7
BOOTSTRAP_DRAWS = 2000
RANDOM_STATE = 42


def block_bootstrap_ci(values: np.ndarray, draws: int = BOOTSTRAP_DRAWS,
                       block: int = BLOCK_DAYS, seed: int = RANDOM_STATE):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < block * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def predictions_on_window(composite: dict, series_by_target: dict, temp: pd.Series,
                          target: str, start: pd.Timestamp, end: pd.Timestamp,
                          temp_prev: pd.Series | None = None) -> dict:
    """MAE par horizon d'un modèle sur les origines de [start, end].

    Reproduit exactement le chemin d'inférence de `local_forecaster` : modèle
    résiduel, alpha, poids saisonnier, plancher à 0 — et rien d'autre (la
    correction de biais a été retirée, cf. docs/ARCHITECTURE_MODELE.md §6).
    """
    entry = composite[target]
    feature_columns = entry["feature_columns"]
    prefix = entry["prefix"]
    delta_col = f"{prefix}_delta_1"

    features_df, _ = build_origin_feature_frame(series_by_target, temp, target,
                                                temp_prev=temp_prev)
    series = series_by_target[target]

    per_horizon = {}
    for horizon in range(1, HORIZON_MAX + 1):
        actual_future = series.shift(-horizon)
        mask = (features_df[feature_columns].notna().all(axis=1)
                & actual_future.notna()
                & (features_df.index >= start) & (features_df.index <= end))
        frame = features_df.loc[mask]
        if frame.empty:
            continue

        persistence = frame[f"{prefix}_0"].to_numpy()
        y_pred = _direct_prediction(
            entry["models"][horizon], entry["alphas"][horizon], entry["seasonal_weights"][horizon],
            frame[feature_columns], horizon, prefix, delta_col, get_anchor(entry, horizon),
        )

        per_horizon[horizon] = {
            "y_true": actual_future.loc[mask].to_numpy(),
            "y_pred": y_pred,
            "persistence": persistence,
            "target_ts": frame.index + pd.to_timedelta(horizon, unit="h"),
        }
    return per_horizon


def daily_series(per_horizon: dict, key: str) -> pd.Series:
    """MAE quotidienne moyennée sur les 24 horizons, indexée par jour cible."""
    frames = []
    for data in per_horizon.values():
        error = np.abs(data["y_true"] - data[key])
        frames.append(pd.Series(error, index=data["target_ts"]).groupby(
            pd.Series(data["target_ts"]).dt.normalize().to_numpy()).mean())
    return pd.concat(frames, axis=1).mean(axis=1)


def summarize(per_horizon: dict) -> dict:
    y_true = np.concatenate([d["y_true"] for d in per_horizon.values()])
    y_pred = np.concatenate([d["y_pred"] for d in per_horizon.values()])
    pers = np.concatenate([d["persistence"] for d in per_horizon.values()])
    mae = float(np.abs(y_true - y_pred).mean())
    mae_pers = float(np.abs(y_true - pers).mean())
    return {"mae": mae, "mae_persistence": mae_pers,
            "gain_pct": 100 * (mae_pers - mae) / mae_pers, "n": int(len(y_true))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", default="final_models")
    parser.add_argument("--challenger", default="candidate_models")
    parser.add_argument("--data", default=os.path.join("datasets", "data.csv"))
    parser.add_argument("--start", required=True, help="début de la fenêtre commune (AAAA-MM-JJ)")
    parser.add_argument("--end", required=True, help="fin de la fenêtre commune (AAAA-MM-JJ)")
    parser.add_argument("--out", default=os.path.join("docs", "model_comparison.json"))
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    full = pd.read_csv(args.data, sep=None, engine="python",
                       parse_dates=["timestamp"], index_col="timestamp").sort_index()
    full = full[~full.index.duplicated(keep="last")].asfreq("h")
    series_by_target = build_series_by_target(full)
    temp = select_temperature(full)
    # Sans les colonnes de prévision, un modèle entraîné avec les features de
    # température cible échouerait ici sur `_align_to_model` — la comparaison
    # doit reproduire le chemin d'inférence, features comprises.
    temp_prev = select_forecast_temperature(full)

    models = {}
    for label, path in (("champion", args.champion), ("challenger", args.challenger)):
        models[label] = load_object(os.path.join(path, "model.pkl"))
        meta_path = os.path.join(path, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            print(f"{label:11s} {meta['model_version']}  entraîné {meta['trained_at_utc'][:16]}")

    print(f"\nFenêtre commune : {start:%Y-%m-%d} -> {end:%Y-%m-%d}\n")
    header = f"{'cible':22s} {'champion':>20s} {'challenger':>20s} {'écart MAE':>11s} {'IC95 gain/j':>22s}"
    print(header)
    print("-" * len(header))

    report = {"window": [str(start), str(end)], "targets": {}}
    for target in TARGET_PREFIXES:
        ph = {label: predictions_on_window(models[label], series_by_target, temp,
                                           target, start, end, temp_prev=temp_prev)
              for label in ("champion", "challenger")}
        if not ph["champion"] or not ph["challenger"]:
            print(f"{target:22s} fenêtre vide — ignoré")
            continue

        stats = {label: summarize(ph[label]) for label in ph}
        # gain > 0 = le challenger fait mieux
        gain_daily = (daily_series(ph["champion"], "y_pred")
                      - daily_series(ph["challenger"], "y_pred")).dropna()
        low, high = block_bootstrap_ci(gain_daily.to_numpy())
        delta = 100 * (stats["champion"]["mae"] - stats["challenger"]["mae"]) / stats["champion"]["mae"]

        verdict = "gagne" if low > 0 else ("perd" if high < 0 else "nul")
        print(f"{target:22s} "
              f"{stats['champion']['mae']:9.1f} ({stats['champion']['gain_pct']:+5.1f}%) "
              f"{stats['challenger']['mae']:9.1f} ({stats['challenger']['gain_pct']:+5.1f}%) "
              f"{delta:+10.2f}% [{low:+8.1f},{high:+8.1f}] {verdict}")

        report["targets"][target] = {
            **{f"{label}_{k}": v for label, s in stats.items() for k, v in s.items()},
            "delta_mae_pct": delta, "ci95_daily_gain": [low, high], "verdict": verdict,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\nRapport écrit dans {args.out}")


if __name__ == "__main__":
    main()
