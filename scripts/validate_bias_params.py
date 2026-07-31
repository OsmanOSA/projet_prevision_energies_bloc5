"""Validation empirique d'une correction de biais en ligne (λ, κ, c, W).

`local_forecaster` a longtemps appliqué une correction de niveau EWMA dont les
quatre paramètres, seuls du système, n'avaient jamais été estimés :
contrairement à `alpha` et `seasonal_weight`, choisis par grid-search sur la
partition de validation (cf. `model_trainer._train_horizons`), ils avaient été
posés à la main. Ce script les a traités comme les autres — et la correction a
échoué le kill-test sur les 5 cibles, ce qui a conduit à la retirer.

Il reste ici comme GARDE-FOU : à rejouer avant toute réintroduction d'une
correction en ligne, et après un retrain qui changerait le profil d'erreur.

Protocole
---------
Le correcteur de biais est une transformation POST-HOC de la paire
(y_true, y_pred_brut) : le valider ne demande donc aucun réentraînement,
seulement des prédictions hors-échantillon. Le modèle livré ayant été
réajusté sur train+valid, la seule partition propre est `test` — on la coupe
en deux dans le temps :

    test[:50 %]  -> SELECTION   : c'est là qu'on choisit (λ, κ, c, W)
    test[50 %:]  -> CONFIRMATION: jamais regardée pendant la sélection

La bande walk-forward est calculée sur tout `test` d'un seul tenant (comme
`backtest_direct`) puis découpée au scoring : sinon l'EWMA de la moitié de
confirmation repartirait de zéro.

Kill-test
---------
L'hypothèse nulle est « ne rien corriger » (κ = 0), présente dans la grille.
Une correction n'est adoptée pour une cible que si elle bat κ = 0 :
  1. sur SELECTION,
  2. ET sur CONFIRMATION,
  3. ET avec un intervalle bootstrap par blocs qui exclut 0.
Sinon le verdict est κ = 0 pour cette cible. Le script ne modifie aucun
fichier : il imprime un tableau et écrit un JSON de résultats.

Usage
-----
    python -m scripts.validate_bias_params
    python -m scripts.validate_bias_params --targets SOLAR WIND_ONSHORE
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.components.data_transformation import (  # noqa: E402
    TRAIN_RATIO, VALID_RATIO,
)
from pipeline_prevision.utils.main_utils.feature_engineering import (  # noqa: E402
    HORIZON_MAX, TARGET_PREFIXES, build_features_for_target, build_series_by_target,
    select_temperature, select_forecast_temperature,
)
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (  # noqa: E402
    _direct_prediction, get_anchor, get_models,
)

# Anciennes constantes `BIAS_*` de `local_forecaster`, conservées ici comme
# CANDIDATES à tester -- plus comme configuration de production. Le module de
# prévision n'applique aujourd'hui aucune correction de niveau : ce script est
# le garde-fou à rejouer avant d'envisager d'en réintroduire une.
BIAS_LAMBDA = 0.99
BIAS_SHRINKAGE = 0.5
BIAS_CAP_FRACTION = 0.05
BIAS_TYPICAL_WINDOW = 24 * 7

# --- Grille ----------------------------------------------------------------
# λ : demi-vie = ln(0.5)/ln(λ) heures -> 6.6 / 13.5 / 22.8 / 34.3 / 69.0 / 138.3
LAMBDA_GRID = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
# κ = 0 est le test nul : « ne rien corriger ». Il DOIT rester dans la grille.
KAPPA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
# c = np.inf : plafond désactivé, pour mesurer ce que le plafond coûte/rapporte.
CAP_GRID = [0.02, 0.05, 0.10, 0.20, np.inf]
WINDOW_GRID = [24, 168, 720]

CURRENT = (BIAS_LAMBDA, BIAS_SHRINKAGE, BIAS_CAP_FRACTION, BIAS_TYPICAL_WINDOW)

BOOTSTRAP_DRAWS = 2000
BLOCK_DAYS = 7          # blocs mobiles de 7 j : absorbe l'autocorrélation hebdo
RANDOM_STATE = 42


def half_life(lam: float) -> float:
    return float(np.log(0.5) / np.log(lam))


def load_test_partition(data_path: str, targets: list[str]) -> dict:
    """Rejoue exactement le découpage de `DataTransformation` et ne garde que
    la partition test (la seule que le modèle livré n'a jamais vue)."""
    full = pd.read_csv(data_path, sep=None, engine="python",
                       parse_dates=["timestamp"], index_col="timestamp")
    full = full.sort_index()
    full = full[~full.index.duplicated(keep="last")].asfreq("h")

    series_by_target = build_series_by_target(full)
    temp = select_temperature(full)
    temp_prev = select_forecast_temperature(full)

    out = {}
    for target in targets:
        features_df, prefix, target_columns = build_features_for_target(
            series_by_target, temp, target, temp_prev=temp_prev
        )
        feature_columns = [c for c in features_df.columns if c not in target_columns]
        features_df = (
            features_df.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=feature_columns + target_columns)
            .copy()
        )
        n = len(features_df)
        valid_cut = int(n * (TRAIN_RATIO + VALID_RATIO))
        out[target] = {
            "test": features_df.iloc[valid_cut:],
            "prefix": prefix,
            "feature_columns": feature_columns,
        }
    return out


def raw_predictions(entry: dict, composite: dict) -> dict:
    """Prévision finale AVANT correction de biais (modèle + alpha + saisonnier),
    pour les 24 horizons, sur la partition test."""
    test = entry["test"]
    feature_columns = entry["feature_columns"]
    prefix = composite["prefix"]
    delta_col = f"{prefix}_delta_1"
    frame = test[feature_columns]

    per_horizon = {}
    for horizon in range(1, HORIZON_MAX + 1):
        y_pred = _direct_prediction(
            composite["models"][horizon], composite["alphas"][horizon],
            composite["seasonal_weights"][horizon], frame, horizon, prefix, delta_col,
            get_anchor(composite, horizon),
        )
        per_horizon[horizon] = {
            "y_true": test[f"target_h{horizon}"].to_numpy(),
            "y_pred": y_pred,
            # index = horodatage CIBLE (origine + h), pour agréger par jour réel
            "target_ts": test.index + pd.to_timedelta(horizon, unit="h"),
        }
    return per_horizon


def bias_components(y_true: np.ndarray, y_pred: np.ndarray, index: pd.DatetimeIndex,
                    lam: float, window: int, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Les deux termes coûteux de la correction, calculés une fois par (λ, W).

    `shift` est le décalage, EN ORIGINES, appliqué avant l'EWMA :

      shift=1  reproduit l'ancien `_walkforward_bias_correction` (qui fuyait) ;
      shift=h  est le seul décalage réellement causal.

    Les séries sont indexées par ORIGINE (`y_true` vient de
    `series.shift(-h)`), donc l'erreur de l'origine `i-k` n'est observable
    qu'à l'instant `i-k+h`. Pour qu'une prévision émise à l'origine `i`
    n'utilise que du réalisé connu à `i`, il faut `i-k+h <= i`, c'est-à-dire
    `k >= h`. Avec `shift=1` et `h=24`, la correction lit un réalisé qui
    n'arrivera que 23 h plus tard.
    """
    error = pd.Series(y_true - y_pred, index=index)
    beta = error.shift(shift).ewm(alpha=1 - lam, adjust=False).mean().fillna(0.0)
    tau = (pd.Series(y_true, index=index).shift(shift).abs()
           .rolling(window, min_periods=24).mean().fillna(0.0))
    return beta.to_numpy(), tau.to_numpy()


def apply_correction(beta: np.ndarray, tau: np.ndarray, kappa: float, cap: float) -> np.ndarray:
    if kappa == 0.0:
        return np.zeros_like(beta)
    raw = kappa * beta
    if not np.isfinite(cap):
        return raw
    limit = cap * tau
    return np.clip(raw, -limit, limit)


def daily_mae(absolute_error: np.ndarray, day_codes: np.ndarray, n_days: int) -> np.ndarray:
    """MAE agrégée par jour : unité de rééchantillonnage du bootstrap.

    Les erreurs horaires sont fortement autocorrélées ; un bootstrap i.i.d.
    sur les points sous-estimerait grossièrement la variance et déclarerait
    significatif à peu près n'importe quoi. `bincount` plutôt qu'un `groupby`
    pandas : appelé ~10^4 fois dans la grille.
    """
    total = np.bincount(day_codes, weights=absolute_error, minlength=n_days)
    count = np.bincount(day_codes, minlength=n_days)
    return total / np.maximum(count, 1)


def block_bootstrap_ci(values: np.ndarray, draws: int = BOOTSTRAP_DRAWS,
                       block: int = BLOCK_DAYS, seed: int = RANDOM_STATE) -> tuple[float, float]:
    """IC 95 % de la moyenne de `values` (gain quotidien) par bootstrap à
    blocs mobiles — les blocs préservent l'autocorrélation intra-semaine."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < block * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(draws, -1)[:, :n]
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def evaluate_target(per_horizon: dict, split_ts: pd.Timestamp, causal: bool) -> dict:
    """Grid-search complet pour une cible : sélection sur la 1re moitié de
    test, confirmation sur la 2de. Les paramètres sont partagés par les 24
    horizons (un jeu par horizon multiplierait par 24 le risque de
    surapprentissage pour un gain qui reste à démontrer)."""
    # --- Pré-calculs invariants de la grille --------------------------------
    # Tout ce qui ne dépend pas de (λ, κ, c, W) est calculé une seule fois :
    # la grille visite 312 configurations x 24 horizons.
    all_days = np.unique(np.concatenate(
        [d["target_ts"].normalize().to_numpy() for d in per_horizon.values()]
    ))
    day_lookup = {day: position for position, day in enumerate(all_days)}
    n_days = len(all_days)
    split_day = np.searchsorted(all_days, split_ts.normalize().to_numpy())

    static = {}
    for horizon, data in per_horizon.items():
        index = data["target_ts"]
        y_base = np.maximum(data["y_pred"], 0.0)
        day_codes = np.array([day_lookup[d] for d in index.normalize().to_numpy()])
        static[horizon] = {
            "y_true": data["y_true"],
            "y_base": y_base,
            "abs_base": np.abs(data["y_true"] - y_base),
            "day_codes": day_codes,
            "mask_sel": np.asarray(index < split_ts),
            "daily_base": daily_mae(np.abs(data["y_true"] - y_base), day_codes, n_days),
        }

    cache = {}
    for horizon, data in per_horizon.items():
        shift = horizon if causal else 1
        for lam in LAMBDA_GRID:
            for window in WINDOW_GRID:
                cache[(horizon, lam, window)] = bias_components(
                    data["y_true"], data["y_pred"], data["target_ts"], lam, window, shift
                )

    def score(lam, kappa, cap, window):
        """MAE + gain quotidien apparié + taux de correction nuisible,
        agrégés sur les 24 horizons."""
        sel_num = sel_den = conf_num = conf_den = 0.0
        harm_hits = harm_total = 0
        gain_daily = np.zeros(n_days)
        for horizon in per_horizon:
            s = static[horizon]
            beta, tau = cache[(horizon, lam, window)]
            y_corr = np.maximum(s["y_base"] + apply_correction(beta, tau, kappa, cap), 0.0)

            abs_corr = np.abs(s["y_true"] - y_corr)
            mask_sel = s["mask_sel"]
            sel_num += abs_corr[mask_sel].sum(); sel_den += int(mask_sel.sum())
            conf_num += abs_corr[~mask_sel].sum(); conf_den += int((~mask_sel).sum())

            # « nuisible » = la correction a éloigné la prévision du réel.
            harm_hits += int((abs_corr > s["abs_base"]).sum())
            harm_total += len(abs_corr)

            # gain > 0 = la correction aide (base - corrigé)
            gain_daily += s["daily_base"] - daily_mae(abs_corr, s["day_codes"], n_days)

        gain_daily /= len(per_horizon)
        return {
            "mae_sel": sel_num / sel_den,
            "mae_conf": conf_num / conf_den,
            "gain_daily_sel": gain_daily[:split_day],
            "gain_daily_conf": gain_daily[split_day:],
            "harm_rate": harm_hits / harm_total,
        }

    null = score(LAMBDA_GRID[0], 0.0, CAP_GRID[0], WINDOW_GRID[0])

    results = []
    for lam in LAMBDA_GRID:
        for kappa in KAPPA_GRID:
            if kappa == 0.0:
                continue
            for cap in CAP_GRID:
                for window in WINDOW_GRID:
                    if not np.isfinite(cap) and window != WINDOW_GRID[0]:
                        continue  # W n'a aucun effet sans plafond
                    s = score(lam, kappa, cap, window)
                    results.append({"lambda": lam, "kappa": kappa, "cap": cap,
                                    "window": window, **s})

    best = min(results, key=lambda r: r["mae_sel"])
    current_lambda, current_kappa, current_cap, current_window = CURRENT
    current = {"lambda": current_lambda, "kappa": current_kappa,
               "cap": current_cap, "window": current_window,
               **score(current_lambda, current_kappa, current_cap, current_window)}

    def verdict(entry):
        lo, hi = block_bootstrap_ci(entry["gain_daily_conf"])
        gain_sel = 100 * (null["mae_sel"] - entry["mae_sel"]) / null["mae_sel"]
        gain_conf = 100 * (null["mae_conf"] - entry["mae_conf"]) / null["mae_conf"]
        return {
            "lambda": entry["lambda"], "kappa": entry["kappa"],
            "cap": entry["cap"], "window": entry["window"],
            "half_life_h": half_life(entry["lambda"]),
            "mae_sel": entry["mae_sel"], "mae_conf": entry["mae_conf"],
            "gain_sel_pct": gain_sel, "gain_conf_pct": gain_conf,
            "ci95_daily_gain_conf": [lo, hi],
            "harm_rate": entry["harm_rate"],
            "passes_kill_test": bool(gain_sel > 0 and gain_conf > 0 and lo > 0),
        }

    return {
        "null_mae_sel": null["mae_sel"], "null_mae_conf": null["mae_conf"],
        "null_harm_rate": null["harm_rate"],
        "best": verdict(best),
        "current": verdict(current),
        "n_configs": len(results),
    }


def interval_coverage(per_horizon: dict, n_calib: int = 200, alpha: float = 0.05) -> dict:
    """Couverture empirique de l'intervalle conforme sur la partition test.

    Réplique `predict_with_conformal_intervals` : à l'origine `i`, la
    demi-largeur est le quantile 1-alpha des `n_calib` derniers résidus
    absolus DISPONIBLES — donc décalés de `h` origines, puisqu'un résidu
    d'origine `j` n'est connu qu'à `j+h`. Seule change la façon dont la
    correction `b` entre dans ces résidus :

      'sans'   : b = 0 -- c'est ce que fait le code aujourd'hui
      'actuel' : b décalé d'une origine (ancien comportement, avec fuite)
      'causal' : b décalé de h origines

    Un intervalle honnête doit couvrir ~95 %. En dessous, il est trop étroit :
    le dashboard annonce une incertitude plus faible que la réalité.
    """
    out = {}
    for label, causal in (("sans", None), ("actuel", False), ("causal", True)):
        inside_total = width_total = count = 0
        for horizon, data in per_horizon.items():
            y_true, y_pred = data["y_true"], data["y_pred"]
            index = data["target_ts"]

            if causal is None:
                correction = np.zeros_like(y_pred)
            else:
                beta, tau = bias_components(
                    y_true, y_pred, index, BIAS_LAMBDA, BIAS_TYPICAL_WINDOW,
                    horizon if causal else 1,
                )
                correction = apply_correction(beta, tau, BIAS_SHRINKAGE, BIAS_CAP_FRACTION)

            y_corr = np.maximum(y_pred + correction, 0.0)
            residual = pd.Series(np.abs(y_true - y_corr))
            half_width = (residual.shift(horizon)
                          .rolling(n_calib, min_periods=n_calib // 4)
                          .quantile(1 - alpha).to_numpy())

            usable = np.isfinite(half_width)
            lower = np.maximum(y_corr - half_width, 0.0)
            upper = y_corr + half_width
            inside_total += int(((y_true >= lower) & (y_true <= upper))[usable].sum())
            width_total += float((upper - lower)[usable].sum())
            count += int(usable.sum())

        out[label] = {"coverage": inside_total / count, "mean_width": width_total / count}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", default=list(TARGET_PREFIXES))
    parser.add_argument("--data", default=os.path.join("datasets", "data.csv"))
    parser.add_argument("--out", default=os.path.join("docs", "bias_validation.json"))
    parser.add_argument(
        "--causal", action="store_true",
        help="décale les erreurs de h origines (seule variante sans fuite) au lieu "
             "de 1, comme le faisait l'ancien `_walkforward_bias_correction`",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="mesure la couverture empirique de l'intervalle conforme au lieu "
             "de lancer le grid-search",
    )
    args = parser.parse_args()

    print(f"Chargement des features et de la partition test ({args.data})...")
    partitions = load_test_partition(args.data, args.targets)
    models = get_models()

    report = {}
    for target in args.targets:
        entry = partitions[target]
        n_test = len(entry["test"])
        split_ts = entry["test"].index[n_test // 2]
        print(f"\n=== {target} === test={n_test} origines "
              f"({entry['test'].index[0]:%Y-%m-%d} -> {entry['test'].index[-1]:%Y-%m-%d}), "
              f"coupure sélection/confirmation = {split_ts:%Y-%m-%d}")

        per_horizon = raw_predictions(entry, models[target])

        if args.coverage:
            report[target] = interval_coverage(per_horizon)
            for label, stats in report[target].items():
                print(f"  couverture [{label:7s}] = {stats['coverage']:6.2%} "
                      f"(nominal 95 %) | largeur moyenne = {stats['mean_width']:.1f}")
            continue

        result = evaluate_target(per_horizon, split_ts, causal=args.causal)
        result["shift_mode"] = "causal (h origines)" if args.causal else "actuel (1 origine)"
        report[target] = result

        null_mae = result["null_mae_conf"]
        for label in ("current", "best"):
            r = result[label]
            cap = "inf" if not np.isfinite(r["cap"]) else f"{r['cap']:.2f}"
            lo, hi = r["ci95_daily_gain_conf"]
            flag = "OK " if r["passes_kill_test"] else "NON"
            print(f"  [{label:7s}] lam={r['lambda']:.3f} (t1/2={r['half_life_h']:5.1f}h) "
                  f"kappa={r['kappa']:.2f} c={cap:>4s} W={r['window']:4d} | "
                  f"gain sel={r['gain_sel_pct']:+6.2f}% conf={r['gain_conf_pct']:+6.2f}% | "
                  f"IC95 gain/j=[{lo:+.3f},{hi:+.3f}] | nuis={r['harm_rate']:.1%} | kill-test {flag}")
        print(f"  [null   ] kappa=0 -> MAE confirmation = {null_mae:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\nRapport écrit dans {args.out}")


if __name__ == "__main__":
    main()
