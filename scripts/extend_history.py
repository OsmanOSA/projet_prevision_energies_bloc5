"""Étend l'historique de `observations` vers le passé.

Pourquoi
--------
Les catégories de jours rares — fériés, lundis, ponts — sont trop peu nombreuses
pour être apprises comme pour être évaluées. Sur 3,58 ans, l'historique ne
contient que ~39 jours fériés, dont 6 seulement tombent dans la fenêtre de test :
aucune conclusion par catégorie n'y est atteignable.

Jusqu'où
--------
La borne n'est PAS le choix de l'utilisateur, c'est une contrainte de source :

| série                          | profondeur disponible        |
|--------------------------------|------------------------------|
| RTE consommation / production  | >= 9 ans (2017 répond)       |
| Meteostat (`temp_fr`)          | décennies                    |
| Open-Meteo analyse (`temp_fr_om`) | >= 2016                   |
| **Open-Meteo prévision J-1 (`temp_fr_prev`)** | **débute ~2021-03-25** |

`temp_fr_prev` porte l'intégralité du gain de +15,7 % établi par l'ablation.
Remonter au-delà de mars 2021 la laisserait vide sur la partie ancienne, et le
seuil de couverture de `select_forecast_temperature` (95 %) désactiverait alors
les features de température cible sur TOUT le jeu — on perdrait le gain pour
gagner des lignes. D'où `DEBUT_ARCHIVE_PREVISION`, qui plafonne la demande.

Idempotent : l'upsert protège l'existant (COALESCE), donc un rejeu ne détruit
rien et complète seulement les trous.

Usage :
    python -m scripts.extend_history --verifier
    python -m scripts.extend_history
    python -m scripts.extend_history --debut 2022-01-01
"""

import argparse
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.db import get_observations, init_db, log_run, upsert_observations  # noqa: E402
from pipeline_prevision.utils.main_utils.utils import concat_all_data  # noqa: E402

# Début de l'archive des prévisions Open-Meteo, mesuré : vide au 2021-03-15,
# complète au 2021-03-25. On prend la première date sûre.
DEBUT_ARCHIVE_PREVISION = date(2021, 3, 25)

# Largeur d'une passe. RTE découpe déjà en interne (SIX_MONTHS / FIVE_MONTHS),
# mais découper ici aussi donne une progression visible et borne la mémoire :
# `concat_all_data` assemble tout en RAM avant l'upsert.
JOURS_PAR_PASSE = 90


# Ordre des colonnes du CSV d'entraînement, tel que le lit `DataIngestion`.
# La validation contrôle la présence, pas l'ordre — mais un fichier régénéré doit
# rester lisible à l'œil et diffable contre le précédent.
COLONNES_CSV = ["production_total", "consommation_totale", "temp",
                "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR",
                "temp_fr", "temp_fr_om", "temp_fr_prev"]

# Colonnes qui doivent être renseignées pour qu'une ligne soit exploitable.
# `temp_fr_prev` en est exclue : son trou d'archive de 492 h est toléré par
# `data_validation.MISSING_TOLERANCE`, alors qu'un trou de consommation ou de
# température d'origine ferait échouer la validation.
COLONNES_REQUISES = ["production_total", "consommation_totale", "temp",
                     "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR",
                     "temp_fr", "temp_fr_om"]


def exporter_csv(obs: pd.DataFrame, chemin: str, simulation: bool) -> None:
    """Reconstruit `datasets/data.csv` depuis la base, historique étendu compris.

    `backfill_prevision_temperature --csv` ne fait que remplir des colonnes sur
    l'index existant du CSV : il ne peut pas l'ALLONGER. Après extension de
    l'historique, il faut donc régénérer le fichier, sinon l'entraînement
    continuerait de lire les 3,58 ans d'origine pendant que la base en contient
    5,35 — l'écart train/base que ce dépôt a déjà connu sur `temp_fr`.

    Les lignes de queue incomplètes sont écartées : RTE publie avec ~1,3 h de
    retard, donc les dernières heures arrivent partiellement vides. Les laisser
    ferait échouer la validation, qui n'admet des NaN que sur les colonnes à
    tolérance déclarée.
    """
    if obs is None or obs.empty:
        sys.exit("Aucune observation en base.")

    manquantes = [c for c in COLONNES_CSV if c not in obs.columns]
    if manquantes:
        sys.exit(f"Colonnes absentes de la base : {manquantes}")

    df = obs[COLONNES_CSV].copy()
    df.index.name = "timestamp"

    complet = df[COLONNES_REQUISES].notna().all(axis=1)
    if not complet.any():
        sys.exit("Aucune ligne complète.")
    # On coupe à la dernière ligne complète plutôt que de filtrer les
    # incomplètes : un trou INTERNE doit rester visible (et faire échouer la
    # validation), seule la queue en cours de publication est légitimement absente.
    df = df.loc[:complet[complet].index.max()]
    restants = int((~df[COLONNES_REQUISES].notna().all(axis=1)).sum())

    print(f"Base   : {len(obs):,} lignes, {obs.index.min()} -> {obs.index.max()}"
          .replace(",", " "))
    print(f"À écrire: {len(df):,} lignes, {df.index.min()} -> {df.index.max()}"
          .replace(",", " "))
    print(f"          ({len(df) / 8766:.2f} ans ; {restants} ligne(s) incomplète(s) "
          "à l'intérieur)")
    for col in ("temp_fr", "temp_fr_om", "temp_fr_prev"):
        n = int(df[col].notna().sum())
        print(f"          {col:14} {n:>7,} / {len(df):,} ({100*n/len(df):6.2f} %)"
              .replace(",", " "))

    if os.path.isfile(chemin):
        ancien = pd.read_csv(chemin, sep=None, engine="python",
                             parse_dates=["timestamp"], index_col="timestamp")
        print(f"CSV actuel : {len(ancien):,} lignes, {ancien.index.min()} -> "
              f"{ancien.index.max()}".replace(",", " "))
        print(f"Gain       : +{len(df) - len(ancien):,} lignes".replace(",", " "))

    if simulation:
        print("\n--verifier : le CSV n'a pas été modifié.")
        return

    if os.path.isfile(chemin):
        sauvegarde = f"{chemin}.bak_pre_extension_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copy2(chemin, sauvegarde)
        print(f"\nSauvegarde : {sauvegarde}")
    df.to_csv(chemin)
    print(f"Écrit : {chemin}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debut", help=f"AAAA-MM-JJ (défaut {DEBUT_ARCHIVE_PREVISION})")
    p.add_argument("--fin", help="AAAA-MM-JJ (défaut : première observation en base)")
    p.add_argument("--verifier", action="store_true", help="diagnostic sans écriture")
    p.add_argument("--csv", action="store_true",
                   help="reconstruit datasets/data.csv depuis la base, sans ingérer")
    p.add_argument("--data", default=os.path.join("datasets", "data.csv"))
    args = p.parse_args()

    if args.csv:
        init_db()
        exporter_csv(get_observations(), args.data, args.verifier)
        return

    init_db()
    obs = get_observations()
    if obs is None or obs.empty:
        sys.exit("Aucune observation en base : utiliser d'abord l'ingestion normale.")

    debut = date.fromisoformat(args.debut) if args.debut else DEBUT_ARCHIVE_PREVISION
    if debut < DEBUT_ARCHIVE_PREVISION:
        print(f"/!\\ {debut} précède le début de l'archive des prévisions "
              f"({DEBUT_ARCHIVE_PREVISION}) : `temp_fr_prev` y sera vide, ce qui "
              f"désactiverait les features de température cible. Borne relevée.")
        debut = DEBUT_ARCHIVE_PREVISION
    fin = date.fromisoformat(args.fin) if args.fin else obs.index.min().date()

    print(f"Base actuelle : {len(obs):,} lignes, {obs.index.min()} -> {obs.index.max()}"
          .replace(",", " "))
    if debut >= fin:
        print(f"\nRien à étendre : la base démarre déjà au {fin}.")
        return

    passes = []
    curseur = debut
    while curseur < fin:
        suivant = min(curseur + timedelta(days=JOURS_PAR_PASSE), fin)
        passes.append((curseur, suivant))
        curseur = suivant

    jours = (fin - debut).days
    print(f"À étendre     : {debut} -> {fin}  ({jours} jours, {len(passes)} passes)")
    print(f"Gain attendu  : {jours / 365.25:.2f} an d'historique en plus "
          f"(~{round(jours / 365.25 * 11)} jours fériés, "
          f"~{round(jours / 7)} lundis)")
    if args.verifier:
        print("\n--verifier : aucune écriture.")
        return

    t0, total = time.time(), 0
    try:
        for i, (a, b) in enumerate(passes, 1):
            df = concat_all_data(a.isoformat(), b.isoformat())
            n = upsert_observations(df) if df is not None and not df.empty else 0
            total += n
            manquants = {c: int(df[c].isna().sum()) for c in
                         ("consommation_totale", "temp_fr", "temp_fr_om", "temp_fr_prev")
                         if c in df.columns}
            print(f"  passe {i}/{len(passes)}  {a} -> {b} : {n:5} lignes  "
                  f"NaN {manquants}")
        log_run("extend_history", "success", rows=total, duration_s=time.time() - t0,
                message=f"historique étendu {debut} -> {fin} · {total} lignes")
    except Exception as e:
        log_run("extend_history", "failed", duration_s=time.time() - t0,
                message=str(e)[:500])
        raise

    obs = get_observations()
    print(f"\n{total:,} lignes traitées en {time.time() - t0:.0f} s".replace(",", " "))
    print(f"Base désormais : {len(obs):,} lignes, {obs.index.min()} -> {obs.index.max()}"
          .replace(",", " "))
    for col in ("consommation_totale", "temp", "temp_fr", "temp_fr_om", "temp_fr_prev"):
        if col in obs.columns:
            n = int(obs[col].notna().sum())
            print(f"    {col:20} {n:>7,} / {len(obs):,}  ({100*n/len(obs):5.2f} %)"
                  .replace(",", " "))


if __name__ == "__main__":
    main()
