"""Rétro-alimente le couple Open-Meteo `temp_fr_om` / `temp_fr_prev`.

`init_db()` crée les deux colonnes, mais une migration additive laisse tout le
passé à NULL. Sans rétro-alimentation, les features de température cible sont
vides et le modèle reste aveugle à la météo future — l'angle mort qui nous a fait
perdre 15 % de MAE contre RTE par bascule thermique (canicule du 27-29/07/2026).

Les deux colonnes viennent de la MÊME grille Open-Meteo, et c'est le point
essentiel : la feature qui porte le signal est l'écart `temp_prev(cible) -
temp_om(origine)`. Mesuré sur 12 mois, le biais entre grille Open-Meteo et
stations Meteostat varie de 0,78 °C selon l'heure et de 0,63 °C selon le niveau de
température (+0,78 °C au-delà de 30 °C) : croiser les sources ferait apprendre cet
écart comme de la thermosensibilité. Cf. `prevision_temperature_france.py`.

CONTRAT DE VINTAGE. `temp_fr_prev` est la prévision telle qu'émise à J-1 et ne
doit JAMAIS être rafraîchie par une prévision de meilleure échéance : c'est ce qui
garantit l'absence de fuite à l'entraînement. Le script ne remplit donc que les
trous. `--remplacer` existe pour corriger une écriture fautive, pas pour
rafraîchir — il prévient avant d'agir.

Idempotent : relançable sans effet de bord.

Usage :
    python -m scripts.backfill_prevision_temperature
    python -m scripts.backfill_prevision_temperature --verifier     # sans écriture
    python -m scripts.backfill_prevision_temperature --diagnostic   # revalide l'absence de fuite
    python -m scripts.backfill_prevision_temperature --debut 2026-01-01
    python -m scripts.backfill_prevision_temperature --csv           # base -> datasets/data.csv
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.db import get_observations, init_db, log_run  # noqa: E402
from pipeline_prevision.db.config import get_engine  # noqa: E402
from pipeline_prevision.utils.main_utils.feature_engineering import (  # noqa: E402
    MIN_TEMPERATURE_COVERAGE,
)
from pipeline_prevision.utils.main_utils.prevision_temperature_france import (  # noqa: E402
    ARCHIVE_DEBUT, analyse_france, archive_prevision_france, diagnostic,
)

LOT = 5000
COLONNES = ("temp_fr_om", "temp_fr_prev")


def etat(obs: pd.DataFrame) -> None:
    """Couverture des deux colonnes, et conséquence sur `select_temperature`."""
    total = len(obs)
    print(f"Base : {total:,} lignes".replace(",", " "))
    for col in COLONNES:
        n = int(obs[col].notna().sum()) if col in obs.columns else 0
        seuil = "OK" if n >= MIN_TEMPERATURE_COVERAGE * total else "sous le seuil"
        print(f"  {col:14} {n:>7,} ({100 * n / total:6.2f} %)  {seuil}"
              .replace(",", " "))
    print(f"Seuil de bascule de select_temperature : {MIN_TEMPERATURE_COVERAGE:.0%}")


def ecrire(colonne: str, valeurs: pd.Series, remplacer: bool) -> int:
    """UPDATE par lots. Un UPDATE par heure sur 31 000 lignes saturerait le
    journal de transaction et rendrait l'opération non interruptible."""
    if valeurs.empty:
        return 0
    # Sans --remplacer, la clause `IS NULL` rend l'écriture idempotente ET
    # protège le contrat de vintage : un rejeu ne peut pas substituer une
    # prévision plus fraîche à celle déjà archivée.
    garde = "" if remplacer else f" AND {colonne} IS NULL"
    requete = text(f"UPDATE observations SET {colonne} = :v WHERE ts = :t{garde}")

    engine = get_engine()
    ecrites = 0
    for debut in range(0, len(valeurs), LOT):
        lot = valeurs.iloc[debut:debut + LOT]
        with engine.begin() as conn:
            conn.execute(requete, [{"v": float(v), "t": pd.Timestamp(t).to_pydatetime()}
                                   for t, v in lot.items()])
        ecrites += len(lot)
        print(f"    {colonne} : {ecrites:,}/{len(valeurs):,}".replace(",", " "), end="\r")
    print()
    return ecrites


def exporter_csv(obs: pd.DataFrame, chemin: str, simulation: bool) -> None:
    """Recopie `temp_fr_om` / `temp_fr_prev` de la base vers le CSV d'entraînement.

    L'entraînement lit `datasets/data.csv` (cf. `data_ingestion.py`) tandis que
    l'inférence et l'évaluation de promotion lisent la base : sans cette étape,
    les features de température cible existent d'un côté et pas de l'autre, et le
    modèle réentraîné reste aveugle à la météo future. Le dépôt a déjà rencontré
    exactement cet écart lors de l'introduction de `temp_fr`.

    La copie part de la BASE et non d'un nouvel appel à l'API : c'est ce qui
    garantit des valeurs identiques de part et d'autre. Un second appel pourrait
    renvoyer des valeurs légèrement différentes si Open-Meteo a rejoué un run.
    """
    if not os.path.isfile(chemin):
        sys.exit(f"{chemin} introuvable.")

    df = pd.read_csv(chemin, sep=None, engine="python",
                     parse_dates=["timestamp"], index_col="timestamp").sort_index()
    print(f"CSV : {len(df):,} lignes, {df.index.min():%Y-%m-%d} -> "
          f"{df.index.max():%Y-%m-%d}".replace(",", " "))

    for col in COLONNES:
        en_base = int(obs[col].notna().sum())
        aligne = obs[col].reindex(df.index)
        print(f"  {col:14} base {en_base:>7,} · alignées sur le CSV "
              f"{int(aligne.notna().sum()):>7,} / {len(df):,}".replace(",", " "))
        if aligne.notna().sum() < MIN_TEMPERATURE_COVERAGE * len(df):
            print(f"     /!\\ couverture sous {MIN_TEMPERATURE_COVERAGE:.0%} : "
                  "les features resteraient désactivées à l'entraînement. "
                  "Lancer le backfill en base d'abord.")

    if simulation:
        print("\n--verifier : le CSV n'a pas été modifié.")
        return

    sauvegarde = f"{chemin}.bak_pre_prevision_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(chemin, sauvegarde)
    for col in COLONNES:
        df[col] = obs[col].reindex(df.index)
    df.to_csv(chemin)
    print(f"\nSauvegarde : {sauvegarde}")
    print(f"Colonnes {', '.join(COLONNES)} écrites dans {chemin}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debut", help="AAAA-MM-JJ (défaut : première observation en base)")
    p.add_argument("--fin", help="AAAA-MM-JJ (défaut : dernière observation en base)")
    p.add_argument("--lead", type=int, default=1,
                   help="échéance de la prévision archivée, en jours (défaut 1)")
    p.add_argument("--csv", action="store_true",
                   help="recopie les deux colonnes de la base vers datasets/data.csv "
                        "(l'entraînement lit le CSV, l'inférence lit la base)")
    p.add_argument("--data", default=os.path.join("datasets", "data.csv"))
    p.add_argument("--verifier", action="store_true", help="diagnostic sans écriture")
    p.add_argument("--diagnostic", action="store_true",
                   help="revalide qu'`previous_dayN` est une vraie prévision, puis sort")
    p.add_argument("--remplacer", action="store_true",
                   help="écrase les valeurs existantes — VIOLE le contrat de vintage "
                        "sur temp_fr_prev, réservé à la correction d'une écriture fautive")
    args = p.parse_args()

    init_db()

    if args.diagnostic:
        # Open-Meteo peut changer ses modèles sous-jacents : ce contrôle vérifie
        # qu'`previous_dayN` reste une vraie prévision (erreur par site ~1 °C à
        # J-1, CROISSANTE avec l'échéance) et non une analyse déguisée, sur
        # laquelle on entraînerait un modèle qui déçoit en production.
        fin = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        debut = fin - pd.Timedelta(days=150)
        print(f"Diagnostic de l'absence de fuite, {debut:%Y-%m-%d} -> {fin:%Y-%m-%d}\n")
        resume = diagnostic(debut, fin)
        print(resume.to_string(index=False))
        print(f"\nRéduction théorique maximale par agrégation : "
              f"racine(17) = {resume.attrs['sqrt_n_theorique']}")
        r1 = float(resume.loc[resume["échéance"] == "J-1", "RMSE par site (°C)"].iloc[0])
        r3 = float(resume.loc[resume["échéance"] == "J-3", "RMSE par site (°C)"].iloc[0])
        if 0.9 <= r1 <= 2.2 and r3 > r1 * 1.15:
            print("\nVERDICT : vraie prévision (erreur par site cohérente et croissante "
                  "avec l'échéance). Entraînement sans fuite.")
        else:
            print(f"\nVERDICT : PROFIL INATTENDU (J-1={r1}, J-3={r3}). Ne pas "
                  "entraîner sur ces prévisions sans investigation.")
            sys.exit(2)
        return

    obs = get_observations()
    if obs is None or obs.empty:
        sys.exit("Aucune observation en base.")

    if args.csv:
        exporter_csv(obs, args.data, args.verifier)
        return

    etat(obs)

    debut = (pd.Timestamp(args.debut) if args.debut else obs.index.min()).to_pydatetime()
    fin = (pd.Timestamp(args.fin) if args.fin else obs.index.max()).to_pydatetime()
    if debut < ARCHIVE_DEBUT:
        print(f"\nL'archive Open-Meteo démarre au {ARCHIVE_DEBUT:%Y-%m-%d} : "
              f"les heures antérieures resteront NULL.")
    print(f"\nFenêtre demandée : {debut:%Y-%m-%d %H:%M} -> {fin:%Y-%m-%d %H:%M}")

    attendus = {}
    for col in COLONNES:
        manquants = obs.index[(obs[col].isna()) & (obs.index >= debut) & (obs.index <= fin)]
        attendus[col] = manquants
        print(f"  {col:14} à combler : {len(manquants):,} heures".replace(",", " "))

    if not args.remplacer and all(len(v) == 0 for v in attendus.values()):
        print("\nRien à rétro-alimenter.")
        return

    if args.verifier:
        print("\n--verifier : aucune écriture.")
        return

    if args.remplacer:
        print("\n/!\\ --remplacer : les valeurs existantes seront écrasées. Sur "
              "temp_fr_prev cela VIOLE le contrat de vintage et peut introduire "
              "une fuite à l'entraînement si la source n'est pas l'échéance J-1.")

    t0 = time.time()
    total_ecrit = 0
    try:
        print(f"\nAnalyse Open-Meteo (côté origine des features)")
        serie_om = analyse_france(debut, fin, verbeux=True)
        cible = obs.index if args.remplacer else attendus["temp_fr_om"]
        total_ecrit += ecrire("temp_fr_om", serie_om.reindex(cible).dropna(), args.remplacer)

        print(f"\nPrévision Open-Meteo archivée à J-{args.lead} (côté cible, sans fuite)")
        serie_prev = archive_prevision_france(debut, fin, lead_jours=args.lead, verbeux=True)
        cible = obs.index if args.remplacer else attendus["temp_fr_prev"]
        total_ecrit += ecrire("temp_fr_prev", serie_prev.reindex(cible).dropna(), args.remplacer)

        duree = time.time() - t0
        log_run("backfill_prevision_temperature", "success", rows=total_ecrit,
                duration_s=duree,
                message=f"temp_fr_om + temp_fr_prev (J-{args.lead}) · "
                        f"{total_ecrit} valeurs · {debut:%Y-%m-%d}->{fin:%Y-%m-%d}")
    except Exception as e:
        log_run("backfill_prevision_temperature", "failed", duration_s=time.time() - t0,
                message=str(e)[:500])
        raise

    print(f"\n{total_ecrit:,} valeurs écrites en {time.time() - t0:.0f} s\n"
          .replace(",", " "))
    etat(get_observations())


if __name__ == "__main__":
    main()
