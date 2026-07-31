"""Construit la température France pondérée et l'ajoute à `datasets/data.csv`.

Ajoute une colonne `temp_fr` **sans toucher** à la colonne `temp` existante
(station unique) : les deux coexistent le temps de mesurer laquelle vaut le coup,
protocole du notebook `comparison_time_series_models.ipynb` à l'appui.

Usage :
    python -m scripts.build_temperature_france                  # toute la plage du CSV
    python -m scripts.build_temperature_france --debut 2026-01-01 --fin 2026-07-27
    python -m scripts.build_temperature_france --sortie-seule   # n'écrit pas le CSV
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.utils.main_utils.temperature_france import (  # noqa: E402
    POIDS_TOTAL, SEUIL_POIDS, VILLES, temperature_france,
)

CSV = os.path.join("datasets", "data.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=CSV)
    p.add_argument("--debut", help="AAAA-MM-JJ (défaut : début du CSV)")
    p.add_argument("--fin", help="AAAA-MM-JJ (défaut : fin du CSV)")
    p.add_argument("--sortie-seule", action="store_true",
                   help="affiche le rapport sans réécrire le CSV")
    args = p.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"],
                     index_col="timestamp").sort_index()
    debut = (datetime.fromisoformat(args.debut) if args.debut
             else df.index.min().to_pydatetime())
    fin = (datetime.fromisoformat(args.fin).replace(hour=23) if args.fin
           else df.index.max().to_pydatetime())

    print(f"{len(VILLES)} villes · {len({v.region for v in VILLES.values()})} régions "
          f"· poids total {POIDS_TOTAL:.1f} M habitants")
    print(f"Plage : {debut:%Y-%m-%d %H:%M} → {fin:%Y-%m-%d %H:%M}\n")
    print("Récupération Meteostat (par lot de stations et par année)...")

    temp_fr, par_ville, qualite = temperature_france(debut, fin, verbeux=True)

    print("\n--- Qualité par ville ---")
    print(qualite.to_string(
        index=False, float_format=lambda v: f"{v:,.2f}".replace(",", " ")))

    fraction = qualite.attrs["fraction_poids"]
    print(f"\nHeures produites          : {int(temp_fr.notna().sum()):,}"
          .replace(",", " "))
    print(f"Heures manquantes         : {int(temp_fr.isna().sum()):,}".replace(",", " "))
    print(f"Poids disponible médian   : {fraction.median():.1%}")
    print(f"Heures sous le seuil {SEUIL_POIDS:.0%}  : {qualite.attrs['heures_faibles']:,}"
          .replace(",", " "))

    if "temp" in df.columns:
        commun = df["temp"].reindex(temp_fr.index)
        both = commun.notna() & temp_fr.notna()
        ecart = (temp_fr[both] - commun[both])
        print("\n--- Comparaison avec la station unique existante (`temp`) ---")
        print(f"  moyenne temp (station)  : {commun[both].mean():6.2f} °C")
        print(f"  moyenne temp_fr (France): {temp_fr[both].mean():6.2f} °C")
        print(f"  écart moyen             : {ecart.mean():+6.2f} °C")
        print(f"  écart-type de l'écart   : {ecart.std():6.2f} °C")
        print(f"  corrélation             : {temp_fr[both].corr(commun[both]):6.3f}")
        print(f"  écart max (France - station) : {ecart.max():+.1f} °C "
              f"le {ecart.idxmax():%Y-%m-%d %H:%M}")
        print(f"  écart min                    : {ecart.min():+.1f} °C "
              f"le {ecart.idxmin():%Y-%m-%d %H:%M}")

    if args.sortie_seule:
        print("\n--sortie-seule : le CSV n'a pas été modifié.")
        return

    sauvegarde = f"{args.data}.bak_pre_temp_fr_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(args.data, sauvegarde)
    df["temp_fr"] = temp_fr.reindex(df.index)
    df.to_csv(args.data)
    print(f"\nSauvegarde : {sauvegarde}")
    print(f"Colonne `temp_fr` écrite dans {args.data} "
          f"({int(df['temp_fr'].notna().sum()):,} valeurs)".replace(",", " "))
    print("`temp` (station unique) est conservée : les deux colonnes coexistent "
          "le temps de trancher par backtest.")


if __name__ == "__main__":
    main()
