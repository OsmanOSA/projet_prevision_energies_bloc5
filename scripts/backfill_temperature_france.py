"""Rétro-alimente `observations.temp_fr` sur tout l'historique déjà en base.

`init_db()` crée la colonne, mais une migration additive laisse le passé à NULL :
l'ingestion horaire ne remplit que les heures qu'elle traite. Tant que
l'historique n'est pas rétro-alimenté, `select_temperature` se replie sur `temp`
(cf. MIN_TEMPERATURE_COVERAGE) et l'entraînement — qui lit `datasets/data.csv`,
déjà complet — diverge de l'évaluation de promotion, qui lit la base.

Ce script referme cet écart. Il est idempotent : relançable sans effet de bord.

Usage :
    python -m scripts.backfill_temperature_france
    python -m scripts.backfill_temperature_france --depuis-csv    # sans réseau
    python -m scripts.backfill_temperature_france --verifier      # diagnostic seul
"""

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.db import get_observations, init_db  # noqa: E402
from pipeline_prevision.db.config import get_engine  # noqa: E402
from pipeline_prevision.utils.main_utils.feature_engineering import (  # noqa: E402
    MIN_TEMPERATURE_COVERAGE,
)

CSV = os.path.join("datasets", "data.csv")
LOT = 5000


def etat(obs: pd.DataFrame) -> tuple[int, int]:
    couverts = int(obs["temp_fr"].notna().sum()) if "temp_fr" in obs.columns else 0
    return couverts, len(obs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depuis-csv", action="store_true",
                   help="lit datasets/data.csv au lieu d'interroger Meteostat")
    p.add_argument("--data", default=CSV)
    p.add_argument("--verifier", action="store_true", help="diagnostic sans écriture")
    args = p.parse_args()

    init_db()
    obs = get_observations()
    if obs is None or obs.empty:
        sys.exit("Aucune observation en base.")

    couverts, total = etat(obs)
    print(f"Base : {total:,} lignes · temp_fr renseignée sur {couverts:,} "
          f"({100*couverts/total:.2f} %)".replace(",", " "))
    print(f"Seuil de bascule de select_temperature : {MIN_TEMPERATURE_COVERAGE:.0%} "
          f"-> actuellement {'OK' if couverts >= MIN_TEMPERATURE_COVERAGE*total else 'REPLI SUR temp'}")

    manquants = obs.index[obs["temp_fr"].isna()] if "temp_fr" in obs.columns else obs.index
    if len(manquants) == 0:
        print("Rien à rétro-alimenter.")
        return
    print(f"À combler : {len(manquants):,} heures "
          f"({manquants.min():%Y-%m-%d} → {manquants.max():%Y-%m-%d})".replace(",", " "))

    if args.verifier:
        print("\n--verifier : aucune écriture.")
        return

    if args.depuis_csv:
        source = pd.read_csv(args.data, parse_dates=["timestamp"],
                             index_col="timestamp").sort_index()
        if "temp_fr" not in source.columns:
            sys.exit(f"{args.data} n'a pas de colonne temp_fr : lancer d'abord "
                     "`python -m scripts.build_temperature_france`")
        serie = source["temp_fr"]
        print(f"Source : {args.data}")
    else:
        from pipeline_prevision.utils.main_utils.temperature_france import (
            temperature_france,
        )
        print("Source : Meteostat (17 stations)")
        serie, _, qualite = temperature_france(manquants.min().to_pydatetime(),
                                               manquants.max().to_pydatetime(),
                                               verbeux=True)
        print(f"Heures sous le seuil de poids : {qualite.attrs['heures_faibles']}")

    valeurs = serie.reindex(manquants).dropna()
    if valeurs.empty:
        sys.exit("Aucune valeur à écrire (la source ne couvre pas les heures manquantes).")

    # Écriture par lots : un UPDATE par heure sur 31 000 lignes sature le
    # journal de transaction et rend l'opération non interruptible.
    engine = get_engine()
    ecrites = 0
    for debut in range(0, len(valeurs), LOT):
        lot = valeurs.iloc[debut:debut + LOT]
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE observations SET temp_fr = :v WHERE ts = :t "
                     "AND temp_fr IS NULL"),
                [{"v": float(v), "t": pd.Timestamp(t).to_pydatetime()}
                 for t, v in lot.items()],
            )
        ecrites += len(lot)
        print(f"  {ecrites:,}/{len(valeurs):,}".replace(",", " "), end="\r")

    obs = get_observations()
    couverts, total = etat(obs)
    print(f"\nAprès rétro-alimentation : {couverts:,}/{total:,} "
          f"({100*couverts/total:.2f} %)".replace(",", " "))
    print("select_temperature utilisera désormais `temp_fr` : "
          f"{'OUI' if couverts >= MIN_TEMPERATURE_COVERAGE*total else 'NON, couverture encore insuffisante'}")


if __name__ == "__main__":
    main()
