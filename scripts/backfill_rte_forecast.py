"""Rétro-alimente l'historique des prévisions RTE (J-1) — **repère externe**.

La prévision RTE sert d'étalon de comparaison et **rien d'autre** : elle n'entre
ni dans les features, ni dans l'entraînement, ni dans une combinaison avec nos
propres prévisions. Cette règle est un choix assumé — se mesurer à un concurrent
en consommant sa production relève du parasitisme — et elle reste vraie même
quand la combinaison serait rentable.

Mais un étalon n'a de valeur que s'il couvre assez de terrain. La base n'en
contenait que **7 origines** (23-29/07/2026), ce qui n'autorise aucune conclusion :
sur six jours, un écart de 4 % avec RTE est indiscernable du bruit. Ce script
étend le repère à tout l'historique.

Deux découvertes faites en l'écrivant
-------------------------------------
1. **L'offset UTC était figé à `+02:00`** dans `extract_conso_forecast_rte`, soit
   l'heure d'été. En hiver (+01:00), RTE ne renvoyait qu'une journée — et pas la
   bonne : le filtre sur la date locale vidait le résultat, `run()` persistait
   0 ligne et journalisait un SUCCÈS. Le DAG `fetch_rte_forecast` aurait donc
   échoué en silence de fin octobre à fin mars. Corrigé (`_offset_paris_url`).
2. **L'archive RTE remonte à au moins juillet 2022**, bien au-delà du début de nos
   observations — une fois l'offset corrigé.

Lacune connue et bornée : le dimanche du passage à l'heure d'été (jour de 23 h),
RTE n'a quasiment pas de prévision D-1. Vérifié sur 2026-03-29 : ~2 h au lieu de
23, quelle que soit la fenêtre demandée. Le script le signale au lieu de le taire.

Idempotent : les jours déjà complets sont ignorés.

Usage :
    python -m scripts.backfill_rte_forecast --debut 2023-01-01
    python -m scripts.backfill_rte_forecast --debut 2026-01-01 --fin 2026-07-30
    python -m scripts.backfill_rte_forecast --verifier          # sans écriture
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.db import get_observations, init_db, log_run, save_forecasts  # noqa: E402
from pipeline_prevision.db.config import get_engine  # noqa: E402
from pipeline_prevision.utils.main_utils.utils import (  # noqa: E402
    extract_conso_forecast_rte_range,
)

PARIS = ZoneInfo("Europe/Paris")
RTE_VARIABLE = "consommation_totale_rte"
RTE_MODEL_VERSION = "RTE (J-1)"
# En dessous de ce nombre d'heures, un jour est considéré incomplet et signalé.
# 23 h est légitime (passage à l'heure d'été), 25 h aussi (heure d'hiver).
HEURES_MIN_PAR_JOUR = 23


def jours_deja_couverts() -> dict:
    """Nombre de points RTE déjà en base, par jour cible LOCAL."""
    df = pd.read_sql(
        f"SELECT target_ts FROM forecasts WHERE variable = '{RTE_VARIABLE}'",
        get_engine(), parse_dates=["target_ts"])
    if df.empty:
        return {}
    local = df["target_ts"].dt.tz_localize("UTC").dt.tz_convert(PARIS)
    return local.dt.date.value_counts().to_dict()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debut", help="AAAA-MM-JJ (défaut : première observation en base)")
    p.add_argument("--fin", help="AAAA-MM-JJ (défaut : hier, heure de Paris)")
    p.add_argument("--verifier", action="store_true", help="diagnostic sans écriture")
    p.add_argument("--remplacer", action="store_true",
                   help="réécrit les jours déjà complets")
    args = p.parse_args()

    init_db()
    obs = get_observations()
    if obs is None or obs.empty:
        sys.exit("Aucune observation en base.")

    debut = pd.Timestamp(args.debut).date() if args.debut else obs.index.min().date()
    # La prévision J-1 du jour courant existe, mais son réalisé n'est pas complet :
    # on s'arrête la veille pour que chaque jour rétro-alimenté soit évaluable.
    defaut_fin = (datetime.now(PARIS).date() - timedelta(days=1))
    fin = pd.Timestamp(args.fin).date() if args.fin else defaut_fin

    couverts = jours_deja_couverts()
    tous = pd.date_range(debut, fin, freq="D").date
    manquants = [j for j in tous
                 if args.remplacer or couverts.get(j, 0) < HEURES_MIN_PAR_JOUR]

    print(f"Fenêtre demandée : {debut} -> {fin}  ({len(tous)} jours)")
    print(f"Déjà en base     : {len(couverts)} jours")
    print(f"À rétro-alimenter: {len(manquants)} jours")
    if not manquants:
        print("\nRien à faire.")
        return
    if args.verifier:
        print(f"\n--verifier : aucune écriture. Premier {manquants[0]}, "
              f"dernier {manquants[-1]}.")
        return

    t0 = time.time()
    total, incomplets = 0, []
    try:
        # On récupère la plage complète en quelques requêtes (fenêtres de 31 j),
        # puis on redécoupe par jour LOCAL : c'est le découpage qui définit une
        # « prévision J-1 », et il ne coïncide pas avec le découpage UTC.
        serie = extract_conso_forecast_rte_range(str(debut), str(fin), verbeux=True)
        local = serie.index.tz_localize("UTC").tz_convert(PARIS)
        serie = serie.assign(jour_local=local.date)

        a_traiter = set(manquants)
        for jour, groupe in serie.groupby("jour_local"):
            if jour not in a_traiter:
                continue
            points = groupe.drop(columns="jour_local")
            if len(points) < HEURES_MIN_PAR_JOUR:
                incomplets.append((jour, len(points)))
            # Origine conventionnelle = la veille du jour couvert, c'est-à-dire le
            # moment où RTE publie. Identique à `scripts/fetch_rte_forecast.py` :
            # la contrainte d'unicité porte sur (origin_ts, horizon_h, variable),
            # donc changer cette convention créerait des doublons.
            origine = datetime.combine(jour - timedelta(days=1), datetime.min.time())
            total += save_forecasts(points.rename(columns={"y_pred": RTE_VARIABLE}),
                                    origin_ts=origine,
                                    model_version=RTE_MODEL_VERSION)

        duree = time.time() - t0
        log_run("backfill_rte_forecast", "success", rows=total, duration_s=duree,
                message=f"repère RTE J-1 · {total} points · {debut}->{fin}")
    except Exception as e:
        log_run("backfill_rte_forecast", "failed", duration_s=time.time() - t0,
                message=str(e)[:500])
        raise

    print(f"\n{total} points persistés en {time.time() - t0:.0f} s")
    if incomplets:
        print(f"\n{len(incomplets)} jour(s) incomplet(s) côté RTE — signalés, non comblés :")
        for jour, n in incomplets[:10]:
            print(f"    {jour} : {n} h")
        if len(incomplets) > 10:
            print(f"    ... et {len(incomplets) - 10} autres")

    couverts = jours_deja_couverts()
    print(f"\nRepère RTE désormais : {len(couverts)} jours couverts, "
          f"du {min(couverts)} au {max(couverts)}")


if __name__ == "__main__":
    main()
