"""Peuplement de la table `observations` depuis le jeu de données versionné.

Rend reproductible le démarrage à froid : `datasets/data.csv` est le seul
historique versionné du dépôt (et celui sur lequel le modèle en production a
été entraîné, cf. `dataset_sha256` dans `final_models/metadata.json`). Sans ce
script, un déploiement neuf repart d'une table vide et n'a d'autre recours que
de rejouer plusieurs années d'API RTE.

Sert aussi de réparation. `observations` est la vérité terrain de
`evaluate_daily` et du backtesting : si elle contient des valeurs aberrantes
ou d'origine inconnue, toutes les métriques sont fausses sans que rien ne le
signale. `--remplacer` réaligne la plage du CSV sur son contenu exact, après
archivage systématique de ce qui est écrasé.

La provenance est inscrite dans la colonne `source` sous la forme
`data.csv:<sha8>` : l'origine de chaque ligne reste vérifiable, et un simple
`SELECT DISTINCT source` révèle un mélange de jeux de données.

Usage :
    python -m scripts.seed_observations                # charge le CSV, puis l'API RTE jusqu'à aujourd'hui
    python -m scripts.seed_observations --remplacer    # archive et réécrit la plage du CSV
    python -m scripts.seed_observations --sans-rte     # s'en tient au CSV
"""

import argparse
import gzip
import hashlib
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.db import init_db, upsert_observations, log_run
from pipeline_prevision.db.config import get_engine
from pipeline_prevision.utils.main_utils.utils import PRODUCTION_BOUNDS

DATASET = os.path.join(_ROOT, "datasets", "data.csv")
ARCHIVES = os.path.join(_ROOT, "datasets")
# PostgreSQL plafonne à 65535 paramètres par requête, et chaque ligne en
# consomme une dizaine (colonnes + source) : 4000 laisse une marge confortable.
CHUNK = 4000


def _sha8(chemin: str) -> str:
    """8 premiers caractères du SHA-256, assez pour identifier un jeu de
    données sans alourdir la colonne `source`."""
    digest = hashlib.sha256()
    with open(chemin, "rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            digest.update(bloc)
    return digest.hexdigest()[:8]


def _lire_dataset(chemin: str) -> pd.DataFrame:
    df = pd.read_csv(chemin, parse_dates=["timestamp"]).set_index("timestamp")
    return df.sort_index()


def _archiver(debut, fin) -> str | None:
    """Sauvegarde compressée des lignes sur le point d'être écrasées.

    Systématique avant tout remplacement : les lignes détruites peuvent être
    la seule copie d'un historique dont on ne retrouvera pas la source.
    """
    with get_engine().connect() as conn:
        resultat = conn.execute(
            text("SELECT * FROM observations WHERE ts BETWEEN :d AND :f ORDER BY ts"),
            {"d": debut.to_pydatetime(), "f": fin.to_pydatetime()},
        )
        existant = pd.DataFrame(resultat.fetchall(), columns=list(resultat.keys()))

    if existant.empty:
        return None

    horodatage = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    destination = os.path.join(ARCHIVES, f"archive_observations_{horodatage}.csv.gz")
    with gzip.open(destination, "wt", newline="", encoding="utf-8") as sortie:
        existant.to_csv(sortie, index=False)
    print(f"  archive : {len(existant)} lignes -> {os.path.relpath(destination, _ROOT)}")
    return destination


def _remplacer(debut, fin) -> int:
    """Supprime la plage avant réinsertion.

    Un simple upsert ne suffirait pas : `upsert_observations` protège les
    valeurs existantes par COALESCE (pour qu'un rejeu d'ingestion ne détruise
    rien), donc une valeur aberrante déjà en base survivrait à un NULL du CSV.
    Ici on veut au contraire que le CSV fasse autorité.
    """
    with get_engine().begin() as conn:
        return conn.execute(
            text("DELETE FROM observations WHERE ts BETWEEN :d AND :f"),
            {"d": debut.to_pydatetime(), "f": fin.to_pydatetime()},
        ).rowcount


def _controler() -> dict:
    """Continuité horaire, bornes physiques et complétude, après coup."""
    from pipeline_prevision.db import get_observations

    obs = get_observations()
    if obs is None or obs.empty:
        return {"lignes": 0}

    obs = obs.sort_index()
    ecarts = obs.index.to_series().diff()
    trous = ecarts[ecarts > pd.Timedelta("1h")]
    hors_bornes = {
        colonne: int(((obs[colonne] < bas) | (obs[colonne] > haut)).sum())
        for colonne, (bas, haut) in PRODUCTION_BOUNDS.items()
        if colonne in obs.columns
    }
    return {
        "lignes": len(obs),
        "debut": obs.index.min(),
        "fin": obs.index.max(),
        "trous": [(str(ts), str(ecart)) for ts, ecart in trous.items()],
        "hors_bornes": {k: v for k, v in hors_bornes.items() if v},
        "incompletes": int(obs.isna().any(axis=1).sum()),
    }


def run(remplacer: bool = False, avec_rte: bool = True, dataset: str = DATASET) -> int:
    t0 = time.time()
    try:
        init_db()

        if not os.path.isfile(dataset):
            raise FileNotFoundError(f"Jeu de données introuvable : {dataset}")

        df = _lire_dataset(dataset)
        source = f"{os.path.basename(dataset)}:{_sha8(dataset)}"
        debut, fin = df.index.min(), df.index.max()
        print(f"Jeu de données : {len(df)} lignes, {debut} -> {fin}")
        print(f"Provenance inscrite en base : {source}")

        if remplacer:
            _archiver(debut, fin)
            print(f"  supprimées : {_remplacer(debut, fin)} lignes")

        persistees = 0
        for i in range(0, len(df), CHUNK):
            persistees += upsert_observations(df.iloc[i:i + CHUNK], source=source)
        print(f"  persistées : {persistees} lignes")

        if avec_rte:
            # Reliquat entre la fin du CSV et aujourd'hui, par le chemin
            # d'ingestion normal (bornage + interpolation appliqués).
            from scripts.ingest import run as ingerer

            depart = (fin.date() - timedelta(days=1)).isoformat()
            arrivee = (date.today() + timedelta(days=1)).isoformat()
            print(f"Ingestion RTE du reliquat : {depart} -> {arrivee}")
            ingerer(depart, arrivee)

        controle = _controler()
        print("\nContrôle final :")
        print(f"  {controle['lignes']} lignes, {controle.get('debut')} -> {controle.get('fin')}")
        print(f"  trous > 1h    : {len(controle.get('trous', []))}")
        for ts, ecart in controle.get("trous", []):
            print(f"      {ecart} avant {ts}")
        print(f"  hors bornes   : {controle.get('hors_bornes') or 'aucune'}")
        print(f"  incomplètes   : {controle.get('incompletes')}")

        duree = time.time() - t0
        log_run("seed_observations", "success", rows=persistees, duration_s=duree,
                message=f"{source} · {len(df)} lignes · remplacer={remplacer} · rte={avec_rte}")
        logging.info("Seed observations OK : %s lignes (%s) en %.1fs", persistees, source, duree)
        return persistees

    except Exception as e:
        log_run("seed_observations", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Seed des observations en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--remplacer", action="store_true",
                         help="archive puis réécrit la plage du CSV (le CSV fait autorité)")
    parseur.add_argument("--sans-rte", action="store_true",
                         help="ne pas ingérer le reliquat depuis l'API RTE")
    parseur.add_argument("--dataset", default=DATASET, help=f"défaut : {DATASET}")
    arguments = parseur.parse_args()
    run(remplacer=arguments.remplacer, avec_rte=not arguments.sans_rte,
        dataset=arguments.dataset)
