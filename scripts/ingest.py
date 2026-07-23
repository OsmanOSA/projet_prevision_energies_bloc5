"""Collecte et persistance des séries horaires réalisées.

Préfigure le futur DAG Airflow `ingest_hourly` : récupère température,
production et consommation via concat_all_data(), puis les upsert dans
PostgreSQL. Idempotent (rejouable sans doublon).

Usage :
    python -m scripts.ingest 2024-01-01 2024-01-07
    python -m scripts.ingest              # 7 derniers jours par défaut
"""

import sys
import time
from datetime import date, timedelta

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.utils.main_utils.utils import concat_all_data
from pipeline_prevision.db import init_db, upsert_observations, log_run


def run(start_date: str, 
        end_date: str) -> int:
    
    init_db()
    t0 = time.time()

    try:

        df = concat_all_data(start_date, end_date)
        n = upsert_observations(df)
        duration = time.time() - t0
        log_run("ingest_hourly", "success", rows=n, duration_s=duration,
                message=f"{start_date} -> {end_date}")
        logging.info("Ingestion OK : %s lignes (%s -> %s) en %.1fs",
                     n, start_date, end_date, duration)
        print(f"OK : {n} lignes persistées ({start_date} -> {end_date}) en {duration:.1f}s")
        return n

    except Exception as e:
        log_run("ingest_hourly", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Ingestion en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        today = date.today()
        start = (today - timedelta(days=7)).isoformat()
        end = today.isoformat()
    run(start, end)
