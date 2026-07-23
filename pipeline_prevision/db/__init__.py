"""Couche de persistance PostgreSQL de la plateforme EnergIA.

Expose l'accès aux données (séries observées, prévisions, métriques,
exécutions de pipelines) pour découpler la collecte de la consultation.
"""

from pipeline_prevision.db.config import (get_engine,
                                          get_database_url)
from pipeline_prevision.db.repository import (
    init_db,
    upsert_observations,
    get_observations,
    save_forecasts,
    save_metrics,
    log_run,
    OBSERVATION_VARIABLES,
)

__all__ = [
    "get_engine",
    "get_database_url",
    "init_db",
    "upsert_observations",
    "get_observations",
    "save_forecasts",
    "save_metrics",
    "log_run",
    "OBSERVATION_VARIABLES",
]
