"""Configuration de la connexion PostgreSQL.

L'URL est construite depuis l'environnement. Les valeurs par défaut
correspondent au service `postgres` du docker-compose.yml, pour que les
scripts lancés sur l'hôte fonctionnent sans configuration supplémentaire.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

_engine: Engine | None = None


def get_database_url() -> str:
    """Retourne l'URL SQLAlchemy de la base.

    Priorité à DATABASE_URL si défini, sinon reconstruite à partir des
    variables POSTGRES_* (mêmes valeurs par défaut que docker-compose).
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user = os.getenv("POSTGRES_USER", "energia")
    password = os.getenv("POSTGRES_PASSWORD", "energia")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "energia")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine() -> Engine:
    """Retourne un moteur SQLAlchemy partagé (créé à la première demande)."""
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url(), pool_pre_ping=True, future=True)
    return _engine
