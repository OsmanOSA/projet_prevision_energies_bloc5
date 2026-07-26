"""Modèle de données de la plateforme EnergIA (SQLAlchemy Core).

Source de vérité unique du schéma : `init_db()` crée ces tables
(idempotent). Quatre tables couvrent le cycle de vie :

- observations     : séries horaires réalisées (température, production, conso)
- forecasts        : prévisions produites (par origine, horizon, variable)
- forecast_metrics : erreurs prévu vs réalisé (backtesting continu)
- pipeline_runs    : traces d'exécution des DAGs (KPI opérationnels)
"""

from sqlalchemy import (
    MetaData, Table, Column, BigInteger, Integer, Text, Float,
    DateTime, UniqueConstraint, Index, func,
)

metadata = MetaData()

# Séries horaires réalisées. Table "large" : une ligne par horodatage,
# alignée sur le DataFrame produit par concat_all_data().
observations = Table(
    "observations", metadata,
    Column("ts", DateTime, primary_key=True),
    Column("temp", Float),
    Column("production_total", Float),  # dérivée : somme des 4 colonnes ci-dessous
    Column("solar", Float),
    Column("biomass", Float),
    Column("wind_onshore", Float),
    Column("nuclear", Float),
    Column("consommation_totale", Float),
    Column("source", Text, nullable=False, server_default="rte+meteostat"),
    Column("ingested_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)
Index("idx_observations_ts", observations.c.ts)

# Prévisions produites. Table "longue" : une ligne par (origine, horizon, variable),
# ce qui porte naturellement les intervalles de confiance et permet le join
# avec les observations pour l'évaluation.
forecasts = Table(
    "forecasts", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_ts", DateTime(timezone=True), nullable=False),   # instant de génération
    Column("origin_ts", DateTime, nullable=False),               # dernier point observé
    Column("target_ts", DateTime, nullable=False),               # horodatage prédit
    Column("horizon_h", Integer, nullable=False),                # target_ts - origin_ts (heures)
    Column("variable", Text, nullable=False),
    Column("y_pred", Float, nullable=False),
    Column("y_lower", Float),
    Column("y_upper", Float),
    Column("model_version", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("origin_ts", "horizon_h", "variable", name="uq_forecast_key"),
)
Index("idx_forecasts_target", forecasts.c.target_ts, forecasts.c.variable)

# Erreurs de prévision (prévu vs réalisé), agrégées par variable et horizon.
forecast_metrics = Table(
    "forecast_metrics", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("eval_ts", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("period_start", DateTime),
    Column("period_end", DateTime),
    Column("variable", Text, nullable=False),
    Column("horizon_h", Integer),
    Column("mae", Float),
    Column("mse", Float),
    Column("rmse", Float),
    Column("mape", Float),
    Column("bias", Float),
    Column("coverage", Float),
    Column("n_points", Integer),
    Column("model_version", Text),
)
Index("idx_metrics_var", forecast_metrics.c.variable, forecast_metrics.c.horizon_h)

# Traces d'exécution des pipelines (alimente les KPI opérationnels).
pipeline_runs = Table(
    "pipeline_runs", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("dag_id", Text, nullable=False),
    Column("run_ts", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("status", Text, nullable=False),         # success / failed / running
    Column("rows", Integer),
    Column("duration_s", Float),
    Column("message", Text),
)
Index("idx_runs_dag", pipeline_runs.c.dag_id, pipeline_runs.c.run_ts)
