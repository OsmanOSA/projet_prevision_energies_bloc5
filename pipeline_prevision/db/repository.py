"""Accès aux données : lecture/écriture des séries, prévisions et métriques.

Toutes les écritures de séries et de prévisions sont idempotentes (upsert),
pour que la ré-exécution d'un DAG ne crée pas de doublons.
"""

import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import and_, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from pipeline_prevision.db.config import get_engine
from pipeline_prevision.db.models import (
    metadata, observations, forecasts, forecast_metrics, pipeline_runs,
)
from pipeline_prevision.exception.exception import ForecastingException

# Correspondance colonnes DataFrame (concat_all_data) <-> colonnes SQL.
# Le DataFrame utilise les noms d'origine (majuscules RTE), la base les
# normalise en snake_case.
DF_TO_DB = {
    "temp": "temp",
    "production_total": "production_total",
    "SOLAR": "solar",
    "BIOMASS": "biomass",
    "WIND_ONSHORE": "wind_onshore",
    "NUCLEAR": "nuclear",
    "consommation_totale": "consommation_totale",
}
DB_TO_DF = {v: k for k, v in DF_TO_DB.items()}
OBSERVATION_VARIABLES = list(DF_TO_DB.keys())


def init_db() -> None:
    """Crée les tables si besoin et applique les migrations légères (idempotent)."""
    try:
        engine = get_engine()
        metadata.create_all(engine)
        # create_all n'ajoute pas de colonne à une table déjà existante :
        # migrations additives explicites.
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE forecast_metrics ADD COLUMN IF NOT EXISTS mse DOUBLE PRECISION"
            ))
            # Historique : passage du détail par filière à l'agrégat unique
            # production_total (cf. concat_all_data), puis retour au détail
            # par filière (décision produit : l'agrégat noyait l'éolien, seule
            # composante volatile, sous des composantes stables/calendaires).
            # production_total reste une colonne dérivée (somme des 4 filières).
            conn.execute(text(
                "ALTER TABLE observations ADD COLUMN IF NOT EXISTS production_total DOUBLE PRECISION"
            ))
            per_source_cols = {"solar", "biomass", "wind_onshore", "nuclear"}
            for col in per_source_cols:
                conn.execute(text(
                    f"ALTER TABLE observations ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION"
                ))
            # Backfill de production_total uniquement s'il manque encore et
            # que le détail par filière est disponible pour le recalculer.
            existing = {c["name"] for c in inspect(conn).get_columns("observations")}
            if per_source_cols & existing:
                sum_expr = " + ".join(f"COALESCE({c}, 0)" for c in per_source_cols if c in existing)
                not_null_expr = " OR ".join(f"{c} IS NOT NULL" for c in per_source_cols if c in existing)
                conn.execute(text(
                    f"UPDATE observations SET production_total = {sum_expr} "
                    f"WHERE production_total IS NULL AND ({not_null_expr})"
                ))
    except Exception as e:
        raise ForecastingException(e, sys)


def upsert_observations(df: pd.DataFrame, source: str = "rte+meteostat") -> int:
    """Insère/actualise les séries horaires réalisées.

    df est indexé par l'horodatage (sortie de concat_all_data) ; seules les
    colonnes connues (DF_TO_DB) sont persistées. Conflit sur `ts` -> mise à jour.
    Retourne le nombre de lignes traitées.
    """
    try:
        if df is None or df.empty:
            return 0

        present = [c for c in DF_TO_DB if c in df.columns]
        records = []
        for ts, row in df.iterrows():
            rec = {"ts": pd.Timestamp(ts).to_pydatetime(), "source": source}
            for df_col in present:
                val = row[df_col]
                rec[DF_TO_DB[df_col]] = None if pd.isna(val) else float(val)
            records.append(rec)

        stmt = pg_insert(observations).values(records)
        # COALESCE : une valeur déjà connue n'est jamais écrasée par un NULL.
        # Chaque ingestion rejoue une fenêtre glissante et ses lignes de bord
        # arrivent forcément incomplètes (fenêtres RTE/Meteostat décalées,
        # publication tardive d'une source) ; sans ce garde-fou, un simple
        # rejeu détruit des observations valides — et un trou de `temp` suffit
        # à invalider 168 h de features en aval. Une correction réelle (valeur
        # non nulle) écrase toujours l'ancienne, comme attendu.
        update_cols = {
            DF_TO_DB[c]: func.coalesce(stmt.excluded[DF_TO_DB[c]], observations.c[DF_TO_DB[c]])
            for c in present
        }
        update_cols["source"] = stmt.excluded.source
        update_cols["ingested_at"] = datetime.now(timezone.utc)
        stmt = stmt.on_conflict_do_update(index_elements=["ts"], set_=update_cols)

        with get_engine().begin() as conn:
            conn.execute(stmt)
        return len(records)
    except Exception as e:
        raise ForecastingException(e, sys)


def get_observations(start=None, end=None) -> pd.DataFrame:
    """Lit les séries réalisées, renvoyées comme le DataFrame attendu par le
    dashboard : indexé par `timestamp`, colonnes aux noms d'origine.
    """
    try:
        cols = [observations.c.ts] + [observations.c[db] for db in DB_TO_DF]
        query = select(*cols)
        if start is not None:
            query = query.where(observations.c.ts >= pd.Timestamp(start).to_pydatetime())
        if end is not None:
            query = query.where(observations.c.ts <= pd.Timestamp(end).to_pydatetime())
        query = query.order_by(observations.c.ts)

        # Exécution directe via SQLAlchemy (robuste 1.4 et 2.0) plutôt que
        # pd.read_sql, dont la détection du Selectable diffère selon la version.
        with get_engine().connect() as conn:
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))

        if df.empty:
            return df
        df = df.rename(columns={"ts": "timestamp", **DB_TO_DF})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df
    except Exception as e:
        raise ForecastingException(e, sys)


def save_forecasts(pred_df: pd.DataFrame, origin_ts, run_ts=None,
                   model_version: str | None = None,
                   lower_df: pd.DataFrame | None = None,
                   upper_df: pd.DataFrame | None = None) -> int:
    """Persiste une prévision multi-horizon.

    pred_df : indexé par l'horodatage cible, une colonne par variable.
    origin_ts : dernier point observé (origine de la prévision).
    lower_df / upper_df : bornes d'intervalle optionnelles (même forme).
    Conflit sur (origin_ts, horizon_h, variable) -> mise à jour.
    """
    try:
        if pred_df is None or pred_df.empty:
            return 0

        origin = pd.Timestamp(origin_ts).to_pydatetime()
        run = pd.Timestamp(run_ts).to_pydatetime() if run_ts else datetime.now(timezone.utc)

        records = []
        for target_ts, row in pred_df.iterrows():
            target = pd.Timestamp(target_ts).to_pydatetime()
            horizon = int(round((target - origin).total_seconds() / 3600))
            for var in pred_df.columns:
                val = row[var]
                if pd.isna(val):
                    continue
                rec = {
                    "run_ts": run,
                    "origin_ts": origin,
                    "target_ts": target,
                    "horizon_h": horizon,
                    "variable": var,
                    "y_pred": float(val),
                    "model_version": model_version,
                }
                if lower_df is not None and var in lower_df.columns:
                    rec["y_lower"] = float(lower_df.loc[target_ts, var])
                if upper_df is not None and var in upper_df.columns:
                    rec["y_upper"] = float(upper_df.loc[target_ts, var])
                records.append(rec)

        if not records:
            return 0

        stmt = pg_insert(forecasts).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_forecast_key",
            set_={
                "run_ts": stmt.excluded.run_ts,
                "target_ts": stmt.excluded.target_ts,
                "y_pred": stmt.excluded.y_pred,
                "y_lower": stmt.excluded.y_lower,
                "y_upper": stmt.excluded.y_upper,
                "model_version": stmt.excluded.model_version,
            },
        )
        with get_engine().begin() as conn:
            conn.execute(stmt)
        return len(records)
    except Exception as e:
        raise ForecastingException(e, sys)


def save_metrics(rows: list[dict]) -> int:
    """Remplace idempotemment les métriques pour une même clé d'évaluation."""
    try:
        if not rows:
            return 0
        with get_engine().begin() as conn:
            for row in rows:
                version_condition = (
                    forecast_metrics.c.model_version.is_(None)
                    if row.get("model_version") is None
                    else forecast_metrics.c.model_version == row["model_version"]
                )
                conn.execute(
                    forecast_metrics.delete().where(
                        and_(
                            forecast_metrics.c.period_start == row.get("period_start"),
                            forecast_metrics.c.period_end == row.get("period_end"),
                            forecast_metrics.c.variable == row["variable"],
                            forecast_metrics.c.horizon_h == row.get("horizon_h"),
                            version_condition,
                        )
                    )
                )
            conn.execute(pg_insert(forecast_metrics), rows)
        return len(rows)
    except Exception as e:
        raise ForecastingException(e, sys)


def log_run(dag_id: str,
            status: str, rows: int | None = None,
            duration_s: float | None = None, 
            message: str | None = None) -> None:
    """Enregistre une trace d'exécution de pipeline (KPI opérationnels)."""
    try:
        with get_engine().begin() as conn:
            conn.execute(pipeline_runs.insert().values(
                dag_id=dag_id, status=status, rows=rows,
                duration_s=duration_s, message=message,
            ))
    except Exception as e:
        raise ForecastingException(e, sys)
