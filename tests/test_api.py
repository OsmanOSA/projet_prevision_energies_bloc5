from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app import app, MIN_HISTORY_HOURS


client = TestClient(app)

OBSERVATION_COLUMNS = ["SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale", "temp"]


def _real_observations(rows: int) -> list[dict]:
    """Dernières `rows` heures du dataset réel, au format attendu par /predict."""
    data = pd.read_csv("datasets/data.csv", parse_dates=["timestamp"]).tail(rows)
    return [
        {
            "timestamp": row.timestamp.isoformat(),
            **{column: float(getattr(row, column)) for column in OBSERVATION_COLUMNS},
        }
        for row in data.itertuples()
    ]


def test_service_contract_and_health() -> None:
    root = client.get("/")
    health = client.get("/health")
    schema = client.get("/openapi.json")

    assert root.status_code == 200
    assert health.status_code == 200
    assert health.json()["model_version"].startswith("sha256:")
    assert schema.status_code == 200
    assert "/predict" in schema.json()["paths"]


def test_predict_rejects_unknown_target() -> None:
    response = client.post(
        "/predict",
        json={"observations": _real_observations(MIN_HISTORY_HOURS), "target": "GAZ"},
    )

    assert response.status_code == 422


def test_predict_rejects_insufficient_history() -> None:
    response = client.post(
        "/predict",
        json={
            "observations": _real_observations(MIN_HISTORY_HOURS - 1),
            "target": "consommation_totale",
        },
    )

    assert response.status_code == 422


def test_predict_real_champion_prediction() -> None:
    response = client.post(
        "/predict",
        json={
            "observations": _real_observations(MIN_HISTORY_HOURS + 5),
            "target": "consommation_totale",
            "horizons": [1, 2, 3],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target"] == "consommation_totale"
    assert body["model_version"].startswith("sha256:")
    assert [row["horizon_h"] for row in body["forecast"]] == [1, 2, 3]
    assert all(isinstance(row["y_pred"], float) for row in body["forecast"])
    # Intervalle conforme cohérent : borne basse <= prévision <= borne haute.
    for row in body["forecast"]:
        if row["y_lower"] is not None and row["y_upper"] is not None:
            assert row["y_lower"] <= row["y_pred"] <= row["y_upper"]
