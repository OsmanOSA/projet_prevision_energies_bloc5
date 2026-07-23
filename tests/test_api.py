from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_service_contract_and_health() -> None:
    root = client.get("/")
    health = client.get("/health")
    schema = client.get("/openapi.json")

    assert root.status_code == 200
    assert health.status_code == 200
    assert health.json()["model_version"].startswith("sha256:")
    assert schema.status_code == 200
    assert "/predict_multistep" in schema.json()["paths"]


def test_multistep_validation_rejects_bad_shape() -> None:
    response = client.post(
        "/predict_multistep",
        json={"data": [[1.0] * 5 for _ in range(36)], "n_future": 24},
    )

    assert response.status_code == 422


def test_multistep_real_champion_prediction() -> None:
    data = pd.read_csv("datasets/data.csv").tail(36)
    payload = data[
        ["temp", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale"]
    ].values.tolist()

    response = client.post(
        "/predict_multistep",
        json={"data": payload, "n_future": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["Pred"]) == 3
    assert all(len(row) == 6 for row in body["Pred"])
    assert body["model_version"].startswith("sha256:")
