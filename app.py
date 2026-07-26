"""FastAPI inference service for the local energy forecasting champion.

Pickle artifacts are loaded only from the trusted local MODEL_DIR. Do not mount
model artifacts received from untrusted users.

Not used by the current Streamlit dashboard (which calls
`local_forecaster.py` directly) — kept for external API consumers, updated
here for consistency with the direct multi-horizon residual architecture
(cf. pipeline_prevision/utils/main_utils/feature_engineering.py).
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
    TARGETS,
    HORIZON_MAX,
    get_model_version,
    predict_with_conformal_intervals,
)

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

LOGGER = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "final_models"))
MIN_HISTORY_HOURS = 340  # marge au-delà du plus grand lag utilisé (336h)

app = FastAPI(title="Energy Forecasting API", version="2.0.0")

cors_origins = [
    value.strip()
    for value in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8501,http://localhost:3000",
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class Observation(BaseModel):
    timestamp: str
    SOLAR: float
    BIOMASS: float
    WIND_ONSHORE: float
    NUCLEAR: float
    consommation_totale: float
    temp: float


class PredictionRequest(BaseModel):
    observations: list[Observation] = Field(min_length=MIN_HISTORY_HOURS)
    target: str
    horizons: list[int] | None = None


@app.get("/", tags=["service"])
async def index():
    return {"message": "Energy Forecasting API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["service"])
async def health():
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "missing_artifacts": ["model.pkl"]},
        )
    return {"status": "ready", "model_version": get_model_version()}


@app.get("/api", include_in_schema=False)
async def api_docs():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


@app.post("/predict", tags=["prediction"])
async def prediction(payload: PredictionRequest):
    if payload.target not in TARGETS:
        raise HTTPException(status_code=422, detail=f"target doit être l'un de {TARGETS}")
    horizons = payload.horizons or list(range(1, HORIZON_MAX + 1))
    if any(h < 1 or h > HORIZON_MAX for h in horizons):
        raise HTTPException(status_code=422, detail=f"horizons doit être entre 1 et {HORIZON_MAX}")

    try:
        observations = pd.DataFrame([o.model_dump() for o in payload.observations])
        observations["timestamp"] = pd.to_datetime(observations["timestamp"])
        observations = observations.set_index("timestamp").sort_index()

        result = predict_with_conformal_intervals(observations, payload.target, horizons=horizons)
        return {
            "target": payload.target,
            "model_version": get_model_version(),
            "forecast": [
                {
                    "target_ts": ts.isoformat(),
                    "horizon_h": int(row["horizon_h"]),
                    "y_pred": float(row["y_pred"]),
                    "y_lower": float(row["y_lower"]) if pd.notna(row["y_lower"]) else None,
                    "y_upper": float(row["y_upper"]) if pd.notna(row["y_upper"]) else None,
                }
                for ts, row in result.iterrows()
            ],
        }
    except ForecastingException as exc:
        LOGGER.exception("Champion inference failed")
        raise HTTPException(status_code=503, detail="Model inference unavailable") from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=422, detail="Invalid prediction input") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
