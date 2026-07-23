"""FastAPI inference service for the local energy forecasting champion.

Pickle artifacts are loaded only from the trusted local MODEL_DIR. Do not mount
model artifacts received from untrusted users.
"""

from __future__ import annotations

import logging
import os
import sys
from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.utils.main_utils.utils import load_object
from pipeline_prevision.utils.ml_utils.metric.forecasting_metric import (
    get_forecast_score,
)
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
    get_model_version,
)

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

LOGGER = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "final_models"))
FEATURE_COUNT = 6
LOOKBACK = 36
MAX_FUTURE_HOURS = 168
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Energy Forecasting API", version="1.1.0")

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

model_cache: dict[str, ForecastModel] = {}


class PredictionMultiStep(BaseModel):
    data: list[list[float]]
    n_future: int = Field(ge=1, le=MAX_FUTURE_HOURS)


def get_forecast_model() -> ForecastModel:
    """Lazy-load the champion from the trusted local artifact directory."""
    try:
        if "forecast_model" not in model_cache:
            preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
            model_path = os.path.join(MODEL_DIR, "model.pkl")
            if not os.path.isfile(preprocessor_path) or not os.path.isfile(model_path):
                raise FileNotFoundError(
                    "Champion artifacts are missing from the configured MODEL_DIR"
                )
            preprocessor = load_object(preprocessor_path)
            final_model = load_object(model_path)
            model_cache["forecast_model"] = ForecastModel(
                preprocessor=preprocessor,
                model=final_model,
            )
            LOGGER.info("Forecast champion loaded: %s", get_model_version())
        return model_cache["forecast_model"]
    except ForecastingException:
        raise
    except Exception as exc:
        raise ForecastingException(exc, sys) from exc


def _validate_feature_rows(data: list[list[float]]) -> None:
    if len(data) < LOOKBACK:
        raise HTTPException(
            status_code=422,
            detail=f"At least {LOOKBACK} hourly rows are required",
        )
    if any(len(row) != FEATURE_COUNT for row in data):
        raise HTTPException(
            status_code=422,
            detail=f"Every row must contain exactly {FEATURE_COUNT} features",
        )
    if not np.isfinite(np.asarray(data, dtype=float)).all():
        raise HTTPException(status_code=422, detail="Input contains NaN or infinity")


@app.get("/", tags=["service"])
async def index():
    return {"message": "Energy Forecasting API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["service"])
async def health():
    required = ["model.pkl", "preprocessor.pkl"]
    missing = [name for name in required if not os.path.isfile(os.path.join(MODEL_DIR, name))]
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "missing_artifacts": missing},
        )
    return {"status": "ready", "model_version": get_model_version()}


@app.get("/api", include_in_schema=False)
async def api_docs():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


@app.post("/predict_batches", tags=["prediction"])
@app.post("/predict_batchs", tags=["prediction"], deprecated=True)
async def predict_batch(file: UploadFile = File(...)):
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded CSV exceeds 10 MiB")

        test = pd.read_csv(
            BytesIO(payload),
            sep=None,
            engine="python",
            parse_dates=[0],
            index_col=0,
        )
        forecast_model = get_forecast_model()
        y_pred, y_test = forecast_model.predict(x=test)
        forecast_metric = get_forecast_score(y_true=y_test, y_pred=y_pred)
        return {
            "MAE": float(forecast_metric.mae),
            "MSE": float(forecast_metric.mse),
            "rows_predicted": int(len(y_pred)),
            "model_version": get_model_version(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Batch prediction failed")
        raise HTTPException(status_code=422, detail="Invalid batch prediction input") from exc


@app.post("/predict_multistep", tags=["prediction"])
async def prediction(payload: PredictionMultiStep):
    _validate_feature_rows(payload.data)
    try:
        forecast_model = get_forecast_model()
        y_pred, y_test = forecast_model.predict_multistep(
            x=payload.data,
            n_futur=payload.n_future,
        )

        response = {
            "Pred": np.asarray(y_pred).tolist(),
            "Test": None,
            "MAE": None,
            "MSE": None,
            "model_version": get_model_version(),
        }
        if y_test is not None:
            forecast_metric = get_forecast_score(y_true=y_test, y_pred=y_pred)
            response.update(
                {
                    "Test": np.asarray(y_test).tolist(),
                    "MAE": float(forecast_metric.mae),
                    "MSE": float(forecast_metric.mse),
                }
            )
        return response
    except ForecastingException as exc:
        LOGGER.exception("Champion inference failed")
        raise HTTPException(status_code=503, detail="Model inference unavailable") from exc
    except Exception as exc:
        LOGGER.exception("Multistep prediction failed")
        raise HTTPException(status_code=422, detail="Invalid prediction input") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
