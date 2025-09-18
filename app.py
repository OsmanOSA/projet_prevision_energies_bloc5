import sys
import os
import pandas as pd
import numpy as np

from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline_prevision.exception.exception import ForecastingException
#from pipeline_prevision.pipeline.training_pipeline import TrainingPipeline
from pipeline_prevision.utils.main_utils.utils import load_object
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel
from pipeline_prevision.utils.ml_utils.metric.forecasting_metric import get_forecast_score



# FastAPI app
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache pour lazy loading des modèles
model_cache = {}

class PredictionMultiStep(BaseModel):
    data: list
    n_future: int

# Lazy loading du modèle
def get_forecast_model():
    if "forecast_model" not in model_cache:
        print("Loading forecast model...")
        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")
        model_cache["forecast_model"] = ForecastModel(preprocessor=preprocessor, model=final_model)
        print("Forecast model loaded.")
    return model_cache["forecast_model"]

# Routes
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise ForecastingException(e, sys)

@app.post("/predict_batchs")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        test = pd.read_csv(file.file, sep=None, engine="python", parse_dates=[0], index_col=0)
        forecast_model = get_forecast_model()
        y_pred, y_test = forecast_model.predict(x=test)
        forecast_metric = get_forecast_score(y_true=y_test, y_pred=y_pred)
        return {
            "MAE": float(forecast_metric.mae),
            "MSE": float(forecast_metric.mse)
        }
    except Exception as e:
        raise ForecastingException(e, sys)

@app.post("/predict_multistep")
async def prediction(payload: PredictionMultiStep):
    try:
        forecast_model = get_forecast_model()
        y_pred, y_test = forecast_model.predict_multistep(x=payload.data, n_futur=payload.n_future)
        forecast_metric = get_forecast_score(y_true=y_test, y_pred=y_pred)
        return {
            "Pred": np.asarray(y_pred).tolist(),
            "Test": np.asarray(y_test).tolist(),
            "MAE": float(forecast_metric.mae),
            "MSE": float(forecast_metric.mse)
        }
    except Exception as e:
        raise ForecastingException(e, sys)

# Démarrage avec port dynamique
if __name__ == "__main__":
   uvicorn.run(app)
