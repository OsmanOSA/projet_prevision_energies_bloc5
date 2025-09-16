import sys
import os
import pandas as pd
import numpy as np

from io import BytesIO
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.pipeline.training_pipeline import TrainingPipeline
from pipeline_prevision.utils.main_utils.utils import load_object, load_numpy_array_data
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel
from pipeline_prevision.utils.ml_utils.metric.forecasting_metric import get_forecast_score

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run 
from fastapi.responses import Response 
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates




app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():

    try:

        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()

        return Response("Training is successful")
    
    except Exception as e:
        raise ForecastingException(e, sys)
    

@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):

    try:

        test = pd.read_csv(file.file, sep=None, engine="python", parse_dates=["Date"], index_col="Date")
        

        print(test.head())

        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")

        forecast_model = ForecastModel(preprocessor = preprocessor,
                                       model = final_model)

        y_pred, y_test = forecast_model.predict(x=test)

        forecast_metric = get_forecast_score(y_true=y_test, 
                                             y_pred=y_pred)

        print(forecast_metric.mae)
        print(forecast_metric.mse)
        
        df = pd.DataFrame({
            "MAE": [forecast_metric.mae], 
            "MSE": [forecast_metric.mse]
        })
        
        #df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        raise ForecastingException(e, sys)
    
if __name__=="__main__":
    app_run(app,host="localhost",port=8000)