import sys

from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.entity.artifact_entity import ForecastMetricArtifact
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_forecast_score(y_true, y_pred) -> ForecastMetricArtifact: 

    try: 
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        return ForecastMetricArtifact(mae, mse)

    except Exception as e:
        raise ForecastingException(e, sys)