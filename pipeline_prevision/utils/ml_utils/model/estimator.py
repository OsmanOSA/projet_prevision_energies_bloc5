import sys
import os
import glob
import numpy as np
import pandas as pd

from pipeline_prevision.constant.training_pipeline import MODEL_FILE_NAME, SAVED_MODEL_DIR, LOOKBACK, HORIZON
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.utils.main_utils.utils import window_generator

class ForecastModel:

    def __init__(self, 
                 preprocessor, 
                 model):
        
        try: 
            self.preprocessor = preprocessor
            self.model = model

        except Exception as e:
            raise ForecastingException(e, sys)
    
    def predict(self, x):
        
        try:
            x_transform = self.preprocessor.transform(x)

            x_transform, y_transform = window_generator(data = x_transform,
                                                        lookback = LOOKBACK,
                                                        horizon = HORIZON)
            
            print("x_transform shape:", x_transform.shape)
            print("y_transform shape:", y_transform.shape)

            if x_transform.shape[0] == 0:
                raise ValueError(f"Pas assez de séquences pour faire une prédiction. "
                                f"Vérifie la taille des données après transformation.")

            x_transform = x_transform.reshape(-1, x_transform.shape[1])
            y_transform = y_transform.reshape(-1)

            y_pred = self.model.predict(x_transform)

            return y_pred, y_transform

        except Exception as e:
            raise ForecastingException(e, sys)