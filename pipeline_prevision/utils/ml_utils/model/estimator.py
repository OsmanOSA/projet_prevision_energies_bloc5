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

            x_transform = x_transform.reshape(-1, x_transform.shape[1]*x_transform.shape[2])
            y_transform = y_transform.reshape(-1, y_transform.shape[2])

            y_pred = self.model.predict(x_transform)

            print("y pred shape:", y_pred.shape)
            print("y tes shape:", y_transform.shape)

            return y_pred, y_transform

        except Exception as e:
            raise ForecastingException(e, sys)
        

    def predict_multistep(self, x, n_futur: int):
    
        try:
            # Transformation des données (scaling, encoding, etc.)
            x_transform = self.preprocessor.transform(x)

            # Génération des séquences
            x_windows, y_windows = window_generator(
                                                data=x_transform,
                                                lookback=LOOKBACK,
                                                horizon=HORIZON)

            if x_windows.shape[0] == 0:
                raise ValueError(
                    f"Pas assez de séquences pour faire une prédiction. "
                    f"(lookback={LOOKBACK}, horizon={HORIZON})"
                )

            # Mise en forme initiale
            n_samples, lookback, n_features = x_windows.shape
            n_targets = y_windows.shape[2]

            # Aplatir pour correspondre au modèle (MLP/sklearn)
            x_seq = x_windows.reshape(n_samples, lookback * n_features)

            # Stockage des prédictions
            y_preds = np.zeros((n_samples, n_futur, n_targets))

            for step in range(n_futur):
                # 1) prédiction du prochain pas
                y_pred = self.model.predict(x_seq)
                y_pred = y_pred.reshape(n_samples, n_targets)

                y_preds[:, step, :] = y_pred

                # 2) mise à jour des séquences : on décale la fenêtre et on insère y_pred
                # reconstruire en 3D (batch, lookback, features)
                x_seq_reshaped = x_seq.reshape(n_samples, lookback, n_features)

                # shift left et insérer la prédiction dans les colonnes cibles (ici dernières colonnes)
                x_seq_reshaped = np.concatenate(
                    [x_seq_reshaped[:, 1:, :],
                    np.repeat(y_pred[:, np.newaxis, :], 1, axis=1)], axis=1
                )

                x_seq = x_seq_reshaped.reshape(n_samples, lookback * n_features)

            print("y_preds shape:", y_preds.shape)
            print("y_true shape:", y_windows.shape)

            return y_preds, y_windows

        except Exception as e:
            raise ForecastingException(e, sys)
