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
            
            
            y_pred_inverse = self.preprocessor.named_steps['scaler'].inverse_transform(y_pred)
            y_transform_inverse = self.preprocessor.named_steps['scaler'].inverse_transform(y_transform)

            return y_pred_inverse, y_transform_inverse

        except Exception as e:
            raise ForecastingException(e, sys)
        

    def predict_multistep(self, x, n_futur: int):
        
        try:
            # Transformation des données
            x_transform = self.preprocessor.transform(x)
            
            # Génération des séquences
            if x_transform.shape[0] == LOOKBACK:
                n_features = x_transform.shape[1]
                n_samples = 1
                x_seq = x_transform.reshape(n_samples, LOOKBACK*n_features)
                y_preds = np.zeros((n_samples, n_futur, n_features))
                
                # Prédiction directe
                for step in range(n_futur):
                    y_pred = self.model.predict(x_seq)
                    y_pred = y_pred.reshape(n_samples, n_features)
                    y_preds[:, step, :] = y_pred

                    # reconstruire en 3D et shift
                    x_seq_reshaped = x_seq.reshape(n_samples, LOOKBACK, n_features)
                    x_seq_reshaped = np.concatenate(
                        [x_seq_reshaped[:, 1:, :], np.repeat(y_pred[:, np.newaxis, :], 1, axis=1)],
                        axis=1
                    )
                    x_seq = x_seq_reshaped.reshape(n_samples, LOOKBACK * n_features)

                y_preds_inverse = self.preprocessor.named_steps['scaler'].inverse_transform(
                    y_preds.reshape(-1, n_features)
                )
                return y_preds_inverse, None

            else:
                # Cas général avec window_generator
                x_windows, y_windows = window_generator(
                    data=x_transform,
                    lookback=LOOKBACK,
                    horizon=HORIZON
                )

                if x_windows.shape[0] == 0:
                    raise ValueError(
                        f"Pas assez de séquences pour faire une prédiction. "
                        f"(lookback={LOOKBACK}, horizon={HORIZON})"
                    )

                n_samples, lookback_len, n_features = x_windows.shape
                n_features = y_windows.shape[2]

                x_seq = x_windows.reshape(n_samples, lookback_len * n_features)
                y_preds = np.zeros((n_samples, n_futur, n_features))

                for step in range(n_futur):
                    y_pred = self.model.predict(x_seq)
                    y_pred = y_pred.reshape(n_samples, n_features)
                    y_preds[:, step, :] = y_pred

                    x_seq_reshaped = x_seq.reshape(n_samples, lookback_len, n_features)
                    x_seq_reshaped = np.concatenate(
                        [x_seq_reshaped[:, 1:, :], np.repeat(y_pred[:, np.newaxis, :], 1, axis=1)],
                        axis=1
                    )
                    x_seq = x_seq_reshaped.reshape(n_samples, lookback_len * n_features)

                y_preds_inverse = self.preprocessor.named_steps['scaler'].inverse_transform(
                    y_preds[::n_futur, :, :].reshape(-1, n_features)
                )
                y_windows_inverse = self.preprocessor.named_steps['scaler'].inverse_transform(
                    y_windows.reshape(-1, n_features)
                )

                return y_preds_inverse, y_windows_inverse

        except Exception as e:
            raise ForecastingException(e, sys)
