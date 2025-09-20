# frontend/api_client.py
import requests
import numpy as np
import pandas as pd
import os
import sys

from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_prevision.exception.exception import ForecastingException

class ForecastAPIClient:
    def __init__(self):
        # URL de l'API FastAPI déployée
        self.api_url = "https://energiesforecasts-7f55a2300a2c.herokuapp.com"
        
    def predict_multistep(self, data: pd.DataFrame, n_future: int):
       
        try:
            # Préparer les données pour l'API
            features = ['BIOMASS', 'NUCLEAR', 'SOLAR', 'WIND_ONSHORE', 'consommation_totale', 'temp']
            
            # Vérifier que toutes les features sont présentes
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ForecastingException(f"Features manquantes: {missing_features}", sys)
            
            # Prendre les 36 dernières valeurs et réorganiser les colonnes
            df_features = data[features].tail(36).copy()
            
            # Convertir en liste pour l'API
            data_list = df_features.values.tolist()
            
            # Appel à l'API
            payload = {
                "data": data_list,
                "n_future": n_future
            }
            
            print(f"Appel API vers: {self.api_url}/predict_multistep")
            print(f"Payload shape: {len(data_list)} x {len(data_list[0]) if data_list else 0}")
            
            response = requests.post(
                f"{self.api_url}/predict_multistep",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Convertir les résultats
                y_pred = np.array(result.get("Pred", []))
                y_test = np.array(result.get("Test", [])) if result.get("Test") else None
                mae = result.get("MAE")
                mse = result.get("MSE")
                
                print(f"Réponse API - y_pred shape: {y_pred.shape if y_pred is not None else None}")
                print(f"MAE: {mae}, MSE: {mse}")
                
                return y_pred, y_test, mae, mse
                
            else:
                print(f"Erreur API: {response.status_code} - {response.text}")
                return None, None, None, None
                
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion à l'API: {str(e)}")
            return None, None, None, None
        except Exception as e:
            print(f"Erreur lors de l'appel API: {str(e)}")
            return None, None, None, None
