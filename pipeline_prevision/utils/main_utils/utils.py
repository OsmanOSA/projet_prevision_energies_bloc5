import sys
import os
import numpy as np
import pandas as pd
import requests
import yaml
import json
import pickle
import dill

from datetime import datetime, timedelta
from typing import Tuple, List, Literal
from dotenv import load_dotenv
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant.training_pipeline import SIX_MONTHS, FIVE_MONTHS, TYPE_SOURCE

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from meteostat import Point, Hourly

load_dotenv()

def read_yaml_file(file_path: str) -> dict:

    try: 
        
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise ForecastingException(e, sys)
    
def write_yaml_file(file_path: str, 
                    content: object, 
                    replace: bool = False) -> None:
    try:

        if replace: 
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise ForecastingException(e, sys)


def save_numpy_array_data(file_path: str,
                           array: np.ndarray):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise ForecastingException(e, sys) from e
    

def save_object(file_path: str,
                 obj: object) -> None:

    try:

        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")

    except Exception as e:
        raise ForecastingException(e, sys) from e

def load_data(filename: str) -> pd.DataFrame:

        try:

            data = pd.read_csv(filename, sep=None, engine="python")
    
            # Indexer les dates
            data.set_index(data.columns[0], inplace=True)
            pd.to_datetime(data.index, format="%d/%m/%y %H:%M:%S", inplace=True)
            
            return data

        except Exception as e:
            raise ForecastingException(e, sys)

def load_object(file_path: str) -> object:

    try: 
        
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} is not exists")
        
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise ForecastingException(sys, e) 
    
def load_numpy_array_data(file_path: str):

    try: 
        
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} is not exists")
        
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise ForecastingException(sys, e) 
    
    
def window_generator(data: np.ndarray, 
                    lookback: int, 
                    horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sliding windows of input and target data for forecasting models 
    based on the specified lookback and prediction horizon.

    Parameters
    ----------
    data : np.ndarray
        Source data used to create the windows.
    lookback : int
        Number of time steps to look back for each input sequence.
    horizon : int
        Number of time steps to predict.

    Returns
    -------
    X : np.ndarray
        Input data sequences for the model.
    y : np.ndarray
        Target prediction sequences for the model.
    """
    try:
        X, y = [], []

        arr = data.values if isinstance(data, pd.DataFrame) else data

        for i in range(lookback, len(arr) - horizon):
            X.append(arr[i - lookback:i, :])
            y.append(arr[i:i + horizon, :])

        return np.array(X), np.array(y)

    except Exception as e:
        raise ForecastingException(e, sys) from e
    

def evaluate_models(X_train, y_train, 
                    X_valid, y_valid, 
                    models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3, error_score='raise')
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_valid)

            train_model_score = mean_absolute_error(y_train, y_train_pred)

            test_model_score = mean_absolute_error(y_valid, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise ForecastingException(e, sys)
    
    
def daterange(start_date, end_date, delta):
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + delta
        yield current_date, min(next_date, end_date)
        current_date = next_date


def extract_conso(start_date: str, end_date: str):

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    dates = []
    consommation = []

    data = {
            "grant_type": "client_credentials"
            }
    
    try:

        URL_TOKEN = os.getenv("URL_TOKEN")
        CLIENT_ID = os.getenv("CLIENT_ID")
        CLIENT_SECRET = os.getenv("CLIENT_SECRET")

        response = requests.post(URL_TOKEN, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
        token = response.json().get("access_token")

        BASE_URL = os.getenv("BASE_URL_CONSO")
        headers = {
                "Host": "digital.iservices.rte-france.com",
                "Authorization": f"Bearer {token}"
                    }

        for start, end in daterange(start_date, end_date, SIX_MONTHS):
            url = f"{BASE_URL}&start_date={start.isoformat()}%2B02:00&end_date={end.isoformat()}%2B02:00"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                for entry in response.json()['short_term'][0]['values']:
                    dates.append(entry['start_date'])
                    consommation.append(entry['value'])
            else:
                print("Request failed")

        df = pd.DataFrame({"timestamp": dates, "consommation_totale": consommation})
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
        df = df.set_index("timestamp").resample("h").mean()
        df.fillna(value=df.interpolate(method="linear", limit_direction="both"), inplace=True)
        df.reset_index(inplace=True)
        df["timestamp"] = df["timestamp"].dt.strftime(date_format="%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame(df)
        return df
    
    except Exception as e:
        raise ForecastingException(e, sys)

    

def extract_temperature(start_date, end_date, var_name="temp"):

  try:
    LON = os.getenv("LON")
    LAT = os.getenv("LAT")
    lon = float(LON)
    lat = float(LAT)
    
    location = Point(lat, lon)
    start_date = start_date + " 00:00"
    end_date = end_date + " 23:00"
    
    start = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
    end = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

    dataframe = Hourly(location, start, end)
    dataframe = dataframe.fetch()
    dataframe.index.rename("timestamp", inplace=True)
    df = dataframe[var_name]
    df = pd.DataFrame(df).astype(float)
    return df

  except Exception as e:
    raise ForecastingException(e, sys)


def extract_production(start_date, end_date):
    
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []

        data = {
                "grant_type": "client_credentials"
                }

        URL_TOKEN = os.getenv("URL_TOKEN")
        CLIENT_ID_2 = os.getenv("CLIENT_ID_2")
        CLIENT_SECRET_2 = os.getenv("CLIENT_SECRET_2")

        response = requests.post(URL_TOKEN, data=data, auth=(CLIENT_ID_2, CLIENT_SECRET_2))
        token = response.json().get("access_token")

        BASE_URL = os.getenv("BASE_URL_PROD")
        headers = {
                "Host": "digital.iservices.rte-france.com",
                "Authorization": f"Bearer {token}"
                    }

        for start, end in daterange(start_date, end_date, FIVE_MONTHS):
            url = f"{BASE_URL}&start_date={start.isoformat()}%2B02:00&end_date={end.isoformat()}%2B02:00"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                
                try:
                    json_data = response.json()
                

                    if isinstance(json_data, dict) and "actual_generations_per_production_type" in json_data:
                        production_data = json_data["actual_generations_per_production_type"]
                        
                        for item in production_data:
                            prod_type = item.get("production_type")
                            if prod_type and 'values' in item:
                                values = item["values"]
                                
                                for value_entry in values:
                                    if value_entry.get("start_date") and value_entry.get("value") is not None:
                                        all_data.append({
                                            "timestamp": value_entry.get("start_date"),
                                            "production_type": prod_type,
                                            "value": value_entry.get("value")
                                        })
                    
                except json.JSONDecodeError as e:
                    raise ForecastingException(e, sys)
           
        if all_data:
            df = pd.DataFrame(all_data)
            
            df_prod = df.pivot_table(
                index='timestamp', 
                columns='production_type', 
                values='value', 
                aggfunc='first'
            )

            # Vérifier que les colonnes TYPE_SOURCE existent
            available_columns = [col for col in TYPE_SOURCE if col in df_prod.columns]
            if available_columns:
                df_prod = df_prod[available_columns]
    
            df_prod = df_prod.reset_index()
            df_prod['timestamp'] = pd.to_datetime(df_prod["timestamp"],  
                                                  format="%Y-%m-%dT%H:%M:%S%z", utc=True)
            
            d_prod = pd.DataFrame(df_prod).set_index('timestamp')
            
            df_prod.fillna(value=df_prod.interpolate(method="linear", limit_direction="both"), inplace=True)
            df_prod.reset_index(inplace=True)
            
            df_prod["timestamp"] = df_prod["timestamp"].dt.strftime(date_format="%Y-%m-%d %H:%M:%S")
            df_prod = pd.DataFrame(df_prod)
            
            return df_prod
        
    except Exception as e:
        raise ForecastingException(e, sys)
            

def concat_all_data(start_date, end_date):
    
    try:
        df_temp = extract_temperature(start_date, end_date)
        df_prod = extract_production(start_date, end_date)
        df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"], format="%Y-%m-%d %H:%M:%S")
        df_prod.set_index("timestamp", inplace=True)
        df_conso = extract_conso(start_date, end_date)
        df_conso["timestamp"] = pd.to_datetime(df_conso["timestamp"], format="%Y-%m-%d %H:%M:%S")
        df_conso.set_index("timestamp", inplace=True)
        
        # Supprimer les doublons
        df_temp = df_temp[~df_temp.index.duplicated()]
        df_prod = df_prod[~df_prod.index.duplicated()]
        df_conso = df_conso[~df_conso.index.duplicated()]
        
        # Trier les index
        df_temp.sort_index(inplace=True)
        df_prod.sort_index(inplace=True)
        df_conso.sort_index(inplace=True)
        
        full_index = pd.date_range(start=df_prod.index[0], end=df_prod.index[-1], freq="h")
        full_index.name = "timestamp"
        
        df_temp = df_temp.reindex(full_index)
        df_prod = df_prod.reindex(full_index)
        df_conso = df_conso.reindex(full_index)
        
        df_prod = df_prod.loc[df_prod.index[0]:df_prod.index[-1], :]
        df_conso = df_conso.loc[df_prod.index[0]:df_prod.index[-1], :]
        df_temp = df_temp.loc[df_prod.index[0]:df_prod.index[-1], :]
        
        df = pd.concat([df_temp, df_prod, df_conso], axis=1)
        df.fillna(value=df.interpolate(method='linear', limit_direction='both'), inplace=True)
    
        return df
    
    except Exception as e:
        raise ForecastingException(e, sys)
        
            