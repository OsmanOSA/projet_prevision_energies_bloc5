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

from meteostat import Point, hourly, Parameter, stations

load_dotenv()

# Bornes physiquement plausibles par source de production (cf. concat_all_data) --
# capacité installée française par filière, avec marge. Identifiées après un
# incident RTE réel : NUCLEAR est tombé à ~0 puis a pic à >100 000 MW sur
# certaines heures (sept-oct 2024), physiquement impossible pour un agrégat
# national (capacité nucléaire installée ~61 000 MW).
PRODUCTION_BOUNDS = {
    "NUCLEAR": (20000, 65000),
    "SOLAR": (0, 25000),
    "BIOMASS": (0, 2000),
    "WIND_ONSHORE": (0, 20000),
}

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
        raise ForecastingException(e, sys) 
    
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

        for i in range(lookback, len(arr) - horizon + 1):
            X.append(arr[i - lookback:i, :])
            y.append(arr[i:i + horizon, :])

        return np.array(X), np.array(y)

    except Exception as e:
        raise ForecastingException(e, sys) from e
    

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
        # limit_area="inside" : ne comble que les trous entourés de valeurs
        # connues. Sans ça, la dernière heure (pas encore publiée par RTE,
        # cf. le délai de publication éCO2mix) se voit dupliquer la valeur de
        # l'heure précédente au lieu de rester NaN -- une fausse observation
        # qui fausse silencieusement les comparaisons prévu/réalisé.
        df.fillna(value=df.interpolate(method="linear", limit_direction="both", limit_area="inside"), inplace=True)
        df.reset_index(inplace=True)
        df["timestamp"] = df["timestamp"].dt.strftime(date_format="%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame(df)
        return df
    
    except Exception as e:
        raise ForecastingException(e, sys)


def extract_conso_forecast_rte(target_date: str) -> pd.DataFrame:
    """Prévision officielle RTE J-1 de la consommation (API Consumption,
    type=D-1) : celle publiée par RTE la veille au soir pour `target_date`
    (YYYY-MM-DD). Sert uniquement de référence externe de crédibilité sur le
    dashboard (comparaison à notre propre modèle) — jamais utilisée en
    entraînement.

    RTE exige une fenêtre d'au moins ~2 jours pour ce type de requête ; on
    interroge donc [target-1j, target+1j] même si on ne veut que `target`.
    """
    try:
        URL_TOKEN = os.getenv("URL_TOKEN")
        CLIENT_ID = os.getenv("CLIENT_ID")
        CLIENT_SECRET = os.getenv("CLIENT_SECRET")

        response = requests.post(URL_TOKEN, data={"grant_type": "client_credentials"},
                                 auth=(CLIENT_ID, CLIENT_SECRET))
        token = response.json().get("access_token")

        BASE_URL = os.getenv("BASE_URL_CONSO_FORECAST")
        headers = {"Host": "digital.iservices.rte-france.com", "Authorization": f"Bearer {token}"}

        target = datetime.strptime(target_date, "%Y-%m-%d")
        start = target - timedelta(days=1)
        end = target + timedelta(days=1)

        url = f"{BASE_URL}&start_date={start.isoformat()}%2B02:00&end_date={end.isoformat()}%2B02:00"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Requête prévision RTE échouée : {response.text}")

        dates, values = [], []
        for entry in response.json().get("short_term", []):
            if entry.get("type") != "D-1":
                continue
            for point in entry.get("values", []):
                dates.append(point["start_date"])
                values.append(point["value"])

        if not dates:
            raise ValueError(f"Aucune prévision RTE (D-1) disponible pour {target_date}")

        df = pd.DataFrame({"timestamp": dates, "y_pred": values})
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
        df = df.set_index("timestamp").resample("h").mean()

        # La fenêtre de requête (imposée par RTE) déborde volontairement du
        # jour ciblé et peut ramener la prévision D-1 d'un jour voisin en
        # prime -> on ne garde que les 24h locales de `target_date`.
        local_index = df.index.tz_convert("Europe/Paris")
        target_day = pd.Timestamp(target_date).date()
        df = df.loc[local_index.date == target_day]

        # Naïf UTC : même convention que le reste du pipeline (cf. extract_conso).
        df.index = df.index.tz_localize(None)
        df.index.name = "timestamp"
        return df

    except Exception as e:
        raise ForecastingException(e, sys)


def extract_temperature(start_date, end_date, var_name="temp"):

  try:
    lon = float(os.getenv("LON"))
    lat = float(os.getenv("LAT"))

    location = Point(lat, lon)

    start = datetime.strptime(start_date + " 00:00", '%Y-%m-%d %H:%M')
    end = datetime.strptime(end_date + " 23:00", '%Y-%m-%d %H:%M')

    # meteostat 2.x : l'API par Point ne renvoie plus de données directement.
    # On résout le Point vers la station météo la plus proche, puis on
    # récupère la température horaire (Parameter.TEMP -> colonne "temp").
    nearby = stations.nearby(location)
    if nearby is None or nearby.empty:
        raise ValueError("Aucune station météo trouvée à proximité")
    station_id = str(nearby.index[0])

    dataframe = hourly(station_id, start, end, parameters=[Parameter.TEMP]).fetch()
    if dataframe is None or dataframe.empty:
        raise ValueError("Aucune donnée de température disponible pour la période demandée")

    dataframe.index.rename("timestamp", inplace=True)
    df = pd.DataFrame(dataframe[var_name]).astype(float)
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
            
            # cf. concat_all_data plus bas : limit_area="inside" pour ne pas
            # dupliquer la dernière heure pas encore publiée par RTE.
            df_prod.fillna(value=df_prod.interpolate(method="linear", limit_direction="both", limit_area="inside"), inplace=True)
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

        # Bornes physiquement plausibles avant toute agrégation : le nucléaire
        # français ne peut pas dépasser sa capacité installée ni tomber à 0
        # sur un agrégat national -- incident RTE identifié empiriquement
        # (~2 mois, sept-oct 2024, + quelques points isolés). Hors bornes ->
        # NaN, comme un trou normal, plutôt qu'une valeur silencieusement fausse.
        for col, (lo, hi) in PRODUCTION_BOUNDS.items():
            if col in df_prod.columns:
                out_of_bounds = (df_prod[col] < lo) | (df_prod[col] > hi)
                df_prod.loc[out_of_bounds, col] = np.nan

        df_conso = df_conso.reindex(full_index)

        df_prod = df_prod.loc[full_index[0]:full_index[-1], :]
        df_conso = df_conso.loc[full_index[0]:full_index[-1], :]
        df_temp = df_temp.loc[full_index[0]:full_index[-1], :]

        # Cibles en premier (les 4 sources de production), consommation ;
        # temp en dernier -- variable exogène, jamais prédite (cf.
        # model_trainer.py). Interpolation d'abord, puis production_total
        # dérivée de la somme des sources déjà interpolées -- jamais l'inverse :
        # interpoler la somme et les sources séparément peut les rendre
        # incohérentes entre elles (deux interpolations indépendantes ne se
        # raccordent pas forcément aux mêmes bornes de trou, d'où une dérive).
        df = pd.concat([df_prod, df_conso, df_temp], axis=1)
        # limit_area="inside" : comble uniquement les trous internes, jamais
        # en tête/fin de série -- la donnée la plus récente n'est souvent pas
        # encore publiée par RTE (délai éCO2mix) ; mieux vaut la laisser NaN
        # (-> NULL en base) qu'une valeur fabriquée qui fausserait les
        # comparaisons prévu/réalisé et les métriques de performance.
        df.fillna(value=df.interpolate(method='linear', limit_direction='both', limit_area='inside'), inplace=True)
        # min_count = nombre de filières : si l'une d'elles manque encore
        # (NaN, cf. limit_area="inside" ci-dessus), le total reste NaN au
        # lieu de sommer silencieusement les 3 autres -- une filière absente
        # (souvent NUCLEAR, >40 000 MW) fausserait sinon massivement le total
        # sans que rien ne le signale.
        sources = list(PRODUCTION_BOUNDS)
        df.insert(0, "production_total", df[sources].sum(axis=1, min_count=len(sources)))

        return df
    
    except Exception as e:
        raise ForecastingException(e, sys)
        
            