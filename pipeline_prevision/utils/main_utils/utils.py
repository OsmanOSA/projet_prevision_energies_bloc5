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
from zoneinfo import ZoneInfo
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
            # Décalage RÉEL de Paris à chaque borne (cf. `_offset_paris_url`) : figé
            # à +02:00, l'appel perdait 24 h par fenêtre en heure d'hiver — mesuré
            # 216 h renvoyées sur 240 demandées en janvier, contre 240/240 en
            # juillet. Sans conséquence tant qu'on n'ingérait que les derniers
            # jours en été ; destructeur sur un backfill pluriannuel.
            url = (f"{BASE_URL}&start_date={start.isoformat()}{_offset_paris_url(start)}"
                   f"&end_date={end.isoformat()}{_offset_paris_url(end)}")
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


# Largeur maximale d'une fenêtre acceptée par l'API Consumption de RTE, mesurée :
# 31 jours passent (2 976 points), 90 jours -> HTTP 400 CONSUMPTION_SHORTTERM_F04.
RTE_MAX_WINDOW_DAYS = 31


def extract_conso_forecast_rte_range(start_date: str, end_date: str,
                                     verbeux: bool = False) -> pd.DataFrame:
    """Prévisions RTE J-1 sur une PLAGE de jours (bornes incluses, AAAA-MM-JJ).

    Même contenu que `extract_conso_forecast_rte`, mais en une poignée de
    requêtes au lieu d'une par jour : l'API accepte des fenêtres allant jusqu'à
    31 jours (mesuré : 2 976 points en 0,7 s ; 90 jours -> HTTP 400
    CONSUMPTION_SHORTTERM_F04). Rétro-alimenter 3,5 ans revient ainsi à ~43
    appels au lieu de ~1 300.

    Sert le **repère externe** : la prévision RTE n'entre jamais dans nos
    features ni dans l'entraînement (cf. `scripts/fetch_rte_forecast.py`). Un
    historique long est indispensable pour savoir si un écart avec RTE est réel
    ou du bruit -- 6 origines n'autorisent aucune conclusion.

    Retourne un DataFrame horaire indexé en UTC naïf (convention du pipeline).
    """
    try:
        token_url = os.getenv("URL_TOKEN")
        reponse = requests.post(token_url, data={"grant_type": "client_credentials"},
                                auth=(os.getenv("CLIENT_ID"), os.getenv("CLIENT_SECRET")))
        token = reponse.json().get("access_token")
        base = os.getenv("BASE_URL_CONSO_FORECAST")
        entetes = {"Host": "digital.iservices.rte-france.com",
                   "Authorization": f"Bearer {token}"}

        debut = datetime.strptime(start_date, "%Y-%m-%d")
        # +1 j : la borne de fin est exclusive côté RTE, on veut `end_date` entier.
        fin = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        morceaux = []
        curseur = debut
        while curseur < fin:
            tranche_fin = min(curseur + timedelta(days=RTE_MAX_WINDOW_DAYS), fin)
            url = (f"{base}"
                   f"&start_date={curseur.isoformat()}{_offset_paris_url(curseur)}"
                   f"&end_date={tranche_fin.isoformat()}{_offset_paris_url(tranche_fin)}")
            r = requests.get(url, headers=entetes)
            if r.status_code != 200:
                raise ValueError(f"Requête RTE échouée ({curseur:%Y-%m-%d} -> "
                                 f"{tranche_fin:%Y-%m-%d}) : {r.text[:200]}")

            dates, valeurs = [], []
            for bloc in r.json().get("short_term", []):
                if bloc.get("type") != "D-1":
                    continue
                for point in bloc.get("values", []):
                    dates.append(point["start_date"])
                    valeurs.append(point["value"])
            if verbeux:
                print(f"  {curseur:%Y-%m-%d} -> {tranche_fin:%Y-%m-%d} : "
                      f"{len(dates)} points")
            if dates:
                morceaux.append(pd.DataFrame({"timestamp": dates, "y_pred": valeurs}))
            curseur = tranche_fin

        if not morceaux:
            raise ValueError(f"Aucune prévision RTE (D-1) sur {start_date} -> {end_date}")

        df = pd.concat(morceaux, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"],
                                         format="%Y-%m-%dT%H:%M:%S%z", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")].resample("h").mean().dropna()
        df.index = df.index.tz_localize(None)
        df.index.name = "timestamp"
        return df

    except Exception as e:
        raise ForecastingException(e, sys)


def _offset_paris_url(moment: datetime) -> str:
    """Décalage UTC de Paris à cette date, encodé pour une URL (`%2B01:00`).

    Le `+` est encodé en `%2B` : non échappé, il serait interprété comme un espace
    dans une chaîne de requête, et RTE rejetterait la borne.
    """
    decalage = moment.replace(tzinfo=ZoneInfo("Europe/Paris")).utcoffset()
    heures = int(decalage.total_seconds() // 3600)
    return f"%2B{heures:02d}:00"


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

        # Décalage RÉEL de Paris à chaque borne, jamais +02:00 en dur.
        #
        # L'offset était figé à l'heure d'été. En hiver (+01:00), RTE ne renvoyait
        # alors qu'UNE journée -- et pas celle demandée : le filtre sur la date
        # locale, juste en dessous, vidait donc le résultat. `dates` étant non
        # vide, le garde-fou `if not dates` ne se déclenchait pas : la fonction
        # rendait un DataFrame VIDE, `run()` persistait 0 ligne et journalisait un
        # SUCCÈS. Le DAG `fetch_rte_forecast` aurait ainsi échoué en silence de
        # fin octobre à fin mars, sans qu'aucune alerte ne se lève. Mesuré :
        # 96 points renvoyés avec +02:00 contre 192 avec +01:00 sur 2026-01-18,
        # 2025-12-10 et 2025-02-05.
        #
        # Les deux bornes sont calculées séparément : une fenêtre de 2 jours peut
        # enjamber un changement d'heure (derniers dimanches de mars et d'octobre).
        url = (f"{BASE_URL}"
               f"&start_date={start.isoformat()}{_offset_paris_url(start)}"
               f"&end_date={end.isoformat()}{_offset_paris_url(end)}")
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


def extract_temperature_france(start_date, end_date):
    """Température France pondérée sur 17 stations (colonne `temp_fr`).

    Complète `extract_temperature`, qui n'interroge qu'une station (Paris via
    LAT/LON). Mesuré par backtesting à origine glissante
    (`python -m scripts.evaluate_features`) : **-2,1 % de MAE supplémentaires
    sur la consommation, gain tenu sur 5 folds sur 6**, une fois le levier
    calendrier déjà appliqué.

    La station unique reste extraite en parallèle : les deux colonnes cohabitent
    dans `observations`, ce qui permet de rejouer une comparaison à tout moment.
    """
    try:
        from pipeline_prevision.utils.main_utils.temperature_france import (
            temperature_france,
        )

        start = datetime.strptime(start_date + " 00:00", "%Y-%m-%d %H:%M")
        end = datetime.strptime(end_date + " 23:00", "%Y-%m-%d %H:%M")
        serie, _, qualite = temperature_france(start, end)

        faibles = qualite.attrs.get("heures_faibles", 0)
        if faibles:
            # Trop de stations manquantes : la moyenne pondérée n'est plus
            # représentative. On le trace au lieu de le laisser passer -- une
            # pondération dégradée est indétectable en aval.
            logging.warning(
                "temp_fr : %d heures sous le seuil de poids disponible "
                "(stations manquantes, moyenne renormalisée)", faibles)

        return pd.DataFrame({"temp_fr": serie.astype(float)})

    except Exception as e:
        raise ForecastingException(e, sys)


def extract_temperature_openmeteo(start_date, end_date):
    """Couple Open-Meteo : `temp_fr_om` (observé) et `temp_fr_prev` (prévu J-1).

    Même indice national pondéré que `temp_fr` (mêmes 17 villes, mêmes poids),
    mais sur la grille Open-Meteo, et surtout décliné en DEUX séries issues de la
    MÊME source. C'est ce qui rend exploitable la feature qui porte le signal,
    l'écart `temp_prev(cible) - temp_om(origine)` : le biais entre grille
    Open-Meteo et stations Meteostat varie de 0,78 °C selon l'heure et de 0,63 °C
    selon le niveau de température, donc croiser les sources y injecterait ce
    décalage à la place du signal (cf. `prevision_temperature_france.py`).

    Sans ces colonnes, le modèle n'a aucune température future en entrée — l'angle
    mort qui nous faisait perdre 15 % de MAE contre RTE en bascule thermique.

    **Non bloquant, délibérément.** Une panne d'Open-Meteo ne doit pas emporter
    l'ingestion RTE/Meteostat, qui porte les cibles elles-mêmes : on trace et on
    rend un cadre vide. L'upsert protège l'existant (COALESCE, cf.
    `upsert_observations`) et la prochaine exécution horaire rattrape le trou.

    Sur le vintage de `temp_fr_prev` : rejouer une fenêtre déjà ingérée est sans
    danger. `previous_day1` est un fait historique figé — ce que le modèle météo
    prédisait la veille pour cette heure-là ne change plus. Les heures encore à
    venir reviennent NaN et l'upsert les laisse tranquilles.
    """
    try:
        from pipeline_prevision.utils.main_utils.prevision_temperature_france import (
            analyse_france, archive_prevision_france,
        )

        start = datetime.strptime(start_date + " 00:00", "%Y-%m-%d %H:%M")
        end = datetime.strptime(end_date + " 23:00", "%Y-%m-%d %H:%M")

        observe = analyse_france(start, end)
        prevu = archive_prevision_france(start, end, lead_jours=1)
        return pd.DataFrame({
            "temp_fr_om": observe.astype(float),
            "temp_fr_prev": prevu.astype(float),
        })

    except Exception as e:
        logging.warning(
            "Open-Meteo indisponible (%s) : `temp_fr_om`/`temp_fr_prev` non "
            "actualisées sur %s -> %s. Les valeurs déjà en base sont conservées ; "
            "si le trou persiste, les features de température cible se "
            "désactiveront et le modèle redeviendra aveugle à la météo future.",
            e, start_date, end_date)
        return pd.DataFrame(columns=["temp_fr_om", "temp_fr_prev"], dtype=float)


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
            # Même correctif que pour la consommation : offset réel de Paris et
            # non +02:00 figé, sinon 24 h perdues par fenêtre en heure d'hiver.
            url = (f"{BASE_URL}&start_date={start.isoformat()}{_offset_paris_url(start)}"
                   f"&end_date={end.isoformat()}{_offset_paris_url(end)}")
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

        # Marge d'un jour de chaque côté sur la température : RTE est interrogé
        # en heure locale (`...%2B02:00`) puis stocké en UTC naïf, alors que
        # Meteostat est interrogé directement en UTC. La fenêtre RTE commence
        # donc 2 h AVANT la fenêtre météo, et les 2 premières lignes de chaque
        # ingestion n'avaient aucune température : `limit_area="inside"` les
        # laisse NaN (ce sont des NaN de tête), et l'upsert écrasait alors des
        # valeurs correctes par NULL. Un seul NaN de `temp` invalidant 168 h de
        # features en aval (temp_lag_168), chaque ingestion sabotait la
        # prévision. Le surplus est sans effet : `df_temp` est réaligné plus bas
        # sur `full_index`, dérivé de la seule plage RTE.
        margin = timedelta(days=1)
        temp_start = (datetime.strptime(start_date, "%Y-%m-%d") - margin).strftime("%Y-%m-%d")
        temp_end = (datetime.strptime(end_date, "%Y-%m-%d") + margin).strftime("%Y-%m-%d")
        df_temp = extract_temperature(temp_start, temp_end)
        # Même marge d'un jour : `temp_fr` alimente les mêmes features de
        # température (lags jusqu'à 168 h), un NaN de tête aurait le même effet.
        df_temp = df_temp.join(
            extract_temperature_france(temp_start, temp_end), how="outer")
        # Couple Open-Meteo (observé + prévu J-1). Même marge, mêmes lags en aval.
        # Sans cette ligne les deux colonnes resteraient figées au backfill
        # initial : chaque nouvelle heure arriverait à NULL, et comme ce sont
        # précisément les heures récentes qui servent d'origine à la prévision,
        # le forecaster reculerait d'origine jour après jour.
        df_temp = df_temp.join(
            extract_temperature_openmeteo(temp_start, temp_end), how="outer")
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
        
            