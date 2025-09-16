import sys
import os
import numpy as np
import pandas as pd
import yaml
import pickle
import dill

from typing import Tuple, List, Literal
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

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