import sys
import os
import glob
import numpy as np
import pandas as pd


from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant import training_pipeline
from pipeline_prevision.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from pipeline_prevision.entity.config_entity import ModelTrainerConfig
from pipeline_prevision.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel
from pipeline_prevision.utils.ml_utils.metric.forecasting_metric import get_forecast_score

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import mlflow
import dagshub


dagshub.init(repo_owner='saidaliosman925', repo_name='projet_prevision_energies_bloc5', mlflow=True)

class ModelTrainer:

    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        
        try:

            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise ForecastingException(e, sys)
        
    def track_mlflow(self, best_model, 
                     forecastmetric):
        
        with mlflow.start_run():
            mae = forecastmetric.mae
            mse = forecastmetric.mse

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.sklearn.log_model(best_model,"model")
        

    
    def train_model(self, X_train, y_train, X_valid, y_valid):

        models = {
            "Gradient Boosting": MultiOutputRegressor(XGBRegressor(tree_method="hist")), 
            "LightGBM": MultiOutputRegressor(LGBMRegressor())
        }

        params = {
                "Gradient Boosting": {
                    'estimator__learning_rate': [0.1, 0.001],
                    'estimator__subsample': [0.7, 0.85],
                    'estimator__n_estimators': [50, 100]
                },
                "LightGBM": {
                    'estimator__learning_rate': [0.1, 0.001],
                    'estimator__subsample': [0.75, 0.85],
                    'estimator__n_estimators': [50, 100]
                }
            }
        
        model_report:dict=evaluate_models(X_train = X_train, y_train = y_train,
                                          X_valid = X_valid, y_valid = y_valid,
                                          models = models, param = params)
        
        ## To get best model score from dict
        best_model_score = min(sorted(model_report.values()))

        ## To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)

        forecast_train_metric = get_forecast_score(y_train, y_train_pred)

        ## Track the experiements with mlflow
        self.track_mlflow(best_model=best_model, 
                          forecastmetric=forecast_train_metric)

        y_valid_pred = best_model.predict(X_valid)
        forecast_valid_metric = get_forecast_score(y_true = y_valid, y_pred = y_valid_pred)

        ## Track the experiements with mlflow
        self.track_mlflow(best_model=best_model, 
                          forecastmetric=forecast_valid_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        forecast_model = ForecastModel(preprocessor=preprocessor, 
                                       model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, 
                    obj=forecast_model)
        
        save_object(file_path="final_models/model.pkl", obj=best_model)
        
        model_trainer_artifact = ModelTrainerArtifact(
                                 trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                 train_metric_artifact=forecast_train_metric, 
                                 valid_metric_artifact= forecast_valid_metric,
                                 )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")

        return model_trainer_artifact
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            submission_file_path = self.data_transformation_artifact.transformed_submission_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Loading training, valid and test array
            train_arr = load_numpy_array_data(train_file_path)
            valid_arr = load_numpy_array_data(submission_file_path)
            #test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1, :], train_arr[:, -1, :]
            X_valid, y_valid = valid_arr[:, :-1, :], valid_arr[:, -1, :]
            #X_test, y_test = test_arr[:, :-1, :], test_arr[:, -1, :]

            X_train = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
            X_valid = X_valid.reshape(-1, X_valid.shape[1]*X_valid.shape[2])
            #X_test = X_test.reshape(-1, X_test.shape[1])

            y_train = y_train.reshape(-1, y_train.shape[1])
            y_valid = y_valid.reshape(-1, y_valid.shape[1])
            #y_test = y_test.reshape(-1, y_test.shape[1])

            model_trainer_artifact=self.train_model(X_train, y_train, 
                                                    X_valid, y_valid)
            
            return model_trainer_artifact

        except Exception as e:
            raise ForecastingException(e, sys)