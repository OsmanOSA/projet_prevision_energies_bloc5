import sys
import os
import glob
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from pipeline_prevision.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from pipeline_prevision.constant.training_pipeline import LOOKBACK, HORIZON
from pipeline_prevision.utils.main_utils.utils import save_numpy_array_data, save_object, window_generator
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant import training_pipeline
from pipeline_prevision.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from pipeline_prevision.entity.config_entity import DataTransformationConfig


class DataTransformation:

    def __init__(self, 
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
                 ):
        
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise ForecastingException(e, sys)
        

    @staticmethod    
    def read_data(file_path) -> pd.DataFrame:
        
        try:
            
            return pd.read_csv(file_path, sep=None, engine="python", parse_dates=["timestamp"], index_col="timestamp")
        
        except Exception as e:
            raise ForecastingException(e, sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Parameters
        ----------
        cls: DataTransformation

        Returns
        -------
        A Pipeline object

        """
        logging.info(
            "Entered get_data_transformer_object method of Transformation class"
        )

        try:
           
           imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
           scaler:MinMaxScaler=MinMaxScaler()
           logging.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
           processor:Pipeline=Pipeline([("imputer", imputer), 
                                        ("scaler", scaler)])

           return processor
        
        except Exception as e:
            raise ForecastingException(e,sys)    

    def initiate_data_transformation(self) -> DataValidationArtifact:

        logging.info("Entered initiate_data_transformation method of DataTransformation class")

        try:

            logging.info("Starting data transformation")
            
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            submission_df = DataTransformation.read_data(self.data_validation_artifact.valid_submission_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            # Impute missing values
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(train_df)
            transformed_train_df = preprocessor_object.transform(train_df)
            transformed_submission_df = preprocessor_object.transform(submission_df)
            transformed_test_df = preprocessor_object.transform(test_df)

            # Generate windows sliding 
            X_train, y_train = window_generator(transformed_train_df, 
                                                lookback=LOOKBACK, horizon=HORIZON)
            X_valid, y_valid = window_generator(transformed_submission_df, 
                                                lookback=LOOKBACK, horizon=HORIZON)
            X_test, y_test = window_generator(transformed_test_df, 
                                              lookback=LOOKBACK, horizon=HORIZON)

            train_arr = np.concatenate([X_train, y_train], axis=1)  

            valid_arr = np.concatenate([X_valid, y_valid], axis=1)
            test_arr = np.concatenate([X_test, y_test], axis=1)


            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, 
                                  train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_submission_file_path, 
                                  valid_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, 
                                  test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, 
                        preprocessor_object)
            
            save_object(file_path="final_models/preprocessor.pkl", obj=preprocessor_object)
            
            #preparing artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                 transformed_submission_file_path = self.data_transformation_config.transformed_submission_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise ForecastingException(e, sys)
        
