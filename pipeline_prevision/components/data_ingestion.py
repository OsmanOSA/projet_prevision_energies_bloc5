import sys
import os
import glob
import numpy as np
import pandas as pd

from pathlib import Path
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant import training_pipeline
from pipeline_prevision.constant.training_pipeline import PATH_FILE_DATASET
## Configuration of the Data Ingestion Config
from pipeline_prevision.entity.config_entity import DataIngestionConfig
from pipeline_prevision.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self, 
                 data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise ForecastingException(e, sys)

    def _extract_from_csv(self, filename) -> pd.DataFrame:

        try:

            data = pd.read_csv(filename, sep=None, engine="python")
    
            # Indexer les dates
            data.set_index(data.columns[0], inplace=True)
            data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
            
            return data

        except Exception as e:
            raise ForecastingException(e, sys)
        
    def export_data_into_feature_store(self, 
                                       dataframe: pd.DataFrame):
        
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, header=True)

        except Exception as e:
            raise ForecastingException(e, sys)
    

    def split_data_as_train_test_valid(self, dataframe: pd.DataFrame):

        try: 

            train_set, test_set = train_test_split(dataframe, 
                                                   test_size=self.data_ingestion_config.train_test_split_ratio, 
                                                   random_state=0, shuffle=False)
            
            logging.info("Performed train test split on the dataframe.")

            train_set, valid_set = train_test_split(train_set, 
                                                   test_size=self.data_ingestion_config.train_valid_split_ratio, 
                                                   random_state=0, shuffle=False)
            
            logging.info("Performed train valid split on the train set.")

            logging.info("Existed split_data_as_train_test_valid method of DataIngestion class.")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train, valid and test set file path")

            train_set.to_csv(self.data_ingestion_config.training_file_path, header=True)
            test_set.to_csv(self.data_ingestion_config.submission_file_path, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, header=True)

            logging.info("Exported train, valid and test set file path")
            
        except Exception as e:
            raise ForecastingException(e, sys)
        pass

    def initiate_data_ingestion(self):

        dataframe_dict = {}

        try:
            for filename in PATH_FILE_DATASET.glob("*.csv"):
                df = self._extract_from_csv(filename)
                series_name = filename.stem  # nom du fichier sans extension
                dataframe_dict[series_name] = df.rename(columns={"Production": series_name})

            # Fusionner toutes les séries sur l'index Date
            dataframe = pd.concat(dataframe_dict.values(), axis=1, join="inner")

            self.export_data_into_feature_store(dataframe)

            self.split_data_as_train_test_valid(dataframe)

            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                submission_file_path=self.data_ingestion_config.submission_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            return dataingestionartifact

        except Exception as e:
            raise ForecastingException(e, sys)


