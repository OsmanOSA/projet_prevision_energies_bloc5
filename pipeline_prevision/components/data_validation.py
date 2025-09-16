import sys
import os
import glob
import numpy as np
import pandas as pd


from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant import training_pipeline
from pipeline_prevision.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from pipeline_prevision.entity.config_entity import DataValidationConfig
from pipeline_prevision.constant.training_pipeline import SCHEMA_FILE_PATH, PATH_FILE_DATASET
from pipeline_prevision.utils.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp


class DataValidation:

    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        
        try:
            
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            

        except Exception as e:
            raise ForecastingException(e, sys)

    @staticmethod    
    def read_data(file_path) -> pd.DataFrame:
        
        try:
            
            return pd.read_csv(file_path, sep=None, engine="python", parse_dates=["timestamp"], index_col="timestamp")
            
        
        except Exception as e:
            raise ForecastingException(e, sys)
        

    def validate_number_of_columns(self, 
                                   dataframe: pd.DataFrame) -> bool:
        try:
            
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns : {number_of_columns}")
            logging.info(f"Data frmae has columns : {len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
                return True
            else:
                return False
        except Exception as e:
            raise ForecastingException(e, sys)

    def detect_dataset_drift(self, 
                              base_df: pd.DataFrame, 
                              current_df: pd.DataFrame, 
                              threshold: float = 0.05) -> bool:
        
        try: 
            status = True
            report = {}

            for column in base_df.columns:
                
                is_same_distribution = ks_2samp(base_df[column], current_df[column])

                if threshold <= is_same_distribution.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update(
                    {column: {
                        "pvalue": float(is_same_distribution.pvalue),
                        "drift_status": is_found
                    }}
                )

                drift_report_file_path = self.data_validation_config.drift_report_file_path

                # Create directory
                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path, exist_ok=True)
                write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise ForecastingException(e, sys)
        


    def initiate_data_validation(self) -> DataValidationArtifact:

        try: 
            train_file_path = self.data_ingestion_artifact.trained_file_path
            submission_file_path = self.data_ingestion_artifact.submission_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read the data from train, submission and test
            train_dataframe = self.read_data(train_file_path)
            submission_dataframe = self.read_data(submission_file_path)
            test_dataframe = self.read_data(test_file_path)

            # Validate number of columns in dataframe 
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            
            if not status:
                error_message = "Train dataframe does not contain all columns. \n"

            status = self.validate_number_of_columns(dataframe=submission_dataframe)

            if not status:
                error_message = "Submission dataframe does not contain all columns. \n"


            status = self.validate_number_of_columns(dataframe=test_dataframe)

            if not status:
                error_message = "Test dataframe does not contain all columns. \n"

            # Lets check data drift
            status = self.detect_dataset_drift(base_df=test_dataframe, 
                                                current_df=submission_dataframe)
            
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, 
                header = True
            )

            submission_dataframe.to_csv(
                self.data_validation_config.valid_submission_file_path,
                header = True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,
                header = True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status = status, 
                valid_train_file_path = self.data_ingestion_artifact.trained_file_path,
                valid_submission_file_path = self.data_ingestion_artifact.submission_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path, 
                invalid_train_file_path = None, 
                invalid_submission_file_path = None, 
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise ForecastingException(e, sys)
