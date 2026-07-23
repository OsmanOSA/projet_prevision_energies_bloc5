import sys
import os
import pandas as pd

from pathlib import Path
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant import training_pipeline
from pipeline_prevision.constant.training_pipeline import PATH_FILE_DATASET
## Configuration of the Data Ingestion Config
from pipeline_prevision.entity.config_entity import DataIngestionConfig
from pipeline_prevision.entity.artifact_entity import DataIngestionArtifact


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

            if not isinstance(dataframe.index, pd.DatetimeIndex):
                raise ValueError("Le découpage temporel exige un DatetimeIndex")

            dataframe = dataframe.sort_index()
            if dataframe.index.hasnans:
                raise ValueError("Des horodatages invalides (NaT) sont présents")
            if dataframe.index.has_duplicates:
                duplicates = int(dataframe.index.duplicated().sum())
                raise ValueError(f"{duplicates} horodatage(s) dupliqué(s) détecté(s)")

            n_rows = len(dataframe)
            test_size = max(1, int(n_rows * self.data_ingestion_config.train_test_split_ratio))
            train_valid = dataframe.iloc[:-test_size]
            test_set = dataframe.iloc[-test_size:]
            valid_size = max(1, int(len(train_valid) * self.data_ingestion_config.train_valid_split_ratio))
            train_set = train_valid.iloc[:-valid_size]
            valid_set = train_valid.iloc[-valid_size:]

            if min(len(train_set), len(valid_set), len(test_set)) == 0:
                raise ValueError(
                    "Dataset insuffisant pour créer trois partitions temporelles non vides"
                )

            logging.info(
                "Découpage temporel: train=%s (%s -> %s), validation=%s (%s -> %s), "
                "test=%s (%s -> %s)",
                len(train_set), train_set.index.min(), train_set.index.max(),
                len(valid_set), valid_set.index.min(), valid_set.index.max(),
                len(test_set), test_set.index.min(), test_set.index.max(),
            )

            logging.info("Existed split_data_as_train_test_valid method of DataIngestion class.")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train, valid and test set file path")

            train_set.to_csv(self.data_ingestion_config.training_file_path, header=True)
            valid_set.to_csv(self.data_ingestion_config.submission_file_path, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, header=True)

            logging.info("Exported train, valid and test set file path")
            return train_set, valid_set, test_set
            
        except Exception as e:
            raise ForecastingException(e, sys)

    def initiate_data_ingestion(self):

        dataframe_dict = {}

        try:
            for filename in PATH_FILE_DATASET.glob("*.csv"):
                df = self._extract_from_csv(filename)
                series_name = filename.stem  # nom du fichier sans extension
                dataframe_dict[series_name] = df.rename(columns={"Production": series_name})

            # Fusionner toutes les séries sur l'index Date
            if not dataframe_dict:
                raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {PATH_FILE_DATASET}")

            dataframe = pd.concat(dataframe_dict.values(), axis=1, join="inner").sort_index()

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


