import os

from pathlib import Path
from datetime import datetime, timedelta


"""
    Defining common constant variable for training pipeline

"""

HOUR: int = 60*60
DAY: int = 24 * HOUR
WEEK: int = 7 * DAY
MONTH: int = 30.5 * DAY
YEAR: float = 365.25 * DAY


SIX_MONTHS = timedelta(days=30*6)
FIVE_MONTHS = timedelta(days=30*5)
TYPE_SOURCE = ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']

PIPELINE_NAME: str = "pipeline_prevision"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "datasets.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SUBMISSION_FILE_NAME: str = "valid.csv"

PATH_FILE_DATASET = Path("datasets")
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
MODEL_FILE_NAME = "model.pkl"

"""
    Data ingestion rlated constant start with DATA_INGESTION VAR NAME 

"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_TRAIN_VALID_SPLIT_RATIO: float = 0.1


"""
    Data Validation related constant start with DATA_VALIDATION VAR NAME

"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


PREPROCESSING_OBJECT_FILE_NAME = "preprocessor.pkl"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model trainer related constant start with MODEL TRAINER VAR NAME

"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXCEPTED_SCORE = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD = 0.05

TRAINING_BUCKET_NAME: str = "forecastingenergies"
