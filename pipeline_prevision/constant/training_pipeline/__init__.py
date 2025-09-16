import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path


"""
    Defining common constant variable for training pipeline

"""

LOOKBACK: int = 36
HORIZON: int = 1

PIPELINE_NAME: str = "pipeline_prevision"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "datasets.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SUBMISSION_FILE_NAME: str = "valid.csv"

PATH_FILE_DATASET = Path("datasets")
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
SAVED_MODEL_DIR =os.path.join("saved_models")
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


## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_VALID_FILE_PATH: str = "valid.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

"""
Model trainer related constant start with MODEL TRAINER VAR NAME

"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXCEPTED_SCORE = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD = 0.05

TRAINING_BUCKET_NAME: str = "forecastingenergies"
