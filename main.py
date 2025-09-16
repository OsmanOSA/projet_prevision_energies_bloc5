import os 
import sys

from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.entity.config_entity import (DataIngestionConfig, TrainingPipelineConfig, 
                                                     DataValidationConfig, DataTransformationConfig)
from pipeline_prevision.components.data_ingestion import DataIngestion
from pipeline_prevision.components.data_validation import DataValidation
from pipeline_prevision.components.data_transformation import DataTransformation

from pipeline_prevision.components.model_trainer import ModelTrainer
from pipeline_prevision.entity.config_entity import ModelTrainerConfig


if __name__ == "__main__":

    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion =DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion ")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data initiation completed")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, 
                       data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        logging.info("Data transformation started")
        data_transformation = DataTransformation(data_validation_artifact, 
                           data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)

        logging.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")
        
    except Exception as e:
        raise ForecastingException(e, sys)
