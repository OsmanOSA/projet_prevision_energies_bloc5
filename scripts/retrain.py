"""Réentraînement avec logique champion / challenger.

Préfigure le DAG `retrain_weekly` :
  1. Entraîne un *challenger* dans un dossier isolé (candidate_models/), sans
     toucher au *champion* servi en production (final_models/).
  2. Compare les deux modèles par backtest sur les observations récentes (MAE).
  3. Promeut le challenger seulement s'il est meilleur (au-delà d'une marge).
     L'ancien champion est archivé avant remplacement.

Usage :
    python -m scripts.retrain
"""

import os
import sys
import time
import shutil
from datetime import datetime

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline_prevision.logging.logger import logging
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.constant.training_pipeline import LOOKBACK
from pipeline_prevision.utils.main_utils.utils import load_object
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel
from pipeline_prevision.db import get_observations, log_run

FEATURES = ["temp", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale"]
CHAMPION_DIR = os.path.join(_ROOT, "final_models")
CANDIDATE_DIR = os.path.join(_ROOT, "candidate_models")
ARCHIVE_DIR = os.path.join(_ROOT, "models_archive")

EVAL_HORIZON = 24   # horizon de backtest pour la comparaison
EVAL_WINDOWS = 30   # nombre de fenêtres récentes évaluées


def _train_challenger():
    """Entraîne un challenger dans CANDIDATE_DIR (mêmes composants que main.py)."""
    from pipeline_prevision.entity.config_entity import (
        TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig,
        DataTransformationConfig, ModelTrainerConfig,
    )
    from pipeline_prevision.components.data_ingestion import DataIngestion
    from pipeline_prevision.components.data_validation import DataValidation
    from pipeline_prevision.components.data_transformation import DataTransformation
    from pipeline_prevision.components.model_trainer import ModelTrainer

    # Redirige la sortie du modèle vers le dossier candidat.
    os.environ["MODEL_OUTPUT_DIR"] = CANDIDATE_DIR

    cfg = TrainingPipelineConfig()
    ingestion = DataIngestion(DataIngestionConfig(cfg)).initiate_data_ingestion()
    validation = DataValidation(ingestion, DataValidationConfig(cfg)).initiate_data_validation()
    transformation = DataTransformation(validation, DataTransformationConfig(cfg)).initiate_data_transformation()
    ModelTrainer(model_trainer_config=ModelTrainerConfig(cfg),
                 data_transformation_artifact=transformation).initiate_model_trainer()


def _load_forecast_model(model_dir):
    preprocessor = load_object(os.path.join(model_dir, "preprocessor.pkl"))
    model = load_object(os.path.join(model_dir, "model.pkl"))
    return ForecastModel(preprocessor=preprocessor, model=model)


def _backtest_mae(forecast_model, feats):
    """MAE moyen du modèle sur les EVAL_WINDOWS fenêtres récentes de `feats`."""
    n_origins = len(feats) - (LOOKBACK + EVAL_HORIZON)
    if n_origins < 3:
        return float("inf")
    start = max(0, n_origins - EVAL_WINDOWS)
    errors = []
    for i in range(start, n_origins):
        x = feats[i:i + LOOKBACK].tolist()
        pred, _ = forecast_model.predict_multistep(x=x, n_futur=EVAL_HORIZON)
        actual = feats[i + LOOKBACK:i + LOOKBACK + EVAL_HORIZON]
        errors.append(np.abs(np.asarray(pred) - actual))
    return float(np.mean(errors)) if errors else float("inf")


def _promote_mlflow_alias():
    """Bascule l'alias `champion` du registre MLflow sur la version challenger.

    Non bloquant : la promotion des artefacts locaux ne doit pas échouer si le
    serveur de suivi est indisponible.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        name = os.getenv("MLFLOW_MODEL_NAME", "energia_forecasting_model")
        client = MlflowClient()
        version = client.get_model_version_by_alias(name, "challenger").version
        client.set_registered_model_alias(name, "champion", version)
        logging.info("MLflow : version %s promue champion", version)
        print(f"MLflow : version {version} promue champion")
    except Exception as e:
        logging.warning("Promotion de l'alias MLflow ignorée : %s", e)


def _promote():
    """Archive le champion courant puis copie le candidat en production."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    if os.path.isfile(os.path.join(CHAMPION_DIR, "model.pkl")):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copytree(CHAMPION_DIR, os.path.join(ARCHIVE_DIR, f"champion_{ts}"))
    os.makedirs(CHAMPION_DIR, exist_ok=True)
    for fname in ("model.pkl", "preprocessor.pkl", "metadata.json"):
        source = os.path.join(CANDIDATE_DIR, fname)
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(CHAMPION_DIR, fname))


def run(margin: float = 0.0) -> bool:
    """Entraîne un challenger et le promeut s'il bat le champion (marge relative).

    Retourne True si le challenger a été promu.
    """
    t0 = time.time()
    try:
        _train_challenger()

        obs = get_observations()
        if obs is None or obs.empty:
            raise ValueError("Aucune observation pour l'évaluation")
        feats = obs.sort_index()[FEATURES].to_numpy(dtype=float)

        challenger_mae = _backtest_mae(_load_forecast_model(CANDIDATE_DIR), feats)

        champion_exists = os.path.isfile(os.path.join(CHAMPION_DIR, "model.pkl"))
        champion_mae = _backtest_mae(_load_forecast_model(CHAMPION_DIR), feats) if champion_exists else float("inf")

        promoted = challenger_mae < champion_mae * (1 - margin)
        if promoted:
            _promote()               # artefacts locaux servis par local_forecaster
            _promote_mlflow_alias()  # alias champion au Model Registry

        duration = time.time() - t0
        message = (f"challenger MAE={challenger_mae:.2f} vs champion MAE={champion_mae:.2f} "
                   f"-> {'PROMU' if promoted else 'conservé'}")
        log_run("retrain_weekly", "success", rows=1, duration_s=duration, message=message)
        logging.info(message)
        print(message)
        return promoted

    except Exception as e:
        log_run("retrain_weekly", "failed", duration_s=time.time() - t0, message=str(e))
        logging.exception("Réentraînement en échec")
        raise ForecastingException(e, sys)


if __name__ == "__main__":
    run()
