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
from pipeline_prevision.utils.main_utils.utils import load_object
from pipeline_prevision.utils.main_utils.feature_engineering import (
    add_target_features, build_series_by_target, seasonal_baseline,
)
from pipeline_prevision.utils.ml_utils.model.local_forecaster import TARGETS
from pipeline_prevision.db import get_observations, log_run

CHAMPION_DIR = os.path.join(_ROOT, "final_models")
CANDIDATE_DIR = os.path.join(_ROOT, "candidate_models")
ARCHIVE_DIR = os.path.join(_ROOT, "models_archive")

# Comparaison champion/challenger : moyenne des MAE sur quelques horizons
# représentatifs (mêmes unités MW pour les deux cibles -> agrégat comparable).
EVAL_HORIZONS = [1, 6, 12, 24]
EVAL_WINDOWS = 30


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


def _backtest_mae(model_dir, series_by_target, temp) -> float:
    """MAE moyen (5 cibles x horizons représentatifs) sur les EVAL_WINDOWS
    dernières origines réelles. Renvoie +inf si le modèle est absent ou
    incompatible (ex. ancien format, avant cette architecture) -> traité
    comme "pas de champion valide", donc le challenger est promu."""
    try:
        composite = load_object(os.path.join(model_dir, "model.pkl"))
        from pipeline_prevision.utils.main_utils.feature_engineering import build_origin_feature_frame

        errors = []
        for target in TARGETS:
            target_composite = composite[target]
            feature_columns = target_composite["feature_columns"]
            prefix = target_composite["prefix"]
            delta_col = f"{prefix}_delta_1"
            series = series_by_target[target]

            features_df, _ = build_origin_feature_frame(series_by_target, temp, target)

            for horizon in EVAL_HORIZONS:
                model = target_composite["models"][horizon]
                alpha = target_composite["alphas"][horizon]
                seasonal_weight = target_composite["seasonal_weights"][horizon]

                actual_future = series.shift(-horizon)
                mask = features_df[feature_columns].notna().all(axis=1) & actual_future.notna()
                frame = features_df.loc[mask].tail(EVAL_WINDOWS)
                if frame.empty:
                    continue
                y_true = actual_future.loc[frame.index].to_numpy()

                X = add_target_features(frame[feature_columns], horizon, delta_col)
                residual_pred = model.predict(X)
                persistence = frame[f"{prefix}_0"].to_numpy()
                seasonal_value = seasonal_baseline(frame, horizon, prefix)
                direct = persistence + alpha * residual_pred
                y_pred = (1 - seasonal_weight) * direct + seasonal_weight * seasonal_value
                y_pred = np.maximum(y_pred, 0.0)

                errors.append(np.abs(y_pred - y_true))

        if not errors:
            return float("inf")
        return float(np.mean(np.concatenate(errors)))

    except Exception as e:
        logging.warning("Backtest impossible sur %s (%s) -> traité comme absent", model_dir, e)
        return float("inf")


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
    for fname in ("model.pkl", "metadata.json"):
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
        obs = obs.sort_index()
        series_by_target = build_series_by_target(obs)
        temp = obs["temp"].astype(float)

        challenger_mae = _backtest_mae(CANDIDATE_DIR, series_by_target, temp)
        champion_mae = _backtest_mae(CHAMPION_DIR, series_by_target, temp)

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
