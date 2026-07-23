import sys
import os
import glob
import hashlib
import json
import platform
import subprocess
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from importlib.metadata import version as package_version


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
FEATURE_NAMES = ["temp", "SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR", "consommation_totale"]

# NB : mlflow et hyperopt sont importés paresseusement (dans les méthodes qui
# les utilisent) afin que l'import du module ne dépende pas de ces librairies
# lourdes — utile pour les environnements légers (image Airflow).

class ModelTrainer:

    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        
        try:

            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise ForecastingException(e, sys)
        
    def track_mlflow(self, best_model, model_family,
                     train_metric, valid_metric, X_sample, y_pred_sample):
        """Enregistre l'expérience dans MLflow : un run complet par entraînement.

        Log : hyperparamètres retenus (TPE), métriques train + validation,
        modèle sérialisé avec signature, et tags de contexte. Le modèle est
        enregistré au Model Registry et reçoit l'alias `challenger`
        (les *stages* MLflow sont supprimés depuis la 3.x — on utilise les alias).

        Retourne (run_id, version) ou (None, None) si le suivi est indisponible.
        """
        import mlflow
        from mlflow.models import infer_signature

        # Console Windows (cp1252) : MLflow écrit des emojis en fin de run, ce
        # qui lève un UnicodeEncodeError. On sécurise la sortie standard.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "energia_forecasting"))
        registered_name = os.getenv("MLFLOW_MODEL_NAME", "energia_forecasting_model")

        run_name = f"{model_family}-{datetime.now():%Y%m%d_%H%M%S}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({
                "model_family": model_family,
                # Pourquoi cet entraînement a été lancé (dégradation, manuel...)
                "trigger": os.getenv("RETRAIN_TRIGGER", "manual"),
                "lookback": training_pipeline.LOOKBACK,
                "horizon": training_pipeline.HORIZON,
            })

            # Hyperparamètres retenus par l'optimisation bayésienne
            params = {
                k: v for k, v in best_model.get_params().items()
                if k.startswith("estimator__") and isinstance(v, (int, float, str, bool))
            }
            if params:
                mlflow.log_params(params)

            mlflow.log_metrics({
                "train_mae": float(train_metric.mae),
                "train_mse": float(train_metric.mse),
                "valid_mae": float(valid_metric.mae),
                "valid_mse": float(valid_metric.mse),
            })

            signature = infer_signature(X_sample, y_pred_sample)
            info = mlflow.sklearn.log_model(
                best_model,
                name="model",
                signature=signature,
                input_example=X_sample[:2],
                registered_model_name=registered_name,
                # skops (défaut MLflow 3.x) refuse les types LightGBM/XGBoost
                # ("untrusted types") : cloudpickle fonctionne quelle que soit
                # la famille de modèle retenue.
                serialization_format="cloudpickle",
            )

            version = getattr(info, "registered_model_version", None)
            if version is not None:
                mlflow.tracking.MlflowClient().set_registered_model_alias(
                    registered_name, "challenger", version)
                logging.info("MLflow: %s v%s enregistré (alias challenger)", registered_name, version)

            return run.info.run_id, version

    def train_model(self, X_train, y_train, X_valid, y_valid,
                    X_test, y_test, persistence_test):

        from hyperopt import hp  # import paresseux (requis pour l'optimisation TPE)

        models = {
            "Gradient Boosting": MultiOutputRegressor(
                XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1)
            ),
            "LightGBM": MultiOutputRegressor(
                LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
            )
        }

        # Espaces de recherche pour l'optimisation bayésienne (hyperopt / TPE).
        params = {
                "Gradient Boosting": {
                    'estimator__learning_rate': hp.uniform('gb_learning_rate', 0.001, 0.1),
                    'estimator__subsample': hp.uniform('gb_subsample', 0.7, 0.85),
                    'estimator__n_estimators': hp.choice('gb_n_estimators', [50, 100, 200])
                },
                "LightGBM": {
                    'estimator__learning_rate': hp.uniform('lgbm_learning_rate', 0.001, 0.1),
                    'estimator__subsample': hp.uniform('lgbm_subsample', 0.75, 0.85),
                    'estimator__n_estimators': hp.choice('lgbm_n_estimators', [50, 100, 200])
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
        y_valid_pred = best_model.predict(X_valid)
        y_test_pred = best_model.predict(X_test)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        # Les cibles sont normalisées (MinMaxScaler) : on inverse la mise à
        # l'échelle pour obtenir des métriques en unités physiques (MW, °C),
        # seules comparables au seuil de dégradation suivi dans Grafana.
        scaler = preprocessor.named_steps["scaler"]
        forecast_train_metric = get_forecast_score(
            scaler.inverse_transform(y_train), scaler.inverse_transform(y_train_pred))
        forecast_valid_metric = get_forecast_score(
            y_true=scaler.inverse_transform(y_valid),
            y_pred=scaler.inverse_transform(y_valid_pred))
        test_true = scaler.inverse_transform(y_test)
        test_pred = scaler.inverse_transform(y_test_pred)
        persistence_pred = scaler.inverse_transform(persistence_test)
        forecast_test_metric = get_forecast_score(y_true=test_true, y_pred=test_pred)

        per_feature_test = {}
        for index, feature in enumerate(FEATURE_NAMES):
            model_error = test_pred[:, index] - test_true[:, index]
            baseline_error = persistence_pred[:, index] - test_true[:, index]
            model_mae = float(np.mean(np.abs(model_error)))
            baseline_mae = float(np.mean(np.abs(baseline_error)))
            per_feature_test[feature] = {
                "unit": "°C" if feature == "temp" else "MW",
                "model_mae": model_mae,
                "persistence_mae": baseline_mae,
                "model_rmse": float(np.sqrt(np.mean(model_error ** 2))),
                "model_bias": float(np.mean(model_error)),
                "beats_persistence": bool(model_mae < baseline_mae),
            }

        ## Suivi MLflow : un seul run par entraînement (train + validation).
        ## Non bloquant : si le serveur de suivi est indisponible, l'entraînement
        ## et la sauvegarde du modèle doivent aboutir malgré tout.
        try:
            self.track_mlflow(best_model=best_model,
                              model_family=best_model_name,
                              train_metric=forecast_train_metric,
                              valid_metric=forecast_valid_metric,
                              X_sample=X_train, y_pred_sample=y_train_pred)
        except Exception as mlflow_error:
            logging.warning("Suivi MLflow ignoré: %s", mlflow_error)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        forecast_model = ForecastModel(preprocessor=preprocessor, 
                                       model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, 
                    obj=forecast_model)
        
        # Un entrainement produit un candidat sans ecraser le champion servi.
        # scripts/retrain.py applique ensuite la promotion explicite.
        output_dir = os.getenv("MODEL_OUTPUT_DIR", "candidate_models")
        model_path = os.path.join(output_dir, "model.pkl")
        preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
        save_object(file_path=model_path, obj=best_model)
        # Le préprocesseur est indispensable à l'inférence locale (local_forecaster)
        save_object(file_path=preprocessor_path, obj=preprocessor)

        with open(model_path, "rb") as model_file:
            model_sha256 = hashlib.sha256(model_file.read()).hexdigest()
        dataset_hashes = {}
        for dataset_path in sorted(training_pipeline.PATH_FILE_DATASET.glob("*.csv")):
            with open(dataset_path, "rb") as dataset_file:
                dataset_hashes[dataset_path.name] = hashlib.sha256(
                    dataset_file.read()
                ).hexdigest()
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.SubprocessError):
            git_commit = None

        metadata = {
            "model_version": f"sha256:{model_sha256[:12]}",
            "model_sha256": model_sha256,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "dataset_sha256": dataset_hashes,
            "feature_order": FEATURE_NAMES,
            "lookback_hours": training_pipeline.LOOKBACK,
            "native_horizon_hours": training_pipeline.HORIZON,
            "model_family": best_model_name,
            "hyperparameters": {
                key: value
                for key, value in best_model.get_params().items()
                if key.startswith("estimator__")
                and isinstance(value, (str, int, float, bool, type(None)))
            },
            "test_protocol": (
                "holdout chronologique final, jamais utilisé pour la sélection"
            ),
            "test_metrics_by_feature": per_feature_test,
            "legacy_macro_metric_warning": (
                "Les agrégats MAE/MSE mélangent °C et MW ; utiliser les métriques "
                "par variable pour toute décision."
            ),
            "runtime": {
                "python": platform.python_version(),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scikit_learn": package_version("scikit-learn"),
            },
        }
        metadata_file_path = os.path.join(output_dir, "metadata.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(metadata_file_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

        artifact_metadata_path = os.path.join(model_dir_path, "metadata.json")
        with open(artifact_metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=forecast_train_metric,
            valid_metric_artifact=forecast_valid_metric,
            test_metric_artifact=forecast_test_metric,
            metadata_file_path=artifact_metadata_path,
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
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1, :], train_arr[:, -1, :]
            X_valid, y_valid = valid_arr[:, :-1, :], valid_arr[:, -1, :]
            X_test, y_test = test_arr[:, :-1, :], test_arr[:, -1, :]
            persistence_test = X_test[:, -1, :].copy()

            X_train = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
            X_valid = X_valid.reshape(-1, X_valid.shape[1] * X_valid.shape[2])
            X_test = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

            y_train = y_train.reshape(-1, y_train.shape[1])
            y_valid = y_valid.reshape(-1, y_valid.shape[1])
            y_test = y_test.reshape(-1, y_test.shape[1])

            model_trainer_artifact = self.train_model(
                X_train, y_train, X_valid, y_valid,
                X_test, y_test, persistence_test,
            )
            
            return model_trainer_artifact

        except Exception as e:
            raise ForecastingException(e, sys)