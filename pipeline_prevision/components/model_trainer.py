import sys
import os
import gc
import hashlib
import json
import platform
import subprocess
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from importlib.metadata import version as package_version

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant import training_pipeline
from pipeline_prevision.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from pipeline_prevision.entity.config_entity import ModelTrainerConfig
from pipeline_prevision.utils.main_utils.utils import save_object, load_object
from pipeline_prevision.utils.main_utils.feature_engineering import (
    HORIZON_MAX, ALPHA_GRID, SEASONAL_WEIGHT_GRID, TARGET_PREFIXES, add_target_features, seasonal_baseline,
)

# NB : optuna est importé paresseusement (dans _tune_hyperparameters) afin que
# l'import du module ne dépende pas de cette librairie lourde — utile pour les
# environnements légers (image Airflow).

# Les 4 sources de production (SOLAR/BIOMASS/WIND_ONSHORE/NUCLEAR) + la
# consommation -- production_total redevient une valeur dérivée (somme des 4
# sources), plus une cible entraînée, cf. scripts/forecast.py.
FEATURE_NAMES = list(TARGET_PREFIXES)

# Recherche Optuna : un seul run par cible, partagé sur les 24 horizons
# (le tuning se ferait sinon 24x -> trop coûteux). Reprend les budgets validés
# en notebook (notebooks/baseline_model.ipynb).
N_TRIALS = 25
CV_SPLITS = 3
TUNING_HORIZONS = [1, 6, 12, 18, 24]
MAX_ESTIMATORS_TUNING = 2200
MAX_ESTIMATORS_FINAL = 4000
EARLY_STOPPING_TUNING = 80
EARLY_STOPPING_FINAL = 150
RANDOM_STATE = 42


def _global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))
    rmse = float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))
    return mae, rmse


class ModelTrainer:

    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ForecastingException(e, sys)

    def track_mlflow(self, target_column: str, best_params: dict, global_metrics: dict):
        """Suivi MLflow léger : un run par cible (params Optuna + métriques
        globales). Pas d'enregistrement de modèle au Model Registry — la
        cible n'est plus un unique estimateur sklearn mais 24 modèles + leurs
        alpha/poids saisonniers, ce que `mlflow.sklearn.log_model` ne sait pas
        représenter directement. Non bloquant : un échec ne doit jamais faire
        échouer l'entraînement."""
        import mlflow

        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "energia_forecasting"))

        run_name = f"direct-residual-{target_column}-{datetime.now():%Y%m%d_%H%M%S}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "model_family": "LightGBM-direct-residual",
                "target": target_column,
                "trigger": os.getenv("RETRAIN_TRIGGER", "manual"),
                "horizon_max": HORIZON_MAX,
            })
            params = {k: v for k, v in best_params.items() if isinstance(v, (int, float, str, bool))}
            mlflow.log_params(params)
            mlflow.log_metrics({
                "global_mae": global_metrics["model_mae"],
                "global_rmse": global_metrics["model_rmse"],
                "persistence_mae": global_metrics["persistence_mae"],
                "gain_vs_persistence_pct": global_metrics["gain_vs_persistence_pct"],
            })

    def _tune_hyperparameters(self, development: pd.DataFrame, feature_columns: list[str],
                              prefix: str, delta_col: str):
        """Optuna + TimeSeriesSplit purgé, comme validé en notebook : minimise
        le ratio moyen MAE_modèle/MAE_persistance sur des horizons
        représentatifs (évite de réentraîner 24 modèles par essai)."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "objective": "regression_l1",
                "boosting_type": "gbdt",
                "n_estimators": MAX_ESTIMATORS_TUNING,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "min_child_samples": trial.suggest_int("min_child_samples", 30, 300, step=10),
                "subsample": trial.suggest_float("subsample", 0.60, 1.00),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "verbosity": -1,
            }

            splitter = TimeSeriesSplit(n_splits=CV_SPLITS, gap=HORIZON_MAX)
            fold_scores = []

            for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(development)):
                fold_train = development.iloc[train_idx]
                fold_valid = development.iloc[valid_idx]

                horizon_ratios = []
                for horizon in TUNING_HORIZONS:
                    target_name = f"target_h{horizon}"

                    X_train_fold = add_target_features(fold_train[feature_columns], horizon, delta_col)
                    X_valid_fold = add_target_features(fold_valid[feature_columns], horizon, delta_col)

                    residual_train = fold_train[target_name] - fold_train[f"{prefix}_0"]
                    residual_valid = fold_valid[target_name] - fold_valid[f"{prefix}_0"]

                    model = LGBMRegressor(**params)
                    model.fit(
                        X_train_fold, residual_train,
                        eval_set=[(X_valid_fold, residual_valid)],
                        eval_metric="mae",
                        callbacks=[
                            early_stopping(stopping_rounds=EARLY_STOPPING_TUNING,
                                          first_metric_only=True, verbose=False),
                            log_evaluation(period=0),
                        ],
                    )

                    predicted_residual = model.predict(X_valid_fold, num_iteration=model.best_iteration_)
                    y_true = fold_valid[target_name].to_numpy()
                    persistence = fold_valid[f"{prefix}_0"].to_numpy()
                    y_pred = persistence + predicted_residual

                    model_mae = mean_absolute_error(y_true, y_pred)
                    persistence_mae = mean_absolute_error(y_true, persistence)
                    horizon_ratios.append(model_mae / max(persistence_mae, 1e-9))

                    del model

                fold_score = float(np.mean(horizon_ratios))
                fold_scores.append(fold_score)

                trial.report(float(np.mean(fold_scores)), step=fold_number)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                gc.collect()

            return float(np.mean(fold_scores))

        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner,
            study_name=f"lightgbm_direct_24h_{prefix}",
        )
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, show_progress_bar=False)

        logging.info("Optuna %s -> meilleur ratio CV=%.4f", prefix, study.best_value)

        return {
            **study.best_params,
            "objective": "regression_l1",
            "boosting_type": "gbdt",
            "n_estimators": MAX_ESTIMATORS_FINAL,
            "subsample_freq": 1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def _train_horizons(self, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                        feature_columns: list[str], prefix: str, delta_col: str, best_params: dict):
        """Boucle des 24 modèles directs (un par horizon) : early stopping
        pour fixer le nombre d'arbres, grid-search (alpha, poids saisonnier)
        sur validation, réentraînement final sur train+valid, évaluation sur
        test jamais vu pendant la sélection."""
        models, alphas, seasonal_weights, best_iterations = {}, {}, {}, {}
        results = []
        importances = []

        n_test = len(test)
        y_true_matrix = np.zeros((n_test, HORIZON_MAX))
        pred_model_matrix = np.zeros((n_test, HORIZON_MAX))
        pred_persistence_matrix = np.zeros((n_test, HORIZON_MAX))
        pred_seasonal_matrix = np.zeros((n_test, HORIZON_MAX))

        for horizon in range(1, HORIZON_MAX + 1):
            target_name = f"target_h{horizon}"

            X_train = add_target_features(train[feature_columns], horizon, delta_col)
            X_valid = add_target_features(valid[feature_columns], horizon, delta_col)
            X_test = add_target_features(test[feature_columns], horizon, delta_col)

            residual_train = train[target_name] - train[f"{prefix}_0"]
            residual_valid = valid[target_name] - valid[f"{prefix}_0"]

            temp_model = LGBMRegressor(**best_params)
            temp_model.fit(
                X_train, residual_train,
                eval_set=[(X_valid, residual_valid)],
                eval_metric="mae",
                callbacks=[
                    early_stopping(stopping_rounds=EARLY_STOPPING_FINAL,
                                  first_metric_only=True, verbose=False),
                    log_evaluation(period=0),
                ],
            )

            best_iteration = temp_model.best_iteration_
            if not best_iteration or best_iteration <= 0:
                best_iteration = MAX_ESTIMATORS_FINAL

            predicted_residual_valid = temp_model.predict(X_valid, num_iteration=best_iteration)
            persistence_valid = valid[f"{prefix}_0"].to_numpy()
            seasonal_valid = seasonal_baseline(valid, horizon, prefix)
            y_valid = valid[target_name].to_numpy()

            best_validation_mae = np.inf
            best_alpha = 1.0
            best_seasonal_weight = 0.0
            for alpha in ALPHA_GRID:
                direct_prediction = persistence_valid + alpha * predicted_residual_valid
                for seasonal_weight in SEASONAL_WEIGHT_GRID:
                    prediction_valid = (
                        (1 - seasonal_weight) * direct_prediction + seasonal_weight * seasonal_valid
                    )
                    score = mean_absolute_error(y_valid, prediction_valid)
                    if score < best_validation_mae:
                        best_validation_mae = score
                        best_alpha = float(alpha)
                        best_seasonal_weight = float(seasonal_weight)

            X_final = pd.concat([X_train, X_valid], axis=0)
            residual_final = pd.concat([residual_train, residual_valid], axis=0)

            final_params = dict(best_params)
            final_params["n_estimators"] = int(best_iteration)
            final_model = LGBMRegressor(**final_params)
            final_model.fit(X_final, residual_final)

            predicted_residual_test = final_model.predict(X_test)
            persistence_test = test[f"{prefix}_0"].to_numpy()
            seasonal_test = seasonal_baseline(test, horizon, prefix)
            y_test = test[target_name].to_numpy()

            direct_prediction_test = persistence_test + best_alpha * predicted_residual_test
            prediction_test = (
                (1 - best_seasonal_weight) * direct_prediction_test + best_seasonal_weight * seasonal_test
            )
            prediction_test = np.maximum(prediction_test, 0)

            mae_model = float(mean_absolute_error(y_test, prediction_test))
            rmse_model = float(np.sqrt(mean_squared_error(y_test, prediction_test)))
            mae_persistence = float(mean_absolute_error(y_test, persistence_test))
            rmse_persistence = float(np.sqrt(mean_squared_error(y_test, persistence_test)))
            mae_seasonal = float(mean_absolute_error(y_test, seasonal_test))
            rmse_seasonal = float(np.sqrt(mean_squared_error(y_test, seasonal_test)))
            gain = 100 * (mae_persistence - mae_model) / mae_persistence

            position = horizon - 1
            y_true_matrix[:, position] = y_test
            pred_model_matrix[:, position] = prediction_test
            pred_persistence_matrix[:, position] = persistence_test
            pred_seasonal_matrix[:, position] = seasonal_test

            models[horizon] = final_model
            alphas[horizon] = best_alpha
            seasonal_weights[horizon] = best_seasonal_weight
            best_iterations[horizon] = int(best_iteration)

            results.append({
                "horizon": horizon,
                "mae_validation": float(best_validation_mae),
                "mae_model": mae_model,
                "rmse_model": rmse_model,
                "mae_persistence": mae_persistence,
                "rmse_persistence": rmse_persistence,
                "mae_seasonal": mae_seasonal,
                "rmse_seasonal": rmse_seasonal,
                "gain_vs_persistence_pct": float(gain),
                "alpha": best_alpha,
                "seasonal_weight": best_seasonal_weight,
                "n_trees": int(best_iteration),
            })

            importances.append(pd.DataFrame({
                "variable": X_train.columns,
                "importance": final_model.feature_importances_,
                "horizon": horizon,
            }))

            logging.info(
                "%s horizon %02d/%d -> MAE=%.2f gain=%+.2f%% alpha=%.2f saisonnier=%.2f arbres=%s",
                prefix, horizon, HORIZON_MAX, mae_model, gain, best_alpha, best_seasonal_weight, best_iteration,
            )

            del temp_model
            gc.collect()

        model_mae, model_rmse = _global_metrics(y_true_matrix, pred_model_matrix)
        persistence_mae, persistence_rmse = _global_metrics(y_true_matrix, pred_persistence_matrix)
        seasonal_mae, seasonal_rmse = _global_metrics(y_true_matrix, pred_seasonal_matrix)
        global_gain = 100 * (persistence_mae - model_mae) / persistence_mae

        importance_mean = (
            pd.concat(importances, ignore_index=True)
            .groupby("variable", as_index=False)["importance"].mean()
            .sort_values("importance", ascending=False)
            .head(25)
        )

        composite = {
            "models": models,
            "alphas": alphas,
            "seasonal_weights": seasonal_weights,
            "best_iterations": best_iterations,
            "feature_columns": feature_columns,
            "prefix": prefix,
        }
        metadata = {
            "hyperparameters": {k: v for k, v in best_params.items()
                               if isinstance(v, (str, int, float, bool, type(None)))},
            "results_by_horizon": results,
            "global": {
                "model_mae": model_mae, "model_rmse": model_rmse,
                "persistence_mae": persistence_mae, "persistence_rmse": persistence_rmse,
                "seasonal_mae": seasonal_mae, "seasonal_rmse": seasonal_rmse,
                "gain_vs_persistence_pct": global_gain,
            },
            "top_importance": importance_mean.to_dict(orient="records"),
        }
        return composite, metadata

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            bundle = load_object(self.data_transformation_artifact.transformed_object_file_path)

            composite_model = {}
            metadata_per_target = {}

            for target_column in FEATURE_NAMES:
                entry = bundle[target_column]
                prefix = entry["prefix"]
                delta_col = f"{prefix}_delta_1"
                feature_columns = entry["feature_columns"]
                train, valid, test = entry["train"], entry["valid"], entry["test"]

                development = pd.concat([train, valid], axis=0)
                best_params = self._tune_hyperparameters(development, feature_columns, prefix, delta_col)

                composite, target_metadata = self._train_horizons(
                    train, valid, test, feature_columns, prefix, delta_col, best_params
                )
                composite_model[target_column] = composite
                metadata_per_target[target_column] = target_metadata

                try:
                    self.track_mlflow(target_column, best_params, target_metadata["global"])
                except Exception as mlflow_error:
                    logging.warning("Suivi MLflow ignoré (%s): %s", target_column, mlflow_error)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=composite_model)

            output_dir = os.getenv("MODEL_OUTPUT_DIR", "candidate_models")
            model_path = os.path.join(output_dir, "model.pkl")
            save_object(file_path=model_path, obj=composite_model)

            with open(model_path, "rb") as model_file:
                model_sha256 = hashlib.sha256(model_file.read()).hexdigest()
            dataset_hashes = {}
            for dataset_path in sorted(training_pipeline.PATH_FILE_DATASET.glob("*.csv")):
                with open(dataset_path, "rb") as dataset_file:
                    dataset_hashes[dataset_path.name] = hashlib.sha256(dataset_file.read()).hexdigest()
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
                "architecture": "direct_multihorizon_residual",
                "horizon_max": HORIZON_MAX,
                "targets": FEATURE_NAMES,
                "exogenous": ["temp"],
                "per_target": metadata_per_target,
                "test_protocol": "holdout chronologique purgé (embargo 24h), jamais utilisé pour la sélection",
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": package_version("numpy"),
                    "pandas": package_version("pandas"),
                    "lightgbm": package_version("lightgbm"),
                    "optuna": package_version("optuna"),
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
                train_metric_artifact=None,
                valid_metric_artifact=None,
                test_metric_artifact=None,
                metadata_file_path=artifact_metadata_path,
            )

            logging.info("Model trainer artifact: %s", model_trainer_artifact)

            return model_trainer_artifact

        except Exception as e:
            raise ForecastingException(e, sys)
