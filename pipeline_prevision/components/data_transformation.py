import sys

import numpy as np
import pandas as pd

from pipeline_prevision.utils.main_utils.utils import save_object
from pipeline_prevision.utils.main_utils.feature_engineering import (
    HORIZON_MAX, build_features_for_target, build_series_by_target, TARGET_PREFIXES,
    select_temperature, select_forecast_temperature,
)
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from pipeline_prevision.entity.config_entity import DataTransformationConfig

# Les 4 sources de production (plutôt que l'agrégat production_total) +
# consommation -- cf. feature_engineering.TARGET_PREFIXES.
TARGET_COLUMNS = list(TARGET_PREFIXES)

# Fenêtre glissée directe : 70 % train / 15 % validation / 15 % test, avec un
# embargo de HORIZON_MAX lignes avant chaque coupure (les cibles t+1..t+24
# d'une ligne juste avant la coupure débordent sinon dans la partition
# suivante -> fuite). Indépendant des ratios de DataIngestionConfig, qui ne
# servent plus qu'à produire des CSV validés en amont.
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15


class DataTransformation:

    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
                 ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise ForecastingException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, sep=None, engine="python", parse_dates=["timestamp"], index_col="timestamp")
        except Exception as e:
            raise ForecastingException(e, sys)

    @staticmethod
    def _purged_split(features_df: pd.DataFrame):
        """Découpage chronologique 70/15/15 avec embargo HORIZON_MAX."""
        n = len(features_df)
        train_cut = int(n * TRAIN_RATIO)
        valid_cut = int(n * (TRAIN_RATIO + VALID_RATIO))

        train = features_df.iloc[:train_cut - HORIZON_MAX].copy()
        valid = features_df.iloc[train_cut:valid_cut - HORIZON_MAX].copy()
        test = features_df.iloc[valid_cut:].copy()
        return train, valid, test

    def initiate_data_transformation(self) -> DataTransformationArtifact:

        logging.info("Entered initiate_data_transformation method of DataTransformation class")

        try:
            # Les 3 CSV validés sont des tranches chronologiques contiguës
            # (cf. DataIngestion.split_data_as_train_test_valid) : les
            # reconcaténer reconstruit exactement la série d'origine, sur
            # laquelle on calcule les features (lags/rolling jusqu'à 336h) et
            # notre propre split purgé, indépendant du split brut en amont.
            parts = [
                DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path),
                DataTransformation.read_data(self.data_validation_artifact.valid_submission_file_path),
                DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path),
            ]
            full = pd.concat(parts).sort_index()
            full = full[~full.index.duplicated(keep="last")].asfreq("h")

            series_by_target = build_series_by_target(full)
            temp = select_temperature(full)
            # Température PRÉVUE à l'heure cible. None si `datasets/data.csv` est
            # antérieur au backfill : les features cible sont alors absentes et
            # l'entraînement reproduit le comportement d'avant, sans erreur. Le
            # CSV doit donc porter la colonne, sinon l'entraînement (qui lit le
            # CSV) diverge de l'inférence (qui lit la base) — cf.
            # `python -m scripts.backfill_prevision_temperature --csv`.
            temp_prev = select_forecast_temperature(full)
            logging.info("Température prévue à l'heure cible : %s",
                         "disponible" if temp_prev is not None else "ABSENTE (features cible désactivées)")

            bundle = {}
            for target_column in TARGET_COLUMNS:
                features_df, prefix, target_columns = build_features_for_target(
                    series_by_target, temp, target_column, temp_prev=temp_prev
                )
                feature_columns = [c for c in features_df.columns if c not in target_columns]

                features_df = (
                    features_df.replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=feature_columns + target_columns)
                    .copy()
                )

                train, valid, test = self._purged_split(features_df)
                if min(len(train), len(valid), len(test)) == 0:
                    raise ValueError(
                        f"Historique insuffisant pour {target_column} après purge "
                        f"(train={len(train)}, valid={len(valid)}, test={len(test)})"
                    )

                bundle[target_column] = {
                    "prefix": prefix,
                    # Nom de la série exogène réellement utilisée (`temp` ou
                    # `temp_fr`) : les colonnes de features sont toutes préfixées
                    # "temp_" quelle que soit la source, donc c'est le seul
                    # endroit où l'information survit jusqu'aux métadonnées.
                    "exogenous_column": getattr(temp, "name", None),
                    "feature_columns": feature_columns,
                    "target_columns": target_columns,
                    "train": train,
                    "valid": valid,
                    "test": test,
                }

                logging.info(
                    "Transformation %s : train=%s valid=%s test=%s (%s features)",
                    target_column, len(train), len(valid), len(test), len(feature_columns),
                )

            save_object(self.data_transformation_config.transformed_object_file_path, bundle)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_submission_file_path=self.data_transformation_config.transformed_submission_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise ForecastingException(e, sys)
