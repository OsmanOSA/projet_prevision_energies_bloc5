import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pipeline_prevision.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from pipeline_prevision.entity.config_entity import DataValidationConfig
from pipeline_prevision.exception.exception import ForecastingException
from pipeline_prevision.logging.logger import logging
from pipeline_prevision.constant.training_pipeline import SCHEMA_FILE_PATH
from pipeline_prevision.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    """Valide le schéma, la chronologie et la qualité des trois partitions."""

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as exc:
            raise ForecastingException(exc, sys) from exc

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                parse_dates=["timestamp"],
                index_col="timestamp",
            )
        except Exception as exc:
            raise ForecastingException(exc, sys) from exc

    @property
    def expected_columns(self) -> dict[str, str]:
        return {
            name: dtype
            for item in self._schema_config["columns"]
            for name, dtype in item.items()
        }

    def validate_dataframe(self, dataframe: pd.DataFrame, split_name: str) -> dict:
        """Retourne un rapport sérialisable ; aucune donnée n'est corrigée en silence."""
        errors: list[str] = []
        warnings: list[str] = []
        expected = self.expected_columns

        missing_columns = sorted(set(expected) - set(dataframe.columns))
        unexpected_columns = sorted(set(dataframe.columns) - set(expected))
        if missing_columns:
            errors.append(f"colonnes manquantes: {missing_columns}")
        if unexpected_columns:
            errors.append(f"colonnes inattendues: {unexpected_columns}")

        if not isinstance(dataframe.index, pd.DatetimeIndex):
            errors.append("index non temporel")
        else:
            if dataframe.index.hasnans:
                errors.append("horodatages invalides (NaT)")
            duplicate_count = int(dataframe.index.duplicated().sum())
            if duplicate_count:
                errors.append(f"{duplicate_count} horodatage(s) dupliqué(s)")
            if not dataframe.index.is_monotonic_increasing:
                errors.append("horodatages non triés")
            deltas = dataframe.index.to_series().diff().dropna()
            non_hourly = int((deltas != pd.Timedelta(hours=1)).sum())
            if non_hourly:
                errors.append(f"{non_hourly} rupture(s) de fréquence horaire")

        common = [column for column in expected if column in dataframe.columns]
        non_numeric = [
            column
            for column in common
            if not pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        if non_numeric:
            errors.append(f"colonnes non numériques: {non_numeric}")

        missing_values = {
            column: int(dataframe[column].isna().sum()) for column in common
        }
        if any(missing_values.values()):
            errors.append(f"valeurs manquantes: {missing_values}")

        infinite_values = {}
        for column in common:
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                infinite_values[column] = int(
                    np.isinf(dataframe[column].to_numpy(dtype=float)).sum()
                )
        if any(infinite_values.values()):
            errors.append(f"valeurs infinies: {infinite_values}")

        for column in [name for name in common if name != "temp"]:
            count = int((dataframe[column] < 0).sum())
            if count:
                errors.append(f"{column}: {count} valeur(s) négative(s)")

        outliers = {}
        for column, bounds in self._schema_config.get("plausible_bounds", {}).items():
            if column not in dataframe.columns:
                continue
            low, high = bounds.get("min"), bounds.get("max")
            mask = pd.Series(False, index=dataframe.index)
            if low is not None:
                mask |= dataframe[column] < low
            if high is not None:
                mask |= dataframe[column] > high
            count = int(mask.sum())
            if count:
                outliers[column] = count
                warnings.append(
                    f"{column}: {count} valeur(s) hors bornes plausibles "
                    f"[{low}, {high}]"
                )

        return {
            "split": split_name,
            "rows": int(len(dataframe)),
            "start": dataframe.index.min().isoformat() if len(dataframe) else None,
            "end": dataframe.index.max().isoformat() if len(dataframe) else None,
            "dtypes": {column: str(dataframe[column].dtype) for column in common},
            "errors": errors,
            "warnings": warnings,
            "missing_values": missing_values,
            "infinite_values": infinite_values,
            "plausibility_outliers": outliers,
            "valid": not errors,
        }

    @staticmethod
    def detect_dataset_drift(
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> dict:
        """KS par variable ; la dérive est signalée, pas utilisée comme veto aveugle."""
        report = {}
        for column in base_df.columns.intersection(current_df.columns):
            result = ks_2samp(base_df[column].dropna(), current_df[column].dropna())
            report[column] = {
                "pvalue": float(result.pvalue),
                "drift_status": bool(result.pvalue < threshold),
            }
        return report

    @staticmethod
    def _fingerprint(dataframe: pd.DataFrame) -> str:
        hashed = pd.util.hash_pandas_object(dataframe, index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train = self.read_data(self.data_ingestion_artifact.trained_file_path)
            validation = self.read_data(
                self.data_ingestion_artifact.submission_file_path
            )
            test = self.read_data(self.data_ingestion_artifact.test_file_path)

            frames = {"train": train, "validation": validation, "test": test}
            quality = {
                name: self.validate_dataframe(frame, name)
                for name, frame in frames.items()
            }

            chronology_errors = []
            if len(train) and len(validation) and train.index.max() >= validation.index.min():
                chronology_errors.append("train et validation se chevauchent")
            if len(validation) and len(test) and validation.index.max() >= test.index.min():
                chronology_errors.append("validation et test se chevauchent")

            fingerprints = {
                name: self._fingerprint(frame) for name, frame in frames.items()
            }
            partition_errors = []
            if fingerprints["validation"] == fingerprints["test"]:
                partition_errors.append("validation et test sont identiques")

            validation_errors = [
                f"{name}: {error}"
                for name, item in quality.items()
                for error in item["errors"]
            ]
            validation_errors.extend(chronology_errors)
            validation_errors.extend(partition_errors)

            report = {
                "validation_status": not validation_errors,
                "quality": quality,
                "chronology_errors": chronology_errors,
                "partition_errors": partition_errors,
                "fingerprints": fingerprints,
                "drift": {
                    "train_to_validation": self.detect_dataset_drift(
                        train, validation
                    ),
                    "validation_to_test": self.detect_dataset_drift(
                        validation, test
                    ),
                },
            }
            report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            write_yaml_file(file_path=report_path, content=report)

            if validation_errors:
                os.makedirs(
                    os.path.dirname(
                        self.data_validation_config.invalid_train_file_path
                    ),
                    exist_ok=True,
                )
                train.to_csv(self.data_validation_config.invalid_train_file_path)
                validation.to_csv(
                    self.data_validation_config.invalid_submission_file_path
                )
                test.to_csv(self.data_validation_config.invalid_test_file_path)
                raise ValueError(
                    "Validation des données échouée: "
                    + "; ".join(validation_errors)
                )

            os.makedirs(
                os.path.dirname(self.data_validation_config.valid_train_file_path),
                exist_ok=True,
            )
            train.to_csv(self.data_validation_config.valid_train_file_path)
            validation.to_csv(
                self.data_validation_config.valid_submission_file_path
            )
            test.to_csv(self.data_validation_config.valid_test_file_path)

            logging.info(
                "Validation OK: %s lignes train, %s validation, %s test",
                len(train),
                len(validation),
                len(test),
            )
            return DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_submission_file_path=self.data_validation_config.valid_submission_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_submission_file_path=self.data_validation_config.invalid_submission_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=report_path,
            )
        except Exception as exc:
            raise ForecastingException(exc, sys) from exc
