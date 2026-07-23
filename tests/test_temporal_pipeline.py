from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline_prevision.components.data_ingestion import DataIngestion
from pipeline_prevision.components.data_transformation import DataTransformation
from pipeline_prevision.components.data_validation import DataValidation
from pipeline_prevision.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from pipeline_prevision.utils.main_utils.utils import load_object, window_generator


FEATURES = [
    "temp",
    "SOLAR",
    "BIOMASS",
    "WIND_ONSHORE",
    "NUCLEAR",
    "consommation_totale",
]


def frame(start: str, rows: int, offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq="h", name="timestamp")
    base = np.arange(rows, dtype=float) + offset
    return pd.DataFrame(
        {
            "temp": 10 + base * 0.01,
            "SOLAR": np.maximum(0, base),
            "BIOMASS": 300 + base * 0.01,
            "WIND_ONSHORE": 2_000 + base,
            "NUCLEAR": 40_000 + base,
            "consommation_totale": 50_000 + base,
        },
        index=index,
    )


def test_temporal_split_creates_three_distinct_ordered_files(tmp_path):
    config = SimpleNamespace(
        train_test_split_ratio=0.2,
        train_valid_split_ratio=0.1,
        training_file_path=str(tmp_path / "train.csv"),
        submission_file_path=str(tmp_path / "valid.csv"),
        testing_file_path=str(tmp_path / "test.csv"),
    )
    ingestion = DataIngestion(config)
    train, valid, test = ingestion.split_data_as_train_test_valid(
        frame("2024-01-01", 100)
    )

    assert (len(train), len(valid), len(test)) == (72, 8, 20)
    assert train.index.max() < valid.index.min() < test.index.min()
    assert Path(config.submission_file_path).read_bytes() != Path(
        config.testing_file_path
    ).read_bytes()


def test_temporal_split_rejects_duplicate_timestamps(tmp_path):
    data = frame("2024-01-01", 20)
    data.index = data.index[:-1].append(pd.DatetimeIndex([data.index[-2]]))
    config = SimpleNamespace(
        train_test_split_ratio=0.2,
        train_valid_split_ratio=0.1,
        training_file_path=str(tmp_path / "train.csv"),
        submission_file_path=str(tmp_path / "valid.csv"),
        testing_file_path=str(tmp_path / "test.csv"),
    )
    with pytest.raises(Exception, match="dupliqué"):
        DataIngestion(config).split_data_as_train_test_valid(data)


def validation_config(tmp_path):
    valid_dir = tmp_path / "validated"
    invalid_dir = tmp_path / "invalid"
    return SimpleNamespace(
        valid_train_file_path=str(valid_dir / "train.csv"),
        valid_submission_file_path=str(valid_dir / "valid.csv"),
        valid_test_file_path=str(valid_dir / "test.csv"),
        invalid_train_file_path=str(invalid_dir / "train.csv"),
        invalid_submission_file_path=str(invalid_dir / "valid.csv"),
        invalid_test_file_path=str(invalid_dir / "test.csv"),
        drift_report_file_path=str(tmp_path / "reports" / "quality.yaml"),
    )


def ingestion_artifact(tmp_path):
    train_path = tmp_path / "source_train.csv"
    valid_path = tmp_path / "source_valid.csv"
    test_path = tmp_path / "source_test.csv"
    frame("2024-01-01", 60).to_csv(train_path)
    frame("2024-01-03 12:00", 20, 60).to_csv(valid_path)
    frame("2024-01-04 08:00", 20, 80).to_csv(test_path)
    return DataIngestionArtifact(
        trained_file_path=str(train_path),
        submission_file_path=str(valid_path),
        test_file_path=str(test_path),
    )


def test_validation_enforces_schema_chronology_and_distinct_partitions(tmp_path):
    result = DataValidation(
        ingestion_artifact(tmp_path), validation_config(tmp_path)
    ).initiate_data_validation()

    assert result.validation_status is True
    assert Path(result.valid_train_file_path).exists()
    report = yaml.safe_load(Path(result.drift_report_file_path).read_text("utf-8"))
    assert report["validation_status"] is True
    assert not report["chronology_errors"]
    assert len(set(report["fingerprints"].values())) == 3


def test_validation_rejects_missing_column(tmp_path):
    artifact = ingestion_artifact(tmp_path)
    broken = pd.read_csv(artifact.submission_file_path).drop(columns=["NUCLEAR"])
    broken.to_csv(artifact.submission_file_path, index=False)

    with pytest.raises(Exception, match="colonnes manquantes"):
        DataValidation(
            artifact, validation_config(tmp_path)
        ).initiate_data_validation()


def test_window_generator_includes_last_eligible_target_without_future_leakage():
    values = np.arange(8, dtype=float).reshape(-1, 1)
    x_values, y_values = window_generator(values, lookback=3, horizon=2)

    assert x_values.shape == (4, 3, 1)
    assert y_values.shape == (4, 2, 1)
    np.testing.assert_array_equal(x_values[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(y_values[0, :, 0], [3, 4])
    np.testing.assert_array_equal(y_values[-1, :, 0], [6, 7])


def test_preprocessor_is_fitted_only_on_training_partition(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"
    train = frame("2024-01-01", 60)
    valid = frame("2024-01-03 12:00", 20, 10_000)
    test = frame("2024-01-04 08:00", 20, 20_000)
    train.to_csv(train_path)
    valid.to_csv(valid_path)
    test.to_csv(test_path)

    validation = DataValidationArtifact(
        validation_status=True,
        valid_train_file_path=str(train_path),
        valid_submission_file_path=str(valid_path),
        valid_test_file_path=str(test_path),
        invalid_train_file_path="",
        invalid_submission_file_path="",
        invalid_test_file_path="",
        drift_report_file_path="",
    )
    config = SimpleNamespace(
        transformed_object_file_path=str(tmp_path / "transform" / "preprocessor.pkl"),
        transformed_train_file_path=str(tmp_path / "transform" / "train.npy"),
        transformed_submission_file_path=str(tmp_path / "transform" / "valid.npy"),
        transformed_test_file_path=str(tmp_path / "transform" / "test.npy"),
    )
    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(tmp_path / "models"))
    artifact = DataTransformation(validation, config).initiate_data_transformation()
    preprocessor = load_object(artifact.transformed_object_file_path)
    scaler = preprocessor.named_steps["scaler"]

    np.testing.assert_allclose(scaler.data_max_, train[FEATURES].max().to_numpy())
    valid_array = np.load(artifact.transformed_submission_file_path)
    test_array = np.load(artifact.transformed_test_file_path)
    assert len(valid_array) == len(valid)
    assert len(test_array) == len(test)
