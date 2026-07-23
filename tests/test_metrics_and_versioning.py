import hashlib

import numpy as np

from pipeline_prevision.utils.ml_utils.metric.forecasting_metric import (
    get_forecast_score,
)
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
    get_model_version,
)
from streamlit_app.data import _conformal_bounds


def test_forecast_metrics_match_independent_formula():
    actual = np.array([[1.0, 10.0], [3.0, 14.0]])
    predicted = np.array([[2.0, 8.0], [1.0, 14.0]])
    score = get_forecast_score(actual, predicted)

    assert score.mae == 1.25
    assert score.mse == 2.25


def test_prequential_bounds_never_use_current_residual():
    predicted = np.arange(20, dtype=float)
    actual = predicted + 1
    lower, upper = _conformal_bounds(
        predicted, actual, alpha=0.05, min_calibration=5
    )

    assert np.isnan(lower[:5]).all()
    np.testing.assert_array_equal(lower[5:], predicted[5:] - 1)
    np.testing.assert_array_equal(upper[5:], predicted[5:] + 1)


def test_model_version_falls_back_to_real_artifact_hash(tmp_path, monkeypatch):
    model_bytes = b"energia-model"
    (tmp_path / "model.pkl").write_bytes(model_bytes)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    expected = hashlib.sha256(model_bytes).hexdigest()[:12]
    assert get_model_version() == f"sha256:{expected}"
