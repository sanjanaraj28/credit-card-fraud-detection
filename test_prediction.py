"""
test_prediction.py
---------------------
Tests for model loading and the prediction function, including the
saved artifacts produced by src/train.py.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pytest

from src.predict import load_artifacts, build_feature_row, predict_transaction

MODEL_PATH = os.path.join("models", "best_model.pkl")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not yet trained; run `python -m src.train` first.")
def test_model_and_pipeline_load():
    model, pipeline, metadata = load_artifacts()
    assert model is not None
    assert pipeline is not None
    assert "feature_names" in metadata
    assert "selected_threshold" in metadata


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not yet trained; run `python -m src.train` first.")
def test_build_feature_row_shape_and_columns():
    _, _, metadata = load_artifacts()
    raw_input = {f: 0.0 for f in metadata["feature_names"] if f != "log_amount"}
    raw_input["Amount"] = 1000.0
    row = build_feature_row(raw_input, metadata)
    assert list(row.columns) == metadata["feature_names"]
    assert row.shape[0] == 1


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not yet trained; run `python -m src.train` first.")
def test_predict_transaction_output_structure():
    model, pipeline, metadata = load_artifacts()
    raw_input = {f: 0.0 for f in metadata["feature_names"] if f != "log_amount"}
    raw_input["Amount"] = 500.0
    result = predict_transaction(raw_input, model, pipeline, metadata)

    assert result["prediction"] in ("FRAUD", "LEGITIMATE")
    assert result["prediction_int"] in (0, 1)
    assert 0.0 <= result["risk_score_pct"] <= 100.0
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert result["model_name"] == metadata["model_name"]


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not yet trained; run `python -m src.train` first.")
def test_predict_transaction_invalid_input_defaults_gracefully():
    model, pipeline, metadata = load_artifacts()
    # Missing fields should default to 0.0 rather than raising
    result = predict_transaction({}, model, pipeline, metadata)
    assert result["prediction"] in ("FRAUD", "LEGITIMATE")
