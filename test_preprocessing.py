"""
test_preprocessing.py
-----------------------
Tests for the cleaning and preprocessing pipeline.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.data_loader import load_raw_data, detect_target_column, detect_id_columns
from src.preprocessing import clean_dataset, split_features_target, build_scaling_pipeline
from src.feature_engineering import add_engineered_features


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


def test_clean_dataset_no_nulls_or_inf(raw_df):
    cleaned = clean_dataset(raw_df)
    numeric_cols = cleaned.select_dtypes(include="number").columns
    assert cleaned.isnull().sum().sum() == 0
    assert not np.isinf(cleaned[numeric_cols]).any().any()


def test_split_features_target_excludes_id_and_target(raw_df):
    target_col = detect_target_column(raw_df)
    id_cols = detect_id_columns(raw_df)
    X, y = split_features_target(raw_df, target_col, id_cols)
    assert target_col not in X.columns
    for col in id_cols:
        assert col not in X.columns
    assert len(X) == len(y)


def test_feature_engineering_adds_log_amount(raw_df):
    target_col = detect_target_column(raw_df)
    id_cols = detect_id_columns(raw_df)
    X, y = split_features_target(raw_df, target_col, id_cols)
    X_fe = add_engineered_features(X)
    if "Amount" in X.columns:
        assert "log_amount" in X_fe.columns
        assert (X_fe["log_amount"] >= 0).all()


def test_scaling_pipeline_transforms_correctly(raw_df):
    target_col = detect_target_column(raw_df)
    id_cols = detect_id_columns(raw_df)
    X, y = split_features_target(raw_df, target_col, id_cols)
    X_fe = add_engineered_features(X)
    pipeline = build_scaling_pipeline(list(X_fe.columns))
    X_scaled = pipeline.fit_transform(X_fe)
    assert X_scaled.shape == X_fe.shape
    # scaled features should have ~zero mean and unit variance
    assert abs(X_scaled.mean()) < 1e-6
    assert abs(X_scaled.std() - 1.0) < 0.1
