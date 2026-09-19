"""
test_data.py
-------------
Tests for dataset loading, target detection, and data-quality checks.
Run from project root: pytest tests/
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.data_loader import load_raw_data, detect_target_column, detect_id_columns, data_quality_summary


@pytest.fixture(scope="module")
def df():
    return load_raw_data()


def test_dataset_loads(df):
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_target_column_detected(df):
    target = detect_target_column(df)
    assert target in df.columns
    unique_vals = set(df[target].unique().tolist())
    assert unique_vals.issubset({0, 1})


def test_id_columns_detected(df):
    id_cols = detect_id_columns(df)
    assert isinstance(id_cols, list)
    for col in id_cols:
        assert df[col].nunique() == len(df)


def test_no_missing_values_undetected(df):
    target = detect_target_column(df)
    summary = data_quality_summary(df, target)
    assert summary["missing_values_total"] == df.isnull().sum().sum()


def test_quality_summary_keys(df):
    target = detect_target_column(df)
    summary = data_quality_summary(df, target)
    expected_keys = {
        "filename", "n_rows", "n_columns", "columns", "dtypes",
        "missing_values_total", "duplicate_rows", "numerical_columns",
        "categorical_columns", "target_column", "target_distribution",
        "id_columns", "constant_columns", "infinite_values_total"
    }
    assert expected_keys.issubset(summary.keys())
