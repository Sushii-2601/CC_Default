"""Tests for real-data feature engineering. Skipped when the real CSV isn't
downloaded -- see tests/test_real_data.py."""
import numpy as np
import pytest

from lending_club.config import REAL_RAW_CSV_PATH
from lending_club.real_data import load_real_loans
from lending_club.real_features import (
    align_columns,
    build_real_deployment_features,
    build_real_research_features,
)

pytestmark = pytest.mark.skipif(
    not REAL_RAW_CSV_PATH.exists(), reason="real LendingClub CSV not downloaded (see README)"
)


def test_deployment_features_numeric_no_leakage_columns():
    df = load_real_loans(nrows=20_000)
    X = build_real_deployment_features(df)
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    for leaky in ["grade", "sub_grade", "int_rate", "installment"]:
        assert leaky not in X.columns


def test_research_features_include_int_rate_and_grade_but_not_leakage():
    df = load_real_loans(nrows=20_000)
    X = build_real_research_features(df)
    assert "int_rate" in X.columns
    assert any(c.startswith("grade_") for c in X.columns)
    assert X.shape[1] > build_real_deployment_features(df).shape[1]


def test_single_row_matches_batch_contract():
    df = load_real_loans(nrows=20_000)
    batch_cols = build_real_deployment_features(df).columns.tolist()
    single = build_real_deployment_features(df.iloc[[0]])
    aligned = align_columns(single, batch_cols)
    assert list(aligned.columns) == batch_cols
    assert len(aligned) == 1
