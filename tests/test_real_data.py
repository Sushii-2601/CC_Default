"""Tests for the real-data loader. Skipped automatically when the 1.6 GB
real LendingClub CSV isn't present (e.g. on CI, which only exercises the
synthetic-data path) -- see README for the download command."""
import pytest

from lending_club.config import REAL_RAW_CSV_PATH
from lending_club.real_data import ALL_DROPPED_COLUMNS, LEAKAGE_COLUMNS, load_real_loans

pytestmark = pytest.mark.skipif(
    not REAL_RAW_CSV_PATH.exists(), reason="real LendingClub CSV not downloaded (see README)"
)


def test_leakage_column_list_has_no_duplicates_and_is_nonempty():
    assert len(LEAKAGE_COLUMNS) == len(set(LEAKAGE_COLUMNS))
    assert len(LEAKAGE_COLUMNS) > 0
    assert set(LEAKAGE_COLUMNS) <= set(ALL_DROPPED_COLUMNS)


def test_load_real_loans_only_resolved_statuses():
    df = load_real_loans(nrows=20_000)
    assert set(df["loan_status"].unique()) <= {0, 1}
    assert len(df) > 0


def test_load_real_loans_drops_all_leakage_columns():
    df = load_real_loans(nrows=20_000)
    assert not any(col in df.columns for col in LEAKAGE_COLUMNS)


def test_load_real_loans_derives_expected_columns():
    df = load_real_loans(nrows=20_000)
    for col in ["credit_history_years", "emp_length_years", "region", "fico_avg"]:
        assert col in df.columns
    assert (df["credit_history_years"] >= 0).all()
    assert set(df["region"].unique()) - {"Other"} != set()


def test_load_real_loans_term_is_numeric():
    df = load_real_loans(nrows=20_000)
    assert set(df["term"].unique()) <= {36, 60}
