import numpy as np

from lending_club.data import REQUIRED_COLUMNS, generate_synthetic_loans


def test_generate_synthetic_loans_shape_and_columns():
    df = generate_synthetic_loans(n_rows=500, seed=1)
    assert len(df) == 500
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert "loan_status" in df.columns


def test_generate_synthetic_loans_target_is_binary_and_nontrivial():
    df = generate_synthetic_loans(n_rows=5000, seed=1)
    assert set(df["loan_status"].unique()) <= {0, 1}
    rate = df["loan_status"].mean()
    # a realistic consumer-credit default rate, and not degenerate
    assert 0.02 < rate < 0.5


def test_generate_synthetic_loans_is_deterministic():
    df1 = generate_synthetic_loans(n_rows=200, seed=7)
    df2 = generate_synthetic_loans(n_rows=200, seed=7)
    assert np.allclose(df1["loan_amnt"], df2["loan_amnt"])
    assert (df1["loan_status"] == df2["loan_status"]).all()


def test_generate_synthetic_loans_no_missing_values():
    df = generate_synthetic_loans(n_rows=300, seed=3)
    assert df.isna().sum().sum() == 0
