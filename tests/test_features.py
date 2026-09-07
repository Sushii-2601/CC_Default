import numpy as np

from lending_club.data import generate_synthetic_loans
from lending_club.features import (
    align_columns,
    build_deployment_features,
    build_research_features,
)


def test_deployment_features_all_numeric_no_na():
    df = generate_synthetic_loans(n_rows=300, seed=1)
    X = build_deployment_features(df)
    assert X.isna().sum().sum() == 0
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)


def test_research_features_superset_of_deployment_dims():
    df = generate_synthetic_loans(n_rows=300, seed=1)
    X_deploy = build_deployment_features(df)
    X_research = build_research_features(df)
    assert X_research.shape[1] > X_deploy.shape[1]
    assert X_research.isna().sum().sum() == 0


def test_align_columns_adds_missing_and_drops_extra():
    df = generate_synthetic_loans(n_rows=50, seed=1)
    X = build_deployment_features(df)
    target_cols = list(X.columns[:5]) + ["a_brand_new_column"]
    aligned = align_columns(X, target_cols)
    assert list(aligned.columns) == target_cols
    assert (aligned["a_brand_new_column"] == 0).all()


def test_single_row_transform_matches_batch_contract():
    df = generate_synthetic_loans(n_rows=50, seed=1)
    batch_cols = build_deployment_features(df).columns.tolist()
    single = build_deployment_features(df.iloc[[0]])
    aligned = align_columns(single, batch_cols)
    assert list(aligned.columns) == batch_cols
    assert len(aligned) == 1
