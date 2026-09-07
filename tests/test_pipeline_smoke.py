"""End-to-end smoke test on a tiny synthetic sample: train, calibrate,
threshold-search, and explain a single prediction, all in-memory (no files
touched). Uses a small n_rows so it runs in a few seconds on CI."""
import numpy as np

from lending_club.data import generate_synthetic_loans
from lending_club.explain import build_explainer, explain_instance
from lending_club.features import align_columns, build_deployment_features
from lending_club.metrics import classification_metrics, find_profit_optimal_threshold
from lending_club.modeling import split_data, train_calibrated_model


def test_full_pipeline_smoke():
    df = generate_synthetic_loans(n_rows=2000, seed=1)
    y = df["loan_status"].to_numpy()
    amnt = df["loan_amnt"].to_numpy()
    X = build_deployment_features(df)

    splits = split_data(X, y, amnt, seed=1)
    model = train_calibrated_model(splits, xgb_params={"n_estimators": 50, "max_depth": 3})

    p_val = model.predict_proba_default(splits.X_val)
    assert p_val.shape == splits.y_val.shape
    assert np.all((p_val >= 0) & (p_val <= 1))

    threshold_result = find_profit_optimal_threshold(splits.y_val, p_val, splits.amnt_val, n_steps=50)
    assert 0.0 <= threshold_result.threshold <= 1.0

    p_test = model.predict_proba_default(splits.X_test)
    metrics = classification_metrics(splits.y_test, p_test, threshold=threshold_result.threshold)
    assert metrics["pr_auc"] > 0.0

    explainer = build_explainer(model)
    one_row = align_columns(splits.X_test.iloc[[0]], model.feature_columns)
    explanation = explain_instance(model, explainer, one_row, top_k=5)
    assert len(explanation) == 5
    assert set(explanation.columns) == {"feature", "value", "shap_value", "direction"}
