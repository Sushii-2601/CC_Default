"""SHAP-based per-prediction explanations for the deployment model.

SHAP values are computed on the raw XGBoost margin (log-odds of default)
via ``shap.TreeExplainer``, which is exact and fast for tree ensembles.
Since the isotonic calibration step is monotonic, the *ranking* and *sign*
of each feature's contribution to risk carries over faithfully to the
calibrated probability even though the calibrator itself isn't linear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import align_columns
from .modeling import CalibratedModel


def build_explainer(model: CalibratedModel) -> shap.TreeExplainer:
    return shap.TreeExplainer(model.base_model)


def explain_instance(
    model: CalibratedModel, explainer: shap.TreeExplainer, X_row: pd.DataFrame, top_k: int = 12,
) -> pd.DataFrame:
    """Return the top-``k`` features driving one prediction, ranked by
    |SHAP value|, as a tidy DataFrame with columns
    [feature, value, shap_value, direction]."""
    X_aligned = align_columns(X_row, model.feature_columns)
    shap_values = explainer.shap_values(X_aligned)
    values = np.asarray(shap_values)[0]

    out = pd.DataFrame({
        "feature": model.feature_columns,
        "value": X_aligned.iloc[0].values,
        "shap_value": values,
    })
    out["direction"] = np.where(out["shap_value"] >= 0, "increases risk", "decreases risk")
    out["abs_shap"] = out["shap_value"].abs()
    out = out.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").head(top_k)
    return out.reset_index(drop=True)
