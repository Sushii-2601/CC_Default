"""Model training, calibration, and deployment-artifact assembly.

Pipeline for each feature set (research / deployment):

1. Fit an :class:`xgboost.XGBClassifier` on the training split.
2. Calibrate its raw scores with isotonic regression fit on a *held-out*
   validation split (never the training or test split), so probabilities
   are honest.
3. Evaluate PR-AUC / ROC-AUC / Brier score on the untouched test split.
4. Search the validation split for the profit-optimal decision threshold
   (see :mod:`lending_club.metrics`), then report realized profit on test.

Only the *deployment* model, its calibrator, its feature contract, its
SHAP explainer, and its chosen threshold are serialized to
``deployment_artifacts.joblib`` -- exactly what ``app.py`` needs to score a
new application. The research model is kept only to compute the PR-AUC
retention figure reported in the model card.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, TEST_SIZE, VAL_SIZE
from .features import align_columns

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


@dataclass
class CalibratedModel:
    """A fitted XGBoost classifier plus an isotonic probability calibrator,
    bound to a fixed feature-column contract. Picklable end-to-end so it can
    be saved with joblib and loaded by the Streamlit app without needing any
    training-time objects (raw dataframes, splits, etc.)."""

    base_model: xgb.XGBClassifier
    isotonic: IsotonicRegression
    feature_columns: list = field(default_factory=list)

    def predict_proba_default(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(default) for each row of a raw feature frame
        (columns are aligned to the training-time contract automatically)."""
        X_aligned = align_columns(X, self.feature_columns)
        raw = self.base_model.predict_proba(X_aligned)[:, 1]
        return np.clip(self.isotonic.predict(raw), 0.0, 1.0)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    amnt_train: np.ndarray
    amnt_val: np.ndarray
    amnt_test: np.ndarray


def split_data(X: pd.DataFrame, y: np.ndarray, loan_amnt: np.ndarray, seed: int = RANDOM_SEED) -> SplitData:
    X_trainval, X_test, y_trainval, y_test, amnt_trainval, amnt_test = train_test_split(
        X, y, loan_amnt, test_size=TEST_SIZE, random_state=seed, stratify=y,
    )
    val_fraction_of_trainval = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val, amnt_train, amnt_val = train_test_split(
        X_trainval, y_trainval, amnt_trainval,
        test_size=val_fraction_of_trainval, random_state=seed, stratify=y_trainval,
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, amnt_train, amnt_val, amnt_test)


def train_calibrated_model(splits: SplitData, xgb_params: dict | None = None) -> CalibratedModel:
    params = dict(XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    n_pos = int(splits.y_train.sum())
    n_neg = int(len(splits.y_train) - n_pos)
    params["scale_pos_weight"] = max(n_neg / max(n_pos, 1), 1.0)

    model = xgb.XGBClassifier(**params)
    model.fit(
        splits.X_train, splits.y_train,
        eval_set=[(splits.X_val, splits.y_val)],
        verbose=False,
    )

    raw_val_pred = model.predict_proba(splits.X_val)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_val_pred, splits.y_val)

    return CalibratedModel(base_model=model, isotonic=isotonic, feature_columns=list(splits.X_train.columns))
