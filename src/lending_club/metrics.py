"""Evaluation metrics, expected-loss cost model, and profit-optimized
threshold search.

The lending decision is not made at the accuracy- or F1-maximizing
threshold. Instead we search over the predicted default-probability
threshold for the one that maximizes *realized expected profit* on a
held-out validation set, under an explicit, labeled cost model:

    approve a loan  -> + PROFIT_MARGIN * loan_amnt   if it fully pays
                        - LOSS_GIVEN_DEFAULT * loan_amnt  if it defaults
    decline a loan  -> 0 (no exposure, no upside)

This is a simplifying, explicitly-stated assumption (flat margin / flat
LGD), not a proprietary lender's real economics -- exactly as documented
in the project README.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import LOSS_GIVEN_DEFAULT, PROFIT_MARGIN


@dataclass
class ThresholdResult:
    threshold: float
    expected_profit: float
    approval_rate: float
    default_rate_of_approved: float


def realized_profit(
    y_true: np.ndarray, p_default: np.ndarray, loan_amnt: np.ndarray, threshold: float,
    loss_given_default: float = LOSS_GIVEN_DEFAULT, profit_margin: float = PROFIT_MARGIN,
) -> float:
    """Total realized profit if loans are approved when p_default <= threshold."""
    approve = p_default <= threshold
    paid = approve & (y_true == 0)
    defaulted = approve & (y_true == 1)
    profit = profit_margin * loan_amnt[paid].sum() - loss_given_default * loan_amnt[defaulted].sum()
    return float(profit)


def find_profit_optimal_threshold(
    y_true: np.ndarray, p_default: np.ndarray, loan_amnt: np.ndarray,
    loss_given_default: float = LOSS_GIVEN_DEFAULT, profit_margin: float = PROFIT_MARGIN,
    n_steps: int = 400,
) -> ThresholdResult:
    """Grid-search the default-probability threshold that maximizes total
    expected profit on the given (validation) set."""
    candidates = np.linspace(0.0, 1.0, n_steps + 1)
    best = None
    for t in candidates:
        profit = realized_profit(y_true, p_default, loan_amnt, t, loss_given_default, profit_margin)
        if best is None or profit > best.expected_profit:
            approve = p_default <= t
            approval_rate = float(approve.mean())
            default_rate = float(y_true[approve].mean()) if approve.any() else 0.0
            best = ThresholdResult(
                threshold=float(t), expected_profit=profit,
                approval_rate=approval_rate, default_rate_of_approved=default_rate,
            )
    assert best is not None
    return best


def classification_metrics(y_true: np.ndarray, p_default: np.ndarray, threshold: float) -> dict:
    y_pred = (p_default > threshold).astype(int)  # predicted "would default" flag
    pr_auc = average_precision_score(y_true, p_default)
    roc_auc = roc_auc_score(y_true, p_default)
    brier = brier_score_loss(y_true, p_default)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
    }
