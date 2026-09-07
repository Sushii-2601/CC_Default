import numpy as np

from lending_club.metrics import (
    classification_metrics,
    find_profit_optimal_threshold,
    realized_profit,
)


def test_realized_profit_all_declined_is_zero():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    amnt = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    assert realized_profit(y, p, amnt, threshold=0.0) == 0.0


def test_realized_profit_all_approved_matches_manual_calc():
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    amnt = np.array([1000.0, 1000.0])
    profit = realized_profit(y, p, amnt, threshold=1.0, loss_given_default=0.6, profit_margin=0.2)
    expected = 0.2 * 1000.0 - 0.6 * 1000.0
    assert np.isclose(profit, expected)


def test_find_profit_optimal_threshold_beats_extremes():
    rng = np.random.default_rng(0)
    n = 2000
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)  # well-calibrated synthetic probabilities
    amnt = rng.uniform(1000, 20000, n)

    result = find_profit_optimal_threshold(y, p, amnt, loss_given_default=0.6, profit_margin=0.2)
    profit_at_best = realized_profit(y, p, amnt, result.threshold, 0.6, 0.2)
    profit_approve_all = realized_profit(y, p, amnt, 1.0, 0.6, 0.2)
    profit_approve_none = realized_profit(y, p, amnt, 0.0, 0.6, 0.2)

    assert profit_at_best >= profit_approve_all
    assert profit_at_best >= profit_approve_none
    assert np.isclose(result.expected_profit, profit_at_best)


def test_classification_metrics_reasonable_ranges():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 1000)
    p = np.clip(y * 0.5 + rng.normal(0.2, 0.15, 1000), 0, 1)
    m = classification_metrics(y, p, threshold=0.5)
    assert 0 <= m["pr_auc"] <= 1
    assert 0 <= m["roc_auc"] <= 1
    assert 0 <= m["brier_score"] <= 1
    assert m["roc_auc"] > 0.5  # signal should beat random
