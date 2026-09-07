#!/usr/bin/env python
"""End-to-end training entrypoint.

Loads (or generates) the dataset, builds both the research and deployment
feature sets, trains + calibrates an XGBoost model on each, picks a
profit-optimal decision threshold for the deployment model, computes SHAP
explanations, and writes:

  * ``artifacts/deployment_artifacts.joblib`` -- everything ``app.py`` needs.
  * ``reports/metrics.json``                  -- machine-readable metrics.
  * ``reports/model_report.md``                -- human-readable model card.

Usage:
    python scripts/train.py [--n-rows 60000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_club.config import (
    ARTIFACTS_DIR,
    DEPLOYMENT_ARTIFACTS_PATH,
    LOSS_GIVEN_DEFAULT,
    METRICS_REPORT_PATH,
    MODEL_CARD_PATH,
    N_SYNTHETIC_ROWS,
    PROFIT_MARGIN,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from lending_club.data import (
    APPLICATION_TYPE,
    GRADES,
    HOME_OWNERSHIP,
    INITIAL_LIST_STATUS,
    PURPOSE,
    REGIONS,
    VERIFICATION_STATUS,
    generate_synthetic_loans,
)
from lending_club.explain import build_explainer
from lending_club.features import build_deployment_features, build_research_features
from lending_club.metrics import (
    classification_metrics,
    find_profit_optimal_threshold,
    realized_profit,
)
from lending_club.modeling import split_data, train_calibrated_model


def main(n_rows: int, seed: int) -> None:
    t0 = time.time()
    print(f"[1/6] Generating synthetic dataset ({n_rows} rows, seed={seed})...")
    df = generate_synthetic_loans(n_rows=n_rows, seed=seed)
    print(f"      default rate: {df[TARGET_COLUMN].mean():.3%}")

    print("[2/6] Building feature sets...")
    y = df[TARGET_COLUMN].to_numpy()
    loan_amnt = df["loan_amnt"].to_numpy()
    X_deploy = build_deployment_features(df)
    X_research = build_research_features(df)
    print(f"      deployment features: {X_deploy.shape[1]}  |  research features: {X_research.shape[1]}")

    print("[3/6] Splitting + training deployment model...")
    splits_deploy = split_data(X_deploy, y, loan_amnt, seed=seed)
    deploy_model = train_calibrated_model(splits_deploy)

    print("[4/6] Splitting + training research (fuller) model for comparison...")
    splits_research = split_data(X_research, y, loan_amnt, seed=seed)
    research_model = train_calibrated_model(splits_research)

    print("[5/6] Evaluating + selecting profit-optimal threshold...")
    p_test_deploy = deploy_model.predict_proba_default(splits_deploy.X_test)
    p_test_research = research_model.predict_proba_default(splits_research.X_test)

    deploy_test_metrics = classification_metrics(splits_deploy.y_test, p_test_deploy, threshold=0.5)
    research_test_metrics = classification_metrics(splits_research.y_test, p_test_research, threshold=0.5)
    pr_auc_retention = deploy_test_metrics["pr_auc"] / research_test_metrics["pr_auc"]

    p_val_deploy = deploy_model.predict_proba_default(splits_deploy.X_val)
    threshold_result = find_profit_optimal_threshold(
        splits_deploy.y_val, p_val_deploy, splits_deploy.amnt_val,
    )
    deploy_test_metrics_at_threshold = classification_metrics(
        splits_deploy.y_test, p_test_deploy, threshold=threshold_result.threshold,
    )
    test_profit = realized_profit(
        splits_deploy.y_test, p_test_deploy, splits_deploy.amnt_test, threshold_result.threshold,
    )
    baseline_approve_all_profit = realized_profit(
        splits_deploy.y_test, p_test_deploy, splits_deploy.amnt_test, threshold=1.0,
    )

    print("[6/6] Building SHAP explainer and saving artifacts...")
    explainer = build_explainer(deploy_model)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    categorical_options = {
        "home_ownership": HOME_OWNERSHIP,
        "verification_status": VERIFICATION_STATUS,
        "purpose": PURPOSE,
        "initial_list_status": INITIAL_LIST_STATUS,
        "application_type": APPLICATION_TYPE,
        "region": REGIONS,
        "grade": GRADES,
    }

    artifacts = {
        "model": deploy_model,
        "explainer": explainer,
        "feature_columns": deploy_model.feature_columns,
        "threshold": threshold_result.threshold,
        "categorical_options": categorical_options,
        "cost_assumptions": {
            "loss_given_default": LOSS_GIVEN_DEFAULT,
            "profit_margin": PROFIT_MARGIN,
        },
        "metrics": {
            "deployment_test": deploy_test_metrics_at_threshold,
            "research_test": research_test_metrics,
            "pr_auc_retention": pr_auc_retention,
            "threshold_search": {
                "threshold": threshold_result.threshold,
                "val_expected_profit": threshold_result.expected_profit,
                "val_approval_rate": threshold_result.approval_rate,
                "val_default_rate_of_approved": threshold_result.default_rate_of_approved,
            },
            "test_realized_profit_at_threshold": test_profit,
            "test_realized_profit_approve_all": baseline_approve_all_profit,
        },
        "n_deployment_features": X_deploy.shape[1],
        "n_research_features": X_research.shape[1],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": seed,
        "package_version": "1.0.0",
    }
    joblib.dump(artifacts, DEPLOYMENT_ARTIFACTS_PATH)
    print(f"      saved {DEPLOYMENT_ARTIFACTS_PATH}")

    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(artifacts["metrics"] | {
            "n_deployment_features": X_deploy.shape[1],
            "n_research_features": X_research.shape[1],
        }, f, indent=2)
    print(f"      saved {METRICS_REPORT_PATH}")

    write_model_card(artifacts, X_deploy.shape[1], X_research.shape[1])
    print(f"      saved {MODEL_CARD_PATH}")

    print(f"\nDone in {time.time() - t0:.1f}s.")
    print(f"Deployment test PR-AUC: {deploy_test_metrics['pr_auc']:.4f}")
    print(f"Research  test PR-AUC: {research_test_metrics['pr_auc']:.4f}")
    print(f"PR-AUC retention: {pr_auc_retention:.1%}")
    print(f"Profit-optimal threshold: {threshold_result.threshold:.3f}")


def write_model_card(artifacts: dict, n_deploy: int, n_research: int) -> None:
    m = artifacts["metrics"]
    lines = [
        "# Model Report -- Lending Club Credit Risk",
        "",
        f"_Generated {artifacts['trained_at']} · seed={artifacts['random_seed']}_",
        "",
        "## Feature sets",
        f"- Research (full) model: **{n_research}** features (includes LendingClub grade/sub-grade "
        "bands plus engineered ratios, logs, polynomials, and binned dummies).",
        f"- Deployment model: **{n_deploy}** features (application-time inputs only, one-hot encoded).",
        "",
        "## Test-set performance",
        "| Metric | Deployment model | Research model |",
        "|---|---|---|",
        f"| PR-AUC | {m['deployment_test']['pr_auc']:.4f} | {m['research_test']['pr_auc']:.4f} |",
        f"| ROC-AUC | {m['deployment_test']['roc_auc']:.4f} | {m['research_test']['roc_auc']:.4f} |",
        f"| Brier score | {m['deployment_test']['brier_score']:.4f} | {m['research_test']['brier_score']:.4f} |",
        "",
        f"**PR-AUC retention: {m['pr_auc_retention']:.1%}** of the research model's PR-AUC, "
        "using only application-time features.",
        "",
        "## Cost-sensitive decision threshold",
        f"- Loss given default assumption: {artifacts['cost_assumptions']['loss_given_default']:.0%} "
        "of outstanding principal.",
        f"- Profit margin assumption: {artifacts['cost_assumptions']['profit_margin']:.0%} net "
        "interest margin on a fully-paid loan.",
        f"- Profit-optimal threshold on P(default), selected on the validation split: "
        f"**{m['threshold_search']['threshold']:.3f}**",
        f"- Validation approval rate at threshold: {m['threshold_search']['val_approval_rate']:.1%}",
        f"- Validation default rate among approved loans: "
        f"{m['threshold_search']['val_default_rate_of_approved']:.1%}",
        "",
        "## Realized profit on the untouched test split",
        f"- At the profit-optimal threshold: **{m['test_realized_profit_at_threshold']:,.0f}** "
        "(sum of loan_amnt-scaled profit/loss units)",
        f"- Naive \"approve everyone\" baseline: {m['test_realized_profit_approve_all']:,.0f}",
        "",
        "## Notes",
        "This is a portfolio/demo project. The dataset is synthetically generated to mirror the "
        "column schema and realistic statistical structure of LendingClub's public accepted-loans "
        "data (the real dataset requires a Kaggle account and is several GB, so it is not vendored "
        "into this repo). Loss-given-default and profit-margin assumptions are explicitly stated "
        "above, not derived from proprietary lender data.",
    ]
    MODEL_CARD_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=N_SYNTHETIC_ROWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    main(n_rows=args.n_rows, seed=args.seed)
