#!/usr/bin/env python
"""Real-data training entrypoint: the actual LendingClub 2007-2018Q4
accepted-loans CSV (not the synthetic generator), leakage-safe research vs.
application-time deployment feature sets, calibrated XGBoost, profit-optimal
thresholding, and SHAP -- writes ``artifacts/deployment_artifacts_real.joblib``,
``reports/metrics_real.json``, ``reports/model_report_real.md``.

Requires the real CSV to already be downloaded to ``data_real/`` (see
``config.REAL_RAW_CSV_PATH`` for the exact path and the download command).

Usage:
    python scripts/train_real.py [--nrows N] [--seed 42]
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
    LOSS_GIVEN_DEFAULT,
    PROFIT_MARGIN,
    RANDOM_SEED,
    REAL_DEPLOYMENT_ARTIFACTS_PATH,
    REAL_METRICS_REPORT_PATH,
    REAL_MODEL_CARD_PATH,
    REAL_RAW_CSV_PATH,
    REPORTS_DIR,
)
from lending_club.explain import build_explainer
from lending_club.metrics import (
    classification_metrics,
    find_profit_optimal_threshold,
    realized_profit,
)
from lending_club.modeling import split_data, train_calibrated_model
from lending_club.real_data import (
    ALL_DROPPED_COLUMNS,
    LEAKAGE_COLUMNS,
    load_real_loans,
)
from lending_club.real_features import (
    KEEP_PURPOSES,
    REAL_APPLICATION_TYPE,
    REAL_HOME_OWNERSHIP,
    REAL_INITIAL_LIST_STATUS,
    REAL_REGIONS,
    REAL_VERIFICATION_STATUS,
    build_real_deployment_features,
    build_real_research_features,
)


def main(nrows: int | None, seed: int) -> None:
    t0 = time.time()
    print(f"[1/6] Loading real LendingClub CSV from {REAL_RAW_CSV_PATH} ...")
    print(f"      dropping {len(LEAKAGE_COLUMNS)} post-issuance leakage columns, "
          f"{len(ALL_DROPPED_COLUMNS) - len(LEAKAGE_COLUMNS)} junk/sparse/high-missing columns")
    df = load_real_loans(REAL_RAW_CSV_PATH, nrows=nrows)
    n_resolved = len(df)
    default_rate = df["loan_status"].mean()
    print(f"      resolved (Fully Paid / Charged Off) loans: {n_resolved:,}")
    print(f"      real charge-off rate: {default_rate:.3%}")

    print("[2/6] Building feature sets...")
    y = df["loan_status"].to_numpy()
    loan_amnt = df["loan_amnt"].to_numpy(dtype="float64")
    X_deploy = build_real_deployment_features(df)
    X_research = build_real_research_features(df)
    print(f"      deployment features: {X_deploy.shape[1]}  |  research features: {X_research.shape[1]}")
    del df  # free the raw frame before training -- memory-constrained environment

    print("[3/6] Splitting + training deployment model (this is the slow step at this row count)...")
    splits_deploy = split_data(X_deploy, y, loan_amnt, seed=seed)
    t_train0 = time.time()
    deploy_model = train_calibrated_model(splits_deploy)
    print(f"      deployment model trained in {time.time() - t_train0:.1f}s")

    print("[4/6] Splitting + training research (fuller, leakage-safe) model for comparison...")
    splits_research = split_data(X_research, y, loan_amnt, seed=seed)
    t_train1 = time.time()
    research_model = train_calibrated_model(splits_research)
    print(f"      research model trained in {time.time() - t_train1:.1f}s")

    print("[5/6] Evaluating + selecting profit-optimal threshold...")
    p_test_deploy = deploy_model.predict_proba_default(splits_deploy.X_test)
    p_test_research = research_model.predict_proba_default(splits_research.X_test)

    deploy_test_metrics = classification_metrics(splits_deploy.y_test, p_test_deploy, threshold=0.5)
    research_test_metrics = classification_metrics(splits_research.y_test, p_test_research, threshold=0.5)
    pr_auc_retention = deploy_test_metrics["pr_auc"] / research_test_metrics["pr_auc"]

    p_val_deploy = deploy_model.predict_proba_default(splits_deploy.X_val)
    threshold_result = find_profit_optimal_threshold(splits_deploy.y_val, p_val_deploy, splits_deploy.amnt_val)
    deploy_test_metrics_at_threshold = classification_metrics(
        splits_deploy.y_test, p_test_deploy, threshold=threshold_result.threshold,
    )
    test_profit = realized_profit(splits_deploy.y_test, p_test_deploy, splits_deploy.amnt_test, threshold_result.threshold)
    baseline_approve_all_profit = realized_profit(splits_deploy.y_test, p_test_deploy, splits_deploy.amnt_test, threshold=1.0)

    # F1-optimal threshold, computed purely for the "profit vs F1-optimal
    # cutoff" comparison the model card reports -- NOT what's shipped.
    import numpy as np
    from sklearn.metrics import f1_score
    f1_candidates = np.linspace(0.0, 1.0, 401)
    f1_scores = [f1_score(splits_deploy.y_val, (p_val_deploy > t).astype(int), zero_division=0) for t in f1_candidates]
    f1_optimal_threshold = float(f1_candidates[int(np.argmax(f1_scores))])
    profit_at_f1_threshold = realized_profit(splits_deploy.y_test, p_test_deploy, splits_deploy.amnt_test, f1_optimal_threshold)

    print("[6/6] Building SHAP explainer and saving artifacts...")
    explainer = build_explainer(deploy_model)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    categorical_options = {
        "home_ownership": REAL_HOME_OWNERSHIP,
        "verification_status": REAL_VERIFICATION_STATUS,
        "purpose": KEEP_PURPOSES,
        "initial_list_status": REAL_INITIAL_LIST_STATUS,
        "application_type": REAL_APPLICATION_TYPE,
        "region": REAL_REGIONS,
    }

    artifacts = {
        "model": deploy_model,
        "explainer": explainer,
        "feature_columns": deploy_model.feature_columns,
        "threshold": threshold_result.threshold,
        "categorical_options": categorical_options,
        "cost_assumptions": {"loss_given_default": LOSS_GIVEN_DEFAULT, "profit_margin": PROFIT_MARGIN},
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
            "f1_optimal_threshold": f1_optimal_threshold,
            "test_realized_profit_at_profit_threshold": test_profit,
            "test_realized_profit_at_f1_threshold": profit_at_f1_threshold,
            "test_realized_profit_approve_all": baseline_approve_all_profit,
        },
        "n_deployment_features": X_deploy.shape[1],
        "n_research_features": X_research.shape[1],
        "n_resolved_loans": n_resolved,
        "real_charge_off_rate": float(default_rate),
        "n_leakage_columns_dropped": len(LEAKAGE_COLUMNS),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": seed,
        "data_source": "real",
        "package_version": "1.0.0",
    }
    joblib.dump(artifacts, REAL_DEPLOYMENT_ARTIFACTS_PATH)
    print(f"      saved {REAL_DEPLOYMENT_ARTIFACTS_PATH}")

    with open(REAL_METRICS_REPORT_PATH, "w") as f:
        json.dump({k: v for k, v in artifacts.items() if k not in ("model", "explainer")}, f, indent=2, default=str)
    print(f"      saved {REAL_METRICS_REPORT_PATH}")

    write_model_card(artifacts)
    print(f"      saved {REAL_MODEL_CARD_PATH}")

    print(f"\nDone in {time.time() - t0:.1f}s.")
    print(f"Resolved loans: {n_resolved:,}  |  real charge-off rate: {default_rate:.3%}")
    print(f"Deployment test PR-AUC: {deploy_test_metrics['pr_auc']:.4f}  ROC-AUC: {deploy_test_metrics['roc_auc']:.4f}")
    print(f"Research  test PR-AUC: {research_test_metrics['pr_auc']:.4f}  ROC-AUC: {research_test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC retention: {pr_auc_retention:.1%}")
    print(f"Profit-optimal threshold: {threshold_result.threshold:.4f}   F1-optimal threshold: {f1_optimal_threshold:.4f}")
    print(f"Test profit @ profit-threshold: {test_profit:,.0f}   @ F1-threshold: {profit_at_f1_threshold:,.0f}   approve-all: {baseline_approve_all_profit:,.0f}")


def write_model_card(artifacts: dict) -> None:
    m = artifacts["metrics"]
    lines = [
        "# Model Report -- Lending Club Credit Risk (REAL DATA)",
        "",
        f"_Generated {artifacts['trained_at']} · seed={artifacts['random_seed']} · "
        f"source: real LendingClub 2007-2018Q4 accepted loans_",
        "",
        "## Dataset",
        f"- Resolved (Fully Paid / Charged Off) loans used: **{artifacts['n_resolved_loans']:,}**",
        f"- Real charge-off rate: **{artifacts['real_charge_off_rate']:.2%}**",
        f"- Post-issuance leakage columns identified and dropped: **{artifacts['n_leakage_columns_dropped']}** "
        "(payment/servicing history, hardship-program fields, debt-settlement fields -- see "
        "`src/lending_club/real_data.py` for the full, grouped, justified list)",
        "",
        "## Feature sets",
        f"- Research (leakage-safe, full) model: **{artifacts['n_research_features']}** features "
        "(includes grade/sub_grade/int_rate plus every remaining bureau-derived field).",
        f"- Deployment model: **{artifacts['n_deployment_features']}** features (application-time only -- "
        "excludes grade/sub_grade/int_rate/installment in addition to all post-issuance leakage).",
        "",
        "## Test-set performance",
        "| Metric | Deployment model | Research model |",
        "|---|---|---|",
        f"| PR-AUC | {m['deployment_test']['pr_auc']:.4f} | {m['research_test']['pr_auc']:.4f} |",
        f"| ROC-AUC | {m['deployment_test']['roc_auc']:.4f} | {m['research_test']['roc_auc']:.4f} |",
        f"| Brier score | {m['deployment_test']['brier_score']:.4f} | {m['research_test']['brier_score']:.4f} |",
        "",
        f"**PR-AUC retention: {m['pr_auc_retention']:.1%}** of the research model's PR-AUC, using only "
        "application-time features.",
        "",
        "## Cost-sensitive decision threshold vs. the F1-optimal cutoff",
        f"- Loss given default assumption: {artifacts['cost_assumptions']['loss_given_default']:.0%} of "
        "outstanding principal.",
        f"- Profit margin assumption: {artifacts['cost_assumptions']['profit_margin']:.0%} net interest "
        "margin on a fully-paid loan.",
        f"- **Profit-optimal threshold** (validation split): **{m['threshold_search']['threshold']:.4f}**",
        f"- F1-optimal threshold (validation split, for comparison): **{m['f1_optimal_threshold']:.4f}**",
        f"- Validation approval rate at the profit-optimal threshold: {m['threshold_search']['val_approval_rate']:.1%}",
        f"- Validation default rate among approved loans: {m['threshold_search']['val_default_rate_of_approved']:.1%}",
        "",
        "## Realized profit on the untouched test split (dollars, since loan_amnt is real USD)",
        f"- Approve everyone (baseline): **${m['test_realized_profit_approve_all']:,.0f}**",
        f"- At the F1-optimal threshold: **${m['test_realized_profit_at_f1_threshold']:,.0f}**",
        f"- At the profit-optimal threshold: **${m['test_realized_profit_at_profit_threshold']:,.0f}**",
        "",
        "## Notes",
        "This run uses the real LendingClub 2007-2018Q4 accepted-loans dataset (not the synthetic "
        "generator in `lending_club.data`). Loss-given-default and profit-margin assumptions remain "
        "explicitly stated constants (`src/lending_club/config.py`), not fitted from data.",
    ]
    REAL_MODEL_CARD_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrows", type=int, default=None, help="Limit raw CSV rows read (for smoke tests)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    main(nrows=args.nrows, seed=args.seed)
