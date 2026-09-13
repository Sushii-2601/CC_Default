"""Streamlit deployment app for the Lending Club credit risk model.

Loads the trained artifact and scores a single loan application entered
through the sidebar form: a calibrated default probability, a lending
decision at the profit-optimal threshold, and a per-prediction SHAP
explanation.

Prefers the model trained on the real LendingClub dataset
(``artifacts/deployment_artifacts_real.joblib``, built by
``scripts/train_real.py``) when present, and falls back to the
synthetic-data model (``artifacts/deployment_artifacts.joblib``, built by
``scripts/train.py``) otherwise -- the two use different feature schemas,
so the UI form and feature-building call are dispatched on which one is
actually loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))

from lending_club.explain import explain_instance
from lending_club.features import build_deployment_features
from lending_club.real_features import build_real_deployment_features

REAL_ARTIFACTS_PATH = APP_DIR / "artifacts" / "deployment_artifacts_real.joblib"
SYNTHETIC_ARTIFACTS_PATH = APP_DIR / "artifacts" / "deployment_artifacts.joblib"


@st.cache_resource
def load_artifacts():
    """Return (artifacts, source) for whichever model is available, real
    preferred. ``source`` is ``"real"`` / ``"synthetic"`` / ``None``."""
    if REAL_ARTIFACTS_PATH.exists():
        return joblib.load(REAL_ARTIFACTS_PATH), "real"
    if SYNTHETIC_ARTIFACTS_PATH.exists():
        return joblib.load(SYNTHETIC_ARTIFACTS_PATH), "synthetic"
    return None, None


def sidebar_form_real(options: dict) -> dict:
    st.sidebar.header("Loan application")
    st.sidebar.caption("Scored by the model trained on real LendingClub 2007-2018Q4 accepted loans.")

    loan_amnt = st.sidebar.number_input("Loan amount ($)", 1000, 40000, 12000, step=500)
    term = st.sidebar.selectbox("Term (months)", [36, 60], index=0)
    annual_inc = st.sidebar.number_input("Annual income ($)", 15000, 500000, 65000, step=1000)
    emp_length_years = st.sidebar.slider("Employment length (years)", 0, 10, 5)
    home_ownership = st.sidebar.selectbox("Home ownership", options["home_ownership"])
    verification_status = st.sidebar.selectbox("Income verification", options["verification_status"])
    purpose = st.sidebar.selectbox("Loan purpose", options["purpose"])
    application_type = st.sidebar.selectbox("Application type", options["application_type"])
    initial_list_status = st.sidebar.selectbox("Initial listing status", options["initial_list_status"])
    region = st.sidebar.selectbox("Region", options["region"])

    st.sidebar.subheader("Credit bureau pull (soft inquiry)")
    fico = st.sidebar.slider("FICO score (approx.)", 620, 850, 700, step=1)
    dti = st.sidebar.slider("Debt-to-income ratio (%)", 0.0, 45.0, 18.0, step=0.5)
    credit_history_years = st.sidebar.slider("Length of credit history (years)", 1.0, 45.0, 12.0, step=0.5)
    revol_util = st.sidebar.slider("Revolving credit utilization (%)", 0.0, 150.0, 40.0, step=1.0)
    revol_bal = st.sidebar.number_input("Revolving balance ($)", 0, 200000, 8000, step=500)
    open_acc = st.sidebar.number_input("Open credit accounts", 0, 40, 8)
    total_acc = st.sidebar.number_input("Total credit accounts", 0, 80, 18)
    mort_acc = st.sidebar.number_input("Mortgage accounts", 0, 10, 0)
    delinq_2yrs = st.sidebar.number_input("Delinquencies (last 2 yrs)", 0, 10, 0)
    pub_rec = st.sidebar.number_input("Public records", 0, 10, 0)
    pub_rec_bankruptcies = st.sidebar.number_input("Public record bankruptcies", 0, 5, 0)
    inq_last_6mths = st.sidebar.number_input("Credit inquiries (last 6 mo.)", 0, 10, 1)
    tot_cur_bal = st.sidebar.number_input("Total current balance, all accounts ($)", 0, 1000000, 60000, step=1000)
    total_rev_hi_lim = st.sidebar.number_input("Total revolving credit limit ($)", 0, 500000, 25000, step=1000)
    acc_open_past_24mths = st.sidebar.number_input("Accounts opened (last 24 mo.)", 0, 20, 2)
    avg_cur_bal = st.sidebar.number_input("Average current balance per account ($)", 0, 200000, 8000, step=500)
    tot_coll_amt = st.sidebar.number_input("Total amount ever in collections ($)", 0, 100000, 0, step=100)
    tot_hi_cred_lim = st.sidebar.number_input("Total high credit limit ($)", 0, 1000000, 90000, step=1000)
    num_actv_bc_tl = st.sidebar.number_input("Active bankcard accounts", 0, 20, 3)

    return {
        "loan_amnt": loan_amnt, "term": term, "annual_inc": annual_inc,
        "emp_length_years": emp_length_years, "home_ownership": home_ownership,
        "verification_status": verification_status, "purpose": purpose,
        "application_type": application_type, "initial_list_status": initial_list_status,
        "region": region, "fico_avg": fico, "fico_range_low": fico - 2, "fico_range_high": fico + 2,
        "dti": dti, "credit_history_years": credit_history_years, "revol_util": revol_util,
        "revol_bal": revol_bal, "open_acc": open_acc, "total_acc": total_acc, "mort_acc": mort_acc,
        "delinq_2yrs": delinq_2yrs, "pub_rec": pub_rec, "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "inq_last_6mths": inq_last_6mths, "tot_cur_bal": tot_cur_bal, "total_rev_hi_lim": total_rev_hi_lim,
        "acc_open_past_24mths": acc_open_past_24mths, "avg_cur_bal": avg_cur_bal,
        "tot_coll_amt": tot_coll_amt, "tot_hi_cred_lim": tot_hi_cred_lim, "num_actv_bc_tl": num_actv_bc_tl,
    }


def sidebar_form_synthetic(options: dict) -> dict:
    st.sidebar.header("Loan application")
    st.sidebar.caption("Scored by the synthetic-data demo model (no real dataset found).")

    loan_amnt = st.sidebar.number_input("Loan amount ($)", 1000, 40000, 12000, step=500)
    term = st.sidebar.selectbox("Term (months)", [36, 60], index=0)
    int_rate = st.sidebar.slider("Interest rate (%)", 5.0, 31.0, 13.0, step=0.1)
    annual_inc = st.sidebar.number_input("Annual income ($)", 15000, 500000, 65000, step=1000)
    emp_length = st.sidebar.selectbox(
        "Employment length",
        ["< 1 year", "1 years", "2 years", "3 years", "4 years", "5 years",
         "6 years", "7 years", "8 years", "9 years", "10+ years"],
        index=5,
    )
    home_ownership = st.sidebar.selectbox("Home ownership", options["home_ownership"])
    verification_status = st.sidebar.selectbox("Income verification", options["verification_status"])
    purpose = st.sidebar.selectbox("Loan purpose", options["purpose"])
    application_type = st.sidebar.selectbox("Application type", options["application_type"])
    initial_list_status = st.sidebar.selectbox("Initial listing status", options["initial_list_status"])
    region = st.sidebar.selectbox("Region", options["region"])

    st.sidebar.subheader("Credit profile")
    fico = st.sidebar.slider("FICO score (approx.)", 620, 850, 700, step=1)
    dti = st.sidebar.slider("Debt-to-income ratio (%)", 0.0, 45.0, 18.0, step=0.5)
    credit_history_years = st.sidebar.slider("Length of credit history (years)", 1.0, 45.0, 12.0, step=0.5)
    revol_util = st.sidebar.slider("Revolving credit utilization (%)", 0.0, 150.0, 40.0, step=1.0)
    revol_bal = st.sidebar.number_input("Revolving balance ($)", 0, 200000, 8000, step=500)
    open_acc = st.sidebar.number_input("Open credit accounts", 0, 40, 8)
    total_acc = st.sidebar.number_input("Total credit accounts", 0, 80, 18)
    mort_acc = st.sidebar.number_input("Mortgage accounts", 0, 10, 0)
    delinq_2yrs = st.sidebar.number_input("Delinquencies (last 2 yrs)", 0, 10, 0)
    pub_rec = st.sidebar.number_input("Public records", 0, 10, 0)
    pub_rec_bankruptcies = st.sidebar.number_input("Public record bankruptcies", 0, 5, 0)
    inq_last_6mths = st.sidebar.number_input("Credit inquiries (last 6 mo.)", 0, 10, 1)

    monthly_rate = (int_rate / 100) / 12
    installment = (
        loan_amnt * monthly_rate * (1 + monthly_rate) ** term
    ) / (((1 + monthly_rate) ** term) - 1)

    return {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment,
        "emp_length": emp_length, "home_ownership": home_ownership, "annual_inc": annual_inc,
        "verification_status": verification_status, "purpose": purpose, "dti": dti,
        "delinq_2yrs": delinq_2yrs, "credit_history_years": credit_history_years,
        "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util,
        "total_acc": total_acc, "initial_list_status": initial_list_status,
        "application_type": application_type, "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies, "region": region,
        "fico_range_low": fico - 2, "fico_range_high": fico + 2, "inq_last_6mths": inq_last_6mths,
    }


def main():
    st.set_page_config(page_title="Lending Club Credit Risk", page_icon="💳", layout="wide")
    st.title("💳 Lending Club Credit Risk Model")
    st.caption(
        "Deployment-scoped credit risk scoring (XGBoost, calibrated, cost-sensitive threshold), "
        "built on application-time-only features. See README for the explicit loss/profit "
        "assumptions this decision threshold is based on."
    )

    artifacts, source = load_artifacts()
    if artifacts is None:
        st.error(
            "No trained model found. Run `python scripts/train_real.py` (real LendingClub data, "
            "requires the CSV in `data_real/`) or `python scripts/train.py` (synthetic demo data) first."
        )
        st.stop()

    if source == "real":
        st.success(
            f"Using the **real-data** model -- trained on {artifacts['n_resolved_loans']:,} resolved "
            f"LendingClub loans (real charge-off rate: {artifacts['real_charge_off_rate']:.1%}).",
            icon="✅",
        )
    else:
        st.warning(
            "Using the **synthetic-data demo** model -- no real LendingClub CSV was found in "
            "`data_real/`. See README for how to download it and run `scripts/train_real.py`.",
            icon="⚠️",
        )

    model = artifacts["model"]
    explainer = artifacts["explainer"]
    threshold = artifacts["threshold"]
    metrics = artifacts["metrics"]
    cost = artifacts["cost_assumptions"]

    if source == "real":
        application = sidebar_form_real(artifacts["categorical_options"])
        build_features = build_real_deployment_features
    else:
        application = sidebar_form_synthetic(artifacts["categorical_options"])
        build_features = build_deployment_features

    row = pd.DataFrame([application])
    X = build_features(row)

    p_default = model.predict_proba_default(X)[0]
    decision = "DECLINE" if p_default > threshold else "APPROVE"

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted default probability", f"{p_default:.1%}")
    col2.metric("Decision threshold", f"{threshold:.1%}")
    col3.metric("Lending decision", decision)

    if decision == "APPROVE":
        st.success(
            f"**APPROVE** -- predicted default probability ({p_default:.1%}) is below the "
            f"profit-optimal threshold ({threshold:.1%})."
        )
    else:
        st.error(
            f"**DECLINE** -- predicted default probability ({p_default:.1%}) exceeds the "
            f"profit-optimal threshold ({threshold:.1%})."
        )

    expected_value = (
        (1 - p_default) * cost["profit_margin"] - p_default * cost["loss_given_default"]
    ) * application["loan_amnt"]
    st.caption(
        f"Expected value of this loan at face value assumptions "
        f"(profit margin {cost['profit_margin']:.0%}, loss given default {cost['loss_given_default']:.0%}): "
        f"**${expected_value:,.0f}**"
    )

    st.subheader("Why the model made this prediction")
    explanation = explain_instance(model, explainer, X, top_k=12)
    display = explanation.copy()
    display["value"] = display["value"].round(3)
    display["shap_value"] = display["shap_value"].round(4)
    st.bar_chart(display.set_index("feature")["shap_value"])
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "SHAP values are computed on the model's raw log-odds output. Positive values push the "
        "prediction toward higher default risk; negative values push toward lower risk."
    )

    with st.expander("Model performance & methodology"):
        lines = [
            f"- **Data source:** {'real LendingClub 2007-2018Q4 accepted loans' if source == 'real' else 'synthetic demo generator'}",
            f"- **Deployment features:** {artifacts['n_deployment_features']} (application-time only)",
            f"- **Research (full) features:** {artifacts['n_research_features']}",
            f"- **PR-AUC retention:** {metrics['pr_auc_retention']:.1%} of the research model's test PR-AUC",
            f"- **Deployment test PR-AUC:** {metrics['deployment_test']['pr_auc']:.4f}",
            f"- **Deployment test ROC-AUC:** {metrics['deployment_test']['roc_auc']:.4f}",
            f"- **Deployment test Brier score:** {metrics['deployment_test']['brier_score']:.4f}",
            f"- **Validation approval rate at threshold:** {metrics['threshold_search']['val_approval_rate']:.1%}",
            f"- **Validation default rate among approved loans:** {metrics['threshold_search']['val_default_rate_of_approved']:.1%}",
        ]
        st.markdown("\n".join(lines))
        st.caption(
            f"Trained {artifacts['trained_at']} (seed={artifacts['random_seed']}). See "
            f"`reports/model_report{'_real' if source == 'real' else ''}.md` for the full model card."
        )


if __name__ == "__main__":
    main()
