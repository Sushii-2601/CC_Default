"""Feature engineering for the real LendingClub dataset: a research (full,
leakage-safe) feature set and a deployment (application-time-only) feature
set, mirroring the two-tier design in :mod:`lending_club.features` but built
against the real 84-column cleaned frame from :mod:`lending_club.real_data`.

Deployment set design rule: excludes not just post-issuance leakage
(handled already in ``real_data.load_real_loans``) but also LendingClub's
own risk-pricing outputs -- ``grade``, ``sub_grade``, ``int_rate`` -- and
``installment`` (which mathematically reconstructs ``int_rate`` given
``loan_amnt``/``term``, so keeping it would silently reintroduce exactly
what excluding ``int_rate`` was meant to avoid). What's left is what a
borrower's application plus a soft credit-bureau pull can supply *before*
LendingClub's underwriting model has priced the loan -- the only inputs a
genuinely pre-decision system could use.
"""
from __future__ import annotations

import pandas as pd

REAL_HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER", "ANY", "NONE"]
REAL_VERIFICATION_STATUS = ["Verified", "Source Verified", "Not Verified"]
REAL_APPLICATION_TYPE = ["Individual", "Joint App"]
REAL_INITIAL_LIST_STATUS = ["w", "f"]
REAL_REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "Mountain", "Pacific", "Other"]
REAL_GRADES = ["A", "B", "C", "D", "E", "F", "G"]
REAL_DISBURSEMENT_METHOD = ["Cash", "DirectPay"]

# Purposes bucketed to the handful with real volume; everything else folds
# into "other" -- both to keep the one-hot block from being dominated by
# near-empty categories and because low-count categories make for noisy,
# overfitting-prone dummy columns regardless of dimensionality goals.
KEEP_PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other",
                  "major_purchase", "small_business"]

DEPLOYMENT_NUMERIC = [
    "loan_amnt", "term", "annual_inc", "dti", "emp_length_years", "delinq_2yrs",
    "fico_avg", "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "credit_history_years", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "tot_coll_amt", "tot_hi_cred_lim", "num_actv_bc_tl",
]

# Everything else numeric that survived the leakage/junk/sparsity audit in
# real_data.py -- used only by the research feature set.
RESEARCH_EXTRA_NUMERIC = [
    "acc_now_delinq", "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
    "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m",
    "max_bal_bc", "all_util", "inq_fi", "total_cu_tl", "inq_last_12m", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "tax_liens", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "collections_12_mths_ex_med", "int_rate", "installment",
]


def _one_hot(df: pd.DataFrame, column: str, categories: list[str], prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or column
    cat = pd.Categorical(df[column].astype(str), categories=categories)
    return pd.get_dummies(cat, prefix=prefix, drop_first=True).astype("float32")


def _numeric_block(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].astype("float32").copy()
    # Median-impute any remaining missingness (bureau fields with a small
    # amount of missing data that survived the high-missingness column drop
    # in real_data.py) and add a per-column missing-indicator so the model
    # can still use "was this reported" as signal.
    for c in columns:
        na = out[c].isna()
        if na.any():
            out[c] = out[c].fillna(out[c].median())
            out[f"{c}_was_missing"] = na.astype("float32")
    return out


def build_real_deployment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Application-time-only feature matrix: no grade/sub_grade/int_rate/
    installment, no post-issuance fields (already excluded upstream)."""
    numeric = _numeric_block(df, DEPLOYMENT_NUMERIC)
    purpose_bucketed = df["purpose"].astype(str).where(df["purpose"].astype(str).isin(KEEP_PURPOSES), "other")
    onehots = [
        _one_hot(df, "home_ownership", REAL_HOME_OWNERSHIP),
        _one_hot(df, "verification_status", REAL_VERIFICATION_STATUS),
        _one_hot(pd.DataFrame({"purpose": purpose_bucketed}), "purpose", KEEP_PURPOSES),
        _one_hot(df, "application_type", REAL_APPLICATION_TYPE),
        _one_hot(df, "initial_list_status", REAL_INITIAL_LIST_STATUS),
        _one_hot(df, "region", REAL_REGIONS),
    ]
    result = pd.concat([numeric, *onehots], axis=1)
    result.columns = [str(c) for c in result.columns]
    return result


def build_real_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full leakage-safe feature matrix: deployment features + grade/
    sub_grade/int_rate/installment + every remaining bureau-derived numeric
    field + addr_state (fine-grained, vs. deployment's coarser region)."""
    deployment = build_real_deployment_features(df)
    extra_numeric = _numeric_block(df, RESEARCH_EXTRA_NUMERIC)
    onehots = [
        _one_hot(df, "grade", REAL_GRADES),
        _one_hot(df, "disbursement_method", REAL_DISBURSEMENT_METHOD),
    ]
    result = pd.concat([deployment, extra_numeric, *onehots], axis=1)
    result.columns = [str(c) for c in result.columns]
    return result


def align_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.reindex(columns=columns, fill_value=0.0)
