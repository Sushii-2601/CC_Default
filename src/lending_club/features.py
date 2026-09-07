"""Feature engineering for the research (full) and deployment (application-
time) feature sets.

Two feature sets are built from the same raw loan record:

* **Research set** -- everything, including the LendingClub-assigned
  ``grade``/``sub_grade`` risk bands, plus engineered ratios, logs,
  polynomials, and binned dummies. This is the "fuller model" used as an
  upper-bound reference.
* **Deployment set** -- only fields a borrower or loan officer could supply
  at application time (no internal risk grade), one-hot encoded. This is
  the set the shipped model actually uses.

Both builders return a DataFrame with a *fixed* column contract (via
``reindex``), so a single raw record can be transformed and scored without
ever needing the full training data in memory -- required for the Streamlit
app's single-application scoring path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    APPLICATION_TYPE,
    GRADES,
    HOME_OWNERSHIP,
    INITIAL_LIST_STATUS,
    PURPOSE,
    REGIONS,
    VERIFICATION_STATUS,
)

_EMP_LENGTH_MAP = {"< 1 year": 0, "10+ years": 10}


def _emp_length_years(series: pd.Series) -> pd.Series:
    def parse(v):
        if pd.isna(v):
            return 0.0
        v = str(v)
        if v in _EMP_LENGTH_MAP:
            return float(_EMP_LENGTH_MAP[v])
        digits = "".join(c for c in v if c.isdigit())
        return float(digits) if digits else 0.0

    return series.map(parse)


def _one_hot(df: pd.DataFrame, column: str, categories: list[str], prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or column
    cat = pd.Categorical(df[column], categories=categories)
    dummies = pd.get_dummies(cat, prefix=prefix, drop_first=True)
    return dummies.astype(float)


DEPLOYMENT_NUMERIC = [
    "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "credit_history_years", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "emp_length_years",
]


def _base_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["loan_amnt"] = df["loan_amnt"].astype(float)
    out["term"] = df["term"].astype(float)
    out["int_rate"] = df["int_rate"].astype(float)
    out["installment"] = df["installment"].astype(float)
    out["annual_inc"] = df["annual_inc"].astype(float)
    out["dti"] = df["dti"].astype(float)
    out["delinq_2yrs"] = df["delinq_2yrs"].astype(float)
    out["credit_history_years"] = df["credit_history_years"].astype(float)
    out["open_acc"] = df["open_acc"].astype(float)
    out["pub_rec"] = df["pub_rec"].astype(float)
    out["revol_bal"] = df["revol_bal"].astype(float)
    out["revol_util"] = df["revol_util"].astype(float)
    out["total_acc"] = df["total_acc"].astype(float)
    out["mort_acc"] = df["mort_acc"].astype(float)
    out["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].astype(float)
    out["fico_range_low"] = df["fico_range_low"].astype(float)
    out["fico_range_high"] = df["fico_range_high"].astype(float)
    out["inq_last_6mths"] = df["inq_last_6mths"].astype(float)
    out["emp_length_years"] = _emp_length_years(df["emp_length"])
    return out


def build_deployment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 39-ish application-time feature matrix used in production."""
    numeric = _base_numeric(df)
    onehots = [
        _one_hot(df, "home_ownership", HOME_OWNERSHIP),
        _one_hot(df, "verification_status", VERIFICATION_STATUS),
        _one_hot(df, "purpose", PURPOSE),
        _one_hot(df, "initial_list_status", INITIAL_LIST_STATUS),
        _one_hot(df, "application_type", APPLICATION_TYPE),
        _one_hot(df, "region", REGIONS),
    ]
    result = pd.concat([numeric, *onehots], axis=1)
    result.columns = [str(c) for c in result.columns]
    return result


def build_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the fuller research feature matrix (deployment set + grade/
    sub-grade bands + engineered ratios, logs, polynomials, and bins)."""
    deployment = build_deployment_features(df)

    sub_grades = sorted({f"{g}{i}" for g in GRADES for i in range(1, 6)})
    grade_dummies = _one_hot(df, "grade", GRADES)
    sub_grade_dummies = _one_hot(df, "sub_grade", sub_grades)

    eng = pd.DataFrame(index=df.index)
    annual_inc = df["annual_inc"].astype(float).clip(lower=1.0)
    eng["loan_to_income"] = df["loan_amnt"].astype(float) / annual_inc
    eng["installment_to_income"] = (df["installment"].astype(float) * 12) / annual_inc
    eng["revol_bal_to_income"] = df["revol_bal"].astype(float) / annual_inc
    eng["fico_avg"] = (df["fico_range_low"].astype(float) + df["fico_range_high"].astype(float)) / 2
    eng["dti_x_int_rate"] = df["dti"].astype(float) * df["int_rate"].astype(float)
    eng["fico_x_int_rate"] = eng["fico_avg"] * df["int_rate"].astype(float)
    eng["log_annual_inc"] = np.log1p(annual_inc)
    eng["log_revol_bal"] = np.log1p(df["revol_bal"].astype(float).clip(lower=0))
    eng["log_loan_amnt"] = np.log1p(df["loan_amnt"].astype(float))
    eng["dti_sq"] = df["dti"].astype(float) ** 2
    eng["int_rate_sq"] = df["int_rate"].astype(float) ** 2

    credit_hist_bin = pd.cut(
        df["credit_history_years"].astype(float),
        bins=[0, 5, 10, 20, 100], labels=["ch_lt5", "ch_5to10", "ch_10to20", "ch_gt20"],
    )
    fico_bin = pd.cut(
        eng["fico_avg"], bins=[0, 660, 700, 740, 780, 900],
        labels=["fico_lt660", "fico_660_700", "fico_700_740", "fico_740_780", "fico_gt780"],
    )
    util_bin = pd.cut(
        df["revol_util"].astype(float), bins=[-1, 30, 60, 90, 200],
        labels=["util_lt30", "util_30to60", "util_60to90", "util_gt90"],
    )
    bin_dummies = pd.concat([
        pd.get_dummies(credit_hist_bin, drop_first=True),
        pd.get_dummies(fico_bin, drop_first=True),
        pd.get_dummies(util_bin, drop_first=True),
    ], axis=1).astype(float)

    flags = pd.DataFrame(index=df.index)
    flags["has_delinq"] = (df["delinq_2yrs"].astype(float) > 0).astype(float)
    flags["has_pub_rec"] = (df["pub_rec"].astype(float) > 0).astype(float)
    flags["has_mort_acc"] = (df["mort_acc"].astype(float) > 0).astype(float)
    flags["high_dti_flag"] = (df["dti"].astype(float) > 30).astype(float)
    flags["high_util_flag"] = (df["revol_util"].astype(float) > 80).astype(float)

    result = pd.concat(
        [deployment, grade_dummies, sub_grade_dummies, eng, bin_dummies, flags], axis=1,
    )
    result.columns = [str(c) for c in result.columns]
    return result


def align_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex a feature matrix onto a fixed training-time column contract."""
    return df.reindex(columns=columns, fill_value=0.0)
