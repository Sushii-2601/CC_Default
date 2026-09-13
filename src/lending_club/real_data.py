"""Loader and leakage audit for the real LendingClub accepted-loans dataset
(2007-2018Q4, 151 raw columns, ~2.26M rows before filtering to resolved
outcomes).

This is the real-data counterpart to :mod:`lending_club.data` (which
generates a schema-matched synthetic dataset for when the 1.6 GB real file
isn't available). Every column-dropping decision below is grouped and
justified explicitly, rather than picked to hit a target count -- see
:data:`LEAKAGE_COLUMNS` for the post-issuance leakage audit that is this
module's central contribution.

Memory note: the raw CSV is loaded with an explicit ``usecols``/``dtype``
contract so pandas' C parser never materializes the ~68 dropped columns at
all (not "load then drop") and numeric columns are read directly as
float32 -- this keeps a ~1.3M-row load to roughly the low hundreds of MB
rather than the multi-GB a naive ``pd.read_csv`` of the full 151-column
file would need.
"""
from __future__ import annotations

import pandas as pd

from .config import REAL_RAW_CSV_PATH, RESOLVED_LOAN_STATUSES

# --- Leakage audit: columns that only exist because of what happened AFTER
# the loan was issued -- i.e. they encode (directly or by strong proxy) the
# very outcome we're trying to predict at application time. Grouped by why
# each group leaks. 17 + 15 + 7 = 39 columns.
PAYMENT_PERFORMANCE_LEAKAGE = [
    "funded_amnt", "funded_amnt_inv",           # funding outcome, set after approval
    "out_prncp", "out_prncp_inv",                 # current outstanding balance -- only exists once the loan is live
    "total_pymnt", "total_pymnt_inv",              # cumulative payments received -- directly encodes repayment
    "total_rec_prncp", "total_rec_int",             # ditto, principal/interest actually collected
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",  # post-default collections activity
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",  # servicing history
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",  # FICO as of the *latest* pull, not application time
]  # 17

HARDSHIP_PROGRAM_LEAKAGE = [
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
]  # 15 -- a hardship plan only exists because the borrower was already struggling to repay

DEBT_SETTLEMENT_LEAKAGE = [
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
]  # 7 -- a settlement only exists on a loan that already went into default

LEAKAGE_COLUMNS = PAYMENT_PERFORMANCE_LEAKAGE + HARDSHIP_PROGRAM_LEAKAGE + DEBT_SETTLEMENT_LEAKAGE

# --- Dropped for other reasons -- NOT leakage, just not useful features ---
NON_PREDICTIVE_JUNK = [
    "id", "member_id",       # identifiers
    "url", "desc", "title",  # free text / redundant with `purpose`
    "zip_code",               # 3-digit + "xx" prefix, effectively redundant with addr_state at this resolution
    "emp_title",               # free-text job title, extremely high cardinality
    "pymnt_plan",               # near-constant (>99.9% "n")
    "policy_code",                # constant column in this export
]  # 9

SPARSE_JOINT_APPLICATION_COLUMNS = [
    "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
    "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog",
]  # 16 -- >=99% missing; only populated for the rare "Joint App" application_type

HIGH_MISSINGNESS_NUMERIC = [
    "mths_since_last_record", "mths_since_recent_bc_dlq",
    "mths_since_last_major_derog", "mths_since_recent_revol_delinq",
]  # 4 -- 64-82% missing in this export; dropped rather than imputed to avoid an imputation-dominated feature

ALL_DROPPED_COLUMNS = (
    LEAKAGE_COLUMNS + NON_PREDICTIVE_JUNK + SPARSE_JOINT_APPLICATION_COLUMNS + HIGH_MISSINGNESS_NUMERIC
)

# Columns kept as categorical/object dtype (everything else that survives
# the drop list is coerced to float32).
_CATEGORICAL_COLUMNS = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership", "verification_status",
    "loan_status", "purpose", "addr_state", "initial_list_status", "application_type",
    "disbursement_method",
]
_DATE_COLUMNS = ["issue_d", "earliest_cr_line"]

_STATE_TO_REGION = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NY", "NJ", "PA"], "Northeast"),
    **dict.fromkeys(["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY",
                      "MS", "TN", "AR", "LA", "OK", "TX"], "Southeast"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"], "Midwest"),
    **dict.fromkeys(["AZ", "NM"], "Southwest"),
    **dict.fromkeys(["CO", "ID", "MT", "UT", "NV", "WY"], "Mountain"),
    **dict.fromkeys(["AK", "CA", "HI", "OR", "WA"], "Pacific"),
}


def _build_usecols_and_dtypes(csv_path) -> tuple[list[str], dict]:
    """Read only the header row to build a memory-efficient usecols/dtype
    contract -- avoids ever parsing the ~68 dropped columns."""
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in header if c not in ALL_DROPPED_COLUMNS]
    dtype = {}
    for c in usecols:
        if c in _CATEGORICAL_COLUMNS or c in _DATE_COLUMNS:
            continue  # leave as default object dtype for now; cast after cleaning
        dtype[c] = "float32"
    return usecols, dtype


def load_real_loans(csv_path=None, nrows: int | None = None, chunksize: int = 150_000) -> pd.DataFrame:
    """Load and clean the real LendingClub dataset: filter to resolved
    outcomes, drop leaked/junk/sparse columns, derive a handful of
    origination-time features from raw dates, and return a tidy DataFrame.

    Reads via ``pd.read_csv(..., chunksize=...)`` rather than one single
    call: even with ``usecols``/``dtype`` already trimming the parsed
    columns to ~83, the C parser's peak tokenizing memory for the *full*
    ~2.26M-row, 1.6 GB file exceeded what this environment had free (a
    single-shot read hit a real ``MemoryError`` during development). Reading
    150K-row chunks, filtering each to resolved loans immediately, and
    concatenating only the much smaller filtered result keeps peak memory
    bounded to one chunk's footprint instead of the whole file's.

    Returns a DataFrame with the binary target as ``loan_status`` (1 =
    Charged Off, 0 = Fully Paid) plus every retained raw column, ready for
    :mod:`lending_club.real_features` to build feature matrices from.
    """
    csv_path = csv_path or REAL_RAW_CSV_PATH
    usecols, dtype = _build_usecols_and_dtypes(csv_path)

    resolved_chunks = []
    rows_seen = 0
    reader = pd.read_csv(
        csv_path, usecols=usecols, dtype=dtype, low_memory=False, chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk[chunk["loan_status"].isin(RESOLVED_LOAN_STATUSES)].copy()
        resolved_chunks.append(chunk)
        rows_seen += chunksize
        if nrows is not None and rows_seen >= nrows:
            break
    df = pd.concat(resolved_chunks, ignore_index=True)
    del resolved_chunks

    # Parse LendingClub's "Mon-YYYY" date strings with an explicit format --
    # orders of magnitude faster than pandas' per-element dateutil fallback
    # at 1M+ rows, and avoids the "could not infer format" warning.
    for c in _DATE_COLUMNS:
        df[c] = pd.to_datetime(df[c], format="%b-%Y", errors="coerce")

    df["loan_status"] = (df["loan_status"] == "Charged Off").astype("int8")

    # term: "36 months" -> 36 (int)
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype("int16")

    # emp_length: "10+ years" -> 10.0, "< 1 year" -> 0.0, "n/a"/NaN -> NaN
    emp_digits = df["emp_length"].astype(str).str.extract(r"(\d+)")[0]
    df["emp_length_years"] = pd.to_numeric(emp_digits, errors="coerce").astype("float32")
    df["emp_length_years"] = df["emp_length_years"].fillna(df["emp_length_years"].median())
    df = df.drop(columns=["emp_length"])

    # credit_history_years: years between the borrower's earliest credit
    # line and this loan's issue date -- an origination-time-derivable
    # feature, not leakage (both dates are known before the loan is funded).
    df["credit_history_years"] = (
        (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    ).astype("float32")
    df["credit_history_years"] = df["credit_history_years"].clip(lower=0)
    df = df.drop(columns=_DATE_COLUMNS)

    # region: a coarser, lower-cardinality bucket of addr_state, used by the
    # deployment feature set to avoid a 50-category one-hot block.
    df["region"] = df["addr_state"].map(_STATE_TO_REGION).fillna("Other")

    # fico_avg convenience column (kept alongside the raw low/high fields).
    df["fico_avg"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).astype("float32")

    for c in _CATEGORICAL_COLUMNS:
        if c in df.columns and c != "loan_status":
            df[c] = df[c].astype("category")

    return df.reset_index(drop=True)
