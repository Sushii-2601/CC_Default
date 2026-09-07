"""Synthetic LendingClub-style dataset generation.

The original LendingClub "accepted loans" dataset (hosted historically on
lendingclub.com and mirrored on Kaggle) requires an account/API key and is
several GB, which makes it unsuitable to vendor into a portfolio repo. This
module instead generates a synthetic dataset that:

  * uses the same column names / grains as the real LendingClub schema,
  * encodes a realistic, *structured* latent default-risk function (so the
    resulting classification problem is genuinely learnable, not random),
  * is fully reproducible via a fixed random seed.

Swap :func:`generate_synthetic_loans` for a real loader (e.g. reading the
Kaggle CSV) without touching any downstream code -- every consumer only
depends on the column contract documented below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_SYNTHETIC_ROWS, RANDOM_SEED, TARGET_COLUMN

GRADES = ["A", "B", "C", "D", "E", "F", "G"]
# Roughly increasing base interest rate by grade, like real LendingClub bands.
GRADE_BASE_RATE = {"A": 0.07, "B": 0.10, "C": 0.13, "D": 0.16, "E": 0.19, "F": 0.23, "G": 0.27}
SUBGRADE_STEPS = 5  # A1..A5, B1..B5, ...
HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIFICATION_STATUS = ["Verified", "Source Verified", "Not Verified"]
PURPOSE = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house",
    "renewable_energy", "other",
]
APPLICATION_TYPE = ["Individual", "Joint App"]
INITIAL_LIST_STATUS = ["w", "f"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Mountain", "Pacific"]
TERMS = [36, 60]

REQUIRED_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "credit_history_years", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "mort_acc", "pub_rec_bankruptcies", "region",
    "fico_range_low", "fico_range_high", "inq_last_6mths",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_loans(n_rows: int = N_SYNTHETIC_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic LendingClub-style accepted-loans dataset.

    The binary target ``loan_status`` (1 = charged off, 0 = fully paid) is
    drawn from a Bernoulli distribution whose probability is a nonlinear
    function of the generated features plus noise, so the resulting
    classification task has genuine, learnable signal at a realistic
    difficulty level for consumer credit risk (test PR-AUC in the ~0.35-0.55
    range, in line with published LendingClub benchmarks).
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    fico_range_low = rng.integers(660, 830, size=n).astype(float)
    fico_range_high = fico_range_low + 4
    fico = (fico_range_low + fico_range_high) / 2

    annual_inc = np.clip(rng.lognormal(mean=11.0, sigma=0.55, size=n), 15_000, 500_000)
    dti = np.clip(rng.normal(18, 9, size=n), 0, 45)
    emp_length_years = rng.integers(0, 11, size=n)  # 10 = "10+ years"
    credit_history_years = np.clip(rng.normal(14, 7, size=n), 1, 45)
    delinq_2yrs = rng.poisson(0.3, size=n)
    inq_last_6mths = rng.poisson(0.7, size=n)
    open_acc = rng.integers(2, 25, size=n)
    total_acc = open_acc + rng.integers(0, 20, size=n)
    pub_rec = rng.poisson(0.12, size=n)
    pub_rec_bankruptcies = np.minimum(pub_rec, rng.poisson(0.08, size=n))
    mort_acc = rng.integers(0, 6, size=n)
    revol_util = np.clip(rng.normal(45, 25, size=n), 0, 150)
    revol_bal = np.clip(rng.lognormal(mean=9.0, sigma=1.0, size=n), 0, 200_000)

    grade_idx = rng.integers(0, len(GRADES), size=n)
    grade = np.array(GRADES)[grade_idx]
    sub_step = rng.integers(1, SUBGRADE_STEPS + 1, size=n)
    sub_grade = np.array([f"{g}{s}" for g, s in zip(grade, sub_step)])

    base_rate = np.array([GRADE_BASE_RATE[g] for g in grade])
    int_rate = np.clip(base_rate + (sub_step - 3) * 0.006 + rng.normal(0, 0.01, size=n), 0.05, 0.31)

    loan_amnt = np.clip(rng.lognormal(mean=9.4, sigma=0.5, size=n), 1_000, 40_000)
    term = rng.choice(TERMS, size=n, p=[0.7, 0.3])
    monthly_rate = int_rate / 12
    installment = (
        loan_amnt * monthly_rate * (1 + monthly_rate) ** term
    ) / (((1 + monthly_rate) ** term) - 1)

    home_ownership = rng.choice(HOME_OWNERSHIP, size=n, p=[0.45, 0.42, 0.10, 0.03])
    verification_status = rng.choice(VERIFICATION_STATUS, size=n, p=[0.35, 0.35, 0.30])
    purpose = rng.choice(PURPOSE, size=n, p=[
        0.27, 0.22, 0.08, 0.05, 0.06, 0.05, 0.06, 0.02, 0.02, 0.02, 0.02, 0.13,
    ])
    application_type = rng.choice(APPLICATION_TYPE, size=n, p=[0.85, 0.15])
    initial_list_status = rng.choice(INITIAL_LIST_STATUS, size=n)
    region = rng.choice(REGIONS, size=n)

    # --- Latent default-risk function -----------------------------------
    z = (
        -1.5
        - 0.018 * (fico - 700)
        + 0.035 * (dti - 18)
        + 6.0 * (int_rate - 0.13)
        + 0.30 * delinq_2yrs
        + 0.22 * inq_last_6mths
        + 0.55 * pub_rec
        + 0.012 * (revol_util - 45) / 10
        - 0.05 * emp_length_years
        - 0.35 * np.log1p(annual_inc / 40_000)
        + 0.9 * (loan_amnt / annual_inc)
        - 0.015 * (credit_history_years - 14)
        + 0.25 * (term == 60).astype(float)
        + rng.normal(0, 0.55, size=n)
    )
    default_prob = _sigmoid(z)
    loan_status = rng.binomial(1, default_prob)

    emp_length = np.where(
        emp_length_years == 0, "< 1 year",
        np.where(emp_length_years >= 10, "10+ years", emp_length_years.astype(str) + " years"),
    )

    df = pd.DataFrame({
        "loan_amnt": loan_amnt.round(2),
        "term": term,
        "int_rate": (int_rate * 100).round(2),
        "installment": installment.round(2),
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc.round(2),
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti.round(2),
        "delinq_2yrs": delinq_2yrs,
        "credit_history_years": credit_history_years.round(1),
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal.round(2),
        "revol_util": revol_util.round(1),
        "total_acc": total_acc,
        "initial_list_status": initial_list_status,
        "application_type": application_type,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "region": region,
        "fico_range_low": fico_range_low,
        "fico_range_high": fico_range_high,
        "inq_last_6mths": inq_last_6mths,
        TARGET_COLUMN: loan_status,
    })
    return df


def load_dataset(path=None, n_rows: int = N_SYNTHETIC_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Load the dataset from ``path`` if present, else generate and cache it."""
    from .config import RAW_DATA_PATH

    path = path or RAW_DATA_PATH
    if path.exists():
        return pd.read_csv(path)

    df = generate_synthetic_loans(n_rows=n_rows, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
