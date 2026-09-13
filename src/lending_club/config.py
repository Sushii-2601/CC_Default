"""Central configuration: paths, constants, and reproducibility settings."""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "lending_club_synthetic.csv"
DEPLOYMENT_ARTIFACTS_PATH = ARTIFACTS_DIR / "deployment_artifacts.joblib"
METRICS_REPORT_PATH = REPORTS_DIR / "metrics.json"
MODEL_CARD_PATH = REPORTS_DIR / "model_report.md"

# --- Reproducibility ----------------------------------------------------
RANDOM_SEED = 42

# --- Dataset ------------------------------------------------------------
N_SYNTHETIC_ROWS = 60_000
TARGET_COLUMN = "loan_status"  # 1 = charged off / default, 0 = fully paid

# --- Business / cost model ----------------------------------------------
# Expected-loss framework used for cost-sensitive thresholding.
# All figures are explicit, stated assumptions for this portfolio project,
# not derived from proprietary lender data.
LOSS_GIVEN_DEFAULT = 0.60   # fraction of outstanding principal lost on default
PROFIT_MARGIN = 0.18        # net interest margin earned on a fully-paid loan

# --- Train / val / test split --------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # taken out of the remaining train split

# --- Real LendingClub data (2007-2018Q4 accepted loans, 151 raw columns) --
# Not vendored into the repo (1.6 GB) -- download separately, e.g.:
#   curl -L -o data_real/LendingClub_2007_to_2018Q4.csv \
#     https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/LendingClub_2007_to_2018Q4.csv
REAL_DATA_DIR = ROOT_DIR / "data_real"
REAL_RAW_CSV_PATH = REAL_DATA_DIR / "LendingClub_2007_to_2018Q4.csv"
REAL_DEPLOYMENT_ARTIFACTS_PATH = ARTIFACTS_DIR / "deployment_artifacts_real.joblib"
REAL_METRICS_REPORT_PATH = REPORTS_DIR / "metrics_real.json"
REAL_MODEL_CARD_PATH = REPORTS_DIR / "model_report_real.md"

# Only these two loan_status values represent a resolved (fully observed)
# outcome; every other status (Current, Late, In Grace Period, Default,
# Issued) is a loan whose eventual outcome we don't yet know, and including
# it would put an undetermined target in the training data.
RESOLVED_LOAN_STATUSES = ["Fully Paid", "Charged Off"]
REAL_TARGET_COLUMN = "loan_status"  # 1 = Charged Off, 0 = Fully Paid, after mapping
