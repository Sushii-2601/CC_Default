#!/usr/bin/env python
"""Regenerate the synthetic LendingClub-style dataset CSV.

Usage:
    python scripts/generate_data.py [--n-rows 60000] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_club.config import N_SYNTHETIC_ROWS, RANDOM_SEED, RAW_DATA_PATH
from lending_club.data import generate_synthetic_loans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=N_SYNTHETIC_ROWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = generate_synthetic_loans(n_rows=args.n_rows, seed=args.seed)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {RAW_DATA_PATH}")
    print(f"Default rate: {df['loan_status'].mean():.2%}")
