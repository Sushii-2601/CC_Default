# Model Report -- Lending Club Credit Risk

_Generated 2026-09-07 20:28:21 · seed=42_

## Feature sets
- Research (full) model: **109** features (includes LendingClub grade/sub-grade bands plus engineered ratios, logs, polynomials, and binned dummies).
- Deployment model: **43** features (application-time inputs only, one-hot encoded).

## Test-set performance
| Metric | Deployment model | Research model |
|---|---|---|
| PR-AUC | 0.3884 | 0.3919 |
| ROC-AUC | 0.7683 | 0.7662 |
| Brier score | 0.1188 | 0.1192 |

**PR-AUC retention: 99.1%** of the research model's PR-AUC, using only application-time features.

## Cost-sensitive decision threshold
- Loss given default assumption: 60% of outstanding principal.
- Profit margin assumption: 18% net interest margin on a fully-paid loan.
- Profit-optimal threshold on P(default), selected on the validation split: **0.233**
- Validation approval rate at threshold: 78.5%
- Validation default rate among approved loans: 10.7%

## Realized profit on the untouched test split
- At the profit-optimal threshold: **8,993,128** (sum of loan_amnt-scaled profit/loss units)
- Naive "approve everyone" baseline: 5,449,500

## Notes
This is a portfolio/demo project. The dataset is synthetically generated to mirror the column schema and realistic statistical structure of LendingClub's public accepted-loans data (the real dataset requires a Kaggle account and is several GB, so it is not vendored into this repo). Loss-given-default and profit-margin assumptions are explicitly stated above, not derived from proprietary lender data.