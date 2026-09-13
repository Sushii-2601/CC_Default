# Model Report -- Lending Club Credit Risk (REAL DATA)

_Generated 2026-09-08 10:36:32 · seed=42 · source: real LendingClub 2007-2018Q4 accepted loans_

## Dataset
- Resolved (Fully Paid / Charged Off) loans used: **1,345,310**
- Real charge-off rate: **19.96%**
- Post-issuance leakage columns identified and dropped: **39** (payment/servicing history, hardship-program fields, debt-settlement fields -- see `src/lending_club/real_data.py` for the full, grouped, justified list)

## Feature sets
- Research (leakage-safe, full) model: **154** features (includes grade/sub_grade/int_rate plus every remaining bureau-derived field).
- Deployment model: **57** features (application-time only -- excludes grade/sub_grade/int_rate/installment in addition to all post-issuance leakage).

## Test-set performance
| Metric | Deployment model | Research model |
|---|---|---|
| PR-AUC | 0.3788 | 0.3963 |
| ROC-AUC | 0.7140 | 0.7286 |
| Brier score | 0.1442 | 0.1420 |

**PR-AUC retention: 95.6%** of the research model's PR-AUC, using only application-time features.

## Cost-sensitive decision threshold vs. the F1-optimal cutoff
- Loss given default assumption: 60% of outstanding principal.
- Profit margin assumption: 18% net interest margin on a fully-paid loan.
- **Profit-optimal threshold** (validation split): **0.2300**
- F1-optimal threshold (validation split, for comparison): **0.2175**
- Validation approval rate at the profit-optimal threshold: 67.7%
- Validation default rate among approved loans: 12.9%

## Realized profit on the untouched test split (dollars, since loan_amnt is real USD)
- Approve everyone (baseline): **$32,250,150**
- At the F1-optimal threshold: **$140,428,580**
- At the profit-optimal threshold: **$140,688,346**

## Notes
This run uses the real LendingClub 2007-2018Q4 accepted-loans dataset (not the synthetic generator in `lending_club.data`). Loss-given-default and profit-margin assumptions remain explicitly stated constants (`src/lending_club/config.py`), not fitted from data.