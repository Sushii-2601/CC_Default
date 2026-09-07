---
title: Lending Club Credit Risk Model
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Lending Club Credit Risk Model

A deployment-scoped credit risk scoring model (XGBoost, calibrated, cost-sensitive threshold) for
consumer installment loans, built end-to-end: synthetic data generation → feature engineering →
model training & calibration → profit-optimized decision thresholding → SHAP explainability → a
Streamlit scoring app.

The model is intentionally restricted to **application-time features** — inputs a borrower or
loan officer could realistically provide before underwriting — rather than the full set of fields
a lender's internal systems accumulate over a loan's life. A fuller "research" model trained on
every available field (including LendingClub's internal risk grade) is kept only as a reference
upper bound; run `python scripts/train.py` to reproduce the exact retention figure on your machine
(printed at the end of the run and written to [`reports/model_report.md`](reports/model_report.md)).

Includes a per-prediction SHAP explanation and a lending decision at a profit-optimized threshold
(not accuracy/F1-optimized), derived from an explicit, labeled expected-loss framework.

> **Portfolio/demo project.** The economic assumptions (loss given default, profit margin) are
> explicitly stated in [`src/lending_club/config.py`](src/lending_club/config.py), not derived
> from proprietary lender data. The dataset is synthetically generated (see **Data** below) rather
> than the real LendingClub accepted-loans dataset, since that dataset requires a Kaggle account
> and is several GB — unsuitable to vendor into a portfolio repo. The generator mirrors the real
> schema and encodes a realistic, structured latent default-risk function, so the resulting
> classification problem has genuine, non-trivial signal rather than being random.

## What it does

1. **Generates data** — [`src/lending_club/data.py`](src/lending_club/data.py) produces a
   LendingClub-schema-compatible dataset (loan terms, borrower financials, credit-bureau fields)
   with a structured latent default probability, so the resulting labels are learnable.
2. **Engineers two feature sets** — [`src/lending_club/features.py`](src/lending_club/features.py):
   - a **research** set: every field, including LendingClub's internal grade/sub-grade bands,
     plus engineered ratios, log transforms, polynomial terms, and binned dummies;
   - a **deployment** set: only application-time fields, one-hot encoded.
3. **Trains & calibrates** — [`src/lending_club/modeling.py`](src/lending_club/modeling.py) fits
   an `XGBClassifier` on each feature set, then calibrates its raw scores with isotonic regression
   fit on a held-out validation split (never train or test), so predicted probabilities are honest.
4. **Picks a decision threshold by profit, not accuracy** —
   [`src/lending_club/metrics.py`](src/lending_club/metrics.py) grid-searches the
   default-probability threshold that maximizes realized expected profit on the validation split,
   under an explicit cost model (`profit_margin` on fully-paid loans, `loss_given_default` on
   defaults) — then reports realized profit on the untouched test split.
5. **Explains every prediction** — [`src/lending_club/explain.py`](src/lending_club/explain.py)
   wraps `shap.TreeExplainer` to rank the features driving each individual score.
6. **Serves it** — [`app.py`](app.py) is a Streamlit app: fill out a loan application in the
   sidebar and get a calibrated default probability, an APPROVE/DECLINE decision, and a SHAP
   breakdown of why.

## Project structure

```
.
├── app.py                       # Streamlit deployment app
├── src/lending_club/
│   ├── config.py                 # paths, seed, cost-model assumptions
│   ├── data.py                   # synthetic LendingClub-style data generator
│   ├── features.py                # research (full) & deployment feature engineering
│   ├── modeling.py                # training + isotonic calibration
│   ├── metrics.py                 # PR-AUC/ROC-AUC + profit-optimal threshold search
│   └── explain.py                 # SHAP per-prediction explanations
├── scripts/
│   ├── generate_data.py          # regenerate the synthetic dataset CSV
│   └── train.py                  # full pipeline: data → features → train → save artifacts
├── tests/                        # pytest unit + pipeline smoke tests
├── artifacts/deployment_artifacts.joblib   # trained model + calibrator + SHAP explainer + threshold
├── reports/                      # generated metrics.json + model_report.md (model card)
├── Dockerfile                    # docker sdk deployment (e.g. Hugging Face Spaces)
└── requirements.txt
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .

# Train the model (generates data on first run, ~1-2 min on a laptop CPU)
python scripts/train.py

# Run the app
streamlit run app.py
```

Run the test suite with `pytest`.

### Reproducing with different data volume / seed

```bash
python scripts/train.py --n-rows 100000 --seed 7
```

### Docker

```bash
docker build -t lending-club-credit-risk .
docker run -p 8501:8501 lending-club-credit-risk
```

The image trains a model at build time if `artifacts/deployment_artifacts.joblib` isn't already
present in the build context (it is committed to this repo, so a normal build just reuses it).

## Data

Real LendingClub "accepted loans" data is not included in this repository — it requires a Kaggle
account, is several GB, and its license doesn't permit redistribution here. Instead,
[`generate_synthetic_loans`](src/lending_club/data.py) synthesizes a dataset that:

- uses the same column names and grains as the real schema (`loan_amnt`, `term`, `int_rate`,
  `dti`, `fico_range_low/high`, `revol_util`, `delinq_2yrs`, `purpose`, `home_ownership`, ...);
- derives `loan_status` from a nonlinear latent-risk function of those fields plus noise (higher
  interest rate, higher DTI, more delinquencies/inquiries, higher utilization, and lower
  income/credit history all push risk up), so the classification task carries genuine signal at a
  realistic difficulty for consumer credit (comparable-order PR-AUC to published LendingClub
  benchmarks — see [`reports/model_report.md`](reports/model_report.md) for this run's exact
  numbers);
- is fully reproducible from a fixed random seed.

To use real data instead, replace the body of `load_dataset()` in `data.py` with a loader for your
own CSV — every downstream module only depends on the column contract documented in
`REQUIRED_COLUMNS`, so nothing else needs to change.

## Methodology notes

- **Why calibrate?** Raw gradient-boosted tree scores are not well-calibrated probabilities.
  Isotonic regression fit on a held-out validation split (distinct from the split used for
  training and the split used for final evaluation) corrects this without leaking test
  information.
- **Why a profit-optimal threshold instead of F1/accuracy?** A lender's actual objective is
  expected profit, which is asymmetric: declining a loan that would have defaulted saves the full
  loss-given-default; declining one that would have paid forfeits the profit margin, not a
  symmetric "error." Optimizing a symmetric metric like F1 optimizes the wrong objective for this
  decision.
- **Why report PR-AUC over ROC-AUC as the headline metric?** Default is a minority class; PR-AUC
  is far more sensitive to performance on the positive (default) class than ROC-AUC, which can
  look deceptively good under class imbalance.

## Model card

After training, see [`reports/model_report.md`](reports/model_report.md) for this run's feature
counts, PR-AUC/ROC-AUC/Brier score for both feature sets, the PR-AUC retention figure, the chosen
threshold, and realized test-set profit versus an "approve everyone" baseline.

## License

MIT — see [LICENSE](LICENSE).
