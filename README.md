---
title: Credit Default Risk Modeling and Deployment
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Credit Default Risk Modeling and Deployment

Predicts loan default probability **at application stage** on the real LendingClub 2007–2018Q4
accepted-loans dataset, using only leakage-safe, application-time features and cost-sensitive
decisioning — not accuracy/F1-optimized decisioning.

- **1,345,310** real, resolved LendingClub loans (Fully Paid / Charged Off), out of ~2.26M accepted
  loans in the source file.
- **39 post-issuance columns** audited and dropped for outcome leakage (payment/servicing history,
  hardship-program fields, debt-settlement fields) — see [`src/lending_club/real_data.py`](src/lending_club/real_data.py)
  for the full, grouped, justified list.
- **0.714 test ROC-AUC / 0.379 test PR-AUC** on the 57-feature, application-time-only deployment
  model — retaining **95.6%** of the 154-feature leakage-safe research model's PR-AUC.
- A **profit-optimal decision threshold (0.230)**, chosen by grid-searching realized profit under an
  explicit expected-loss cost model rather than by F1/accuracy — lifting realized test-portfolio
  profit from **$32.3M** (approve-everyone baseline) to **$140.7M** at the chosen threshold.
- A Streamlit app with per-prediction **SHAP** explanations, run locally (see **Status** below for
  exactly what "shipped" means here).

## Why this exists

Most public tutorials on this exact dataset train on the full historical record of a loan — including
fields that only exist because the loan was already issued and serviced (payment history,
collections, hardship programs, debt settlement). A model trained that way looks excellent offline
and is unusable in production, because none of those fields exist at the actual decision moment.
This project audits that gap explicitly instead of hoping no leaked column slipped in, quantifies
exactly what restricting to honest, application-time-only inputs costs in predictive power, and
optimizes the resulting decision threshold against real loan economics rather than a generic
classification metric.

## Status — what's real, precisely

| Claim | Status |
|---|---|
| Trained on the real LendingClub dataset (not synthetic) | **Yes** — see **Data** below |
| 1.35M loans, leakage audit, ROC-AUC, PR-AUC, threshold, profit figures | **Yes** — every number above is read directly from [`reports/metrics_real.json`](reports/metrics_real.json), regenerable by running [`scripts/train_real.py`](scripts/train_real.py) |
| Streamlit app with SHAP explanations | **Yes**, runs locally — verified serving real predictions in development |
| **Publicly hosted / deployed** | **No** — not currently live anywhere; see **Deployment** below for how to run it yourself (locally or via the included Dockerfile) |

## Two data paths

- **Real** (primary, [`scripts/train_real.py`](scripts/train_real.py) / [`src/lending_club/real_data.py`](src/lending_club/real_data.py) / [`src/lending_club/real_features.py`](src/lending_club/real_features.py)) —
  the actual LendingClub 2007–2018Q4 accepted-loans CSV (151 raw columns, ~1.6 GB). Not vendored into
  the repo; download it yourself (see **Data**) and `app.py` will use it automatically once trained.
- **Synthetic** (fallback demo, [`scripts/train.py`](scripts/train.py) / [`src/lending_club/data.py`](src/lending_club/data.py) / [`src/lending_club/features.py`](src/lending_club/features.py)) —
  a schema-matched synthetic generator with a structured latent-risk function, for anyone who wants to
  explore the pipeline without a 1.6 GB download. `app.py` falls back to this automatically if no real
  artifact is present.

## What it does

1. **Loads & audits the real data** — [`src/lending_club/real_data.py`](src/lending_club/real_data.py)
   reads the CSV in memory-bounded chunks (a single-shot read of the full file hit a real
   out-of-memory error during development — chunked reads fixed it), filters to resolved outcomes,
   and drops 39 grouped, justified leakage columns plus 29 junk/sparse/high-missingness columns.
2. **Engineers two feature sets** — [`src/lending_club/real_features.py`](src/lending_club/real_features.py):
   a **research** set (154 features: every retained field, including LendingClub's own
   grade/sub_grade/int_rate) and a **deployment** set (57 features: application-time-only, excluding
   grade/sub_grade/int_rate/installment too, since those are LendingClub's own risk-pricing output,
   not something a pre-decision system could use).
3. **Trains & calibrates** — [`src/lending_club/modeling.py`](src/lending_club/modeling.py) fits an
   `XGBClassifier` on each feature set with dynamic class-imbalance weighting, then calibrates raw
   scores with isotonic regression fit on a held-out validation split.
4. **Picks a decision threshold by profit, not accuracy** — [`src/lending_club/metrics.py`](src/lending_club/metrics.py)
   grid-searches the default-probability threshold that maximizes realized profit on the validation
   split under an explicit cost model, then reports realized profit on the untouched test split.
5. **Explains every prediction** — [`src/lending_club/explain.py`](src/lending_club/explain.py) wraps
   `shap.TreeExplainer` for exact, per-application feature attribution.
6. **Serves it** — [`app.py`](app.py) is a Streamlit app that auto-detects and prefers the real-data
   model, with a schema-appropriate sidebar form for whichever model is loaded.

## Project structure

```
.
├── app.py                          # Streamlit app -- prefers the real model, falls back to synthetic
├── src/lending_club/
│   ├── config.py                    # paths, seed, cost-model assumptions, real-data constants
│   ├── real_data.py                  # real CSV loader + leakage/junk/sparsity column audit
│   ├── real_features.py               # research (154-feat) & deployment (57-feat) real feature engineering
│   ├── data.py                         # synthetic LendingClub-style data generator (fallback demo)
│   ├── features.py                      # synthetic research/deployment feature engineering
│   ├── modeling.py                       # training + isotonic calibration (shared by both paths)
│   ├── metrics.py                         # PR-AUC/ROC-AUC + profit-optimal threshold search (shared)
│   └── explain.py                          # SHAP per-prediction explanations (shared)
├── scripts/
│   ├── train_real.py                # full pipeline on the real CSV
│   ├── generate_data.py              # regenerate the synthetic dataset CSV
│   └── train.py                       # full pipeline on synthetic data
├── tests/                           # 22 pytest tests (8 real-data tests skip gracefully without the CSV)
├── artifacts/
│   ├── deployment_artifacts_real.joblib   # real-data model + calibrator + explainer + threshold
│   └── deployment_artifacts.joblib         # synthetic-data model (fallback demo)
├── reports/                         # metrics_real.json + model_report_real.md (+ synthetic counterparts)
├── data_real/                       # gitignored -- put the downloaded real CSV here
├── Dockerfile
└── requirements.txt
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

### Real data (recommended)

```bash
mkdir -p data_real
curl -L -o data_real/LendingClub_2007_to_2018Q4.csv \
  https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/LendingClub_2007_to_2018Q4.csv
# 1.6 GB download; the mirror supports resuming with `curl -C -` if it drops

python scripts/train_real.py     # ~3 minutes on a 12-core / 16 GB laptop
streamlit run app.py
```

### Synthetic demo (no download required)

```bash
python scripts/train.py
streamlit run app.py             # falls back to the synthetic model automatically
```

Run the test suite with `pytest` (the 8 real-data tests skip automatically if the real CSV isn't
present).

### Docker

```bash
docker build -t lending-club-credit-risk .
docker run -p 8501:8501 lending-club-credit-risk
```

The 1.6 GB real CSV is never part of the Docker build context. The image reuses a committed artifact
if present (real preferred, synthetic fallback), otherwise trains the synthetic demo model at build
time.

## Data

The real LendingClub 2007–2018Q4 accepted-loans dataset (151 raw columns, ~2.26M loans) is publicly
available; this project sources it from an unauthenticated mirror
([bigblue.depaul.edu](https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/)) rather than
the Kaggle API, since no Kaggle credentials were available in the build environment. It is **not**
vendored into this repository (1.6 GB) — download it yourself with the command above.

Only loans with a **resolved** outcome (`loan_status` in `{"Fully Paid", "Charged Off"}`) are used —
1,345,310 of the ~2.26M total. Every other status (`Current`, `Late`, `In Grace Period`, `Default`,
`Issued`) represents a loan whose eventual outcome isn't known yet, and including it would put an
undetermined label in the training data.

### The leakage audit

39 columns are dropped because they only exist due to events that happen *after* a loan is issued —
i.e. they leak the outcome. Grouped and justified in [`src/lending_club/real_data.py`](src/lending_club/real_data.py):

- **17 payment/servicing fields** — `total_pymnt`, `out_prncp`, `recoveries`, `last_fico_range_high`, etc.
- **15 hardship-program fields** — a hardship plan only exists because the borrower was already struggling.
- **7 debt-settlement fields** — a settlement only exists on a loan that already defaulted.

A further 29 columns are dropped for unrelated reasons (identifiers, free text, near-constant
columns, joint-application fields that are >99% missing, numeric fields that are 64–82% missing) —
not leakage, just not useful features.

### Memory note

A single-shot `pd.read_csv` of the full 1.6 GB / 151-column file raised a real
`pandas.errors.ParserError: ... out of memory` during development. `real_data.load_real_loans`
reads the file in 150K-row chunks, filtering each chunk to resolved outcomes immediately, so peak
memory scales with chunk size rather than file size.

To swap in your own data instead, point `config.REAL_RAW_CSV_PATH` at your CSV — every downstream
module only depends on the column contract documented in `real_data.py`.

## Methodology notes

- **Why exclude grade/sub_grade/int_rate from the deployment model, not just post-issuance fields?**
  They're LendingClub's own risk-pricing output, not a raw application fact — a genuinely pre-decision
  system wouldn't have them yet. `installment` is excluded too, since it mathematically reconstructs
  `int_rate` given `loan_amnt`/`term`, silently reintroducing exactly what excluding `int_rate` was
  meant to avoid.
- **Why calibrate?** Raw gradient-boosted tree scores are not well-calibrated probabilities. Isotonic
  regression fit on a held-out validation split (distinct from train and test) corrects this without
  leaking test information.
- **Why a profit-optimal threshold instead of F1/accuracy?** A lender's actual objective is
  asymmetric: declining a loan that would have defaulted saves the full loss-given-default; declining
  one that would have paid forfeits the profit margin, not a symmetric "error." On this run, the
  F1-optimal (0.2175) and profit-optimal (0.230) thresholds happen to land close together and produce
  nearly identical test profit ($140.43M vs. $140.69M) — the real, larger effect is threshold
  optimization *at all* versus an approve-everyone baseline ($32.3M), a ~4.4× difference.
- **Why PR-AUC over ROC-AUC as the headline metric?** Default is a minority class (19.96% of resolved
  loans); PR-AUC is far more sensitive to positive-class performance than ROC-AUC, which can look
  deceptively good under class imbalance.

## Model card

See [`reports/model_report_real.md`](reports/model_report_real.md) for the full real-data model card,
and [`reports/model_report.md`](reports/model_report.md) for the synthetic-demo one.

## License

MIT — see [LICENSE](LICENSE).
