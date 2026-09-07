"""lending_club: end-to-end credit risk modeling package.

Modules
-------
config     : paths, constants, and reproducibility settings.
data       : synthetic LendingClub-style dataset generation / loading.
features   : feature engineering for the "research" (full) and
             "deployment" (application-time only) feature sets.
metrics    : evaluation metrics, expected-loss cost model, and
             profit-optimized threshold search.
modeling   : training, calibration, and artifact assembly.
explain    : SHAP-based per-prediction explanations.
"""

__version__ = "1.0.0"
