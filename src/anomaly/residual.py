"""
residual.py — Compute prediction residuals for anomaly detection.

Residuals are the foundation of the residual-based Isolation Forest pipeline
(literature: A-12, N-2). Training on residuals rather than raw values removes
the influence of the natural EC level, so the detector focuses on
*unexpected deviations from what the model predicted* rather than on
absolute magnitude.

    residual(t) = EC_actual(t) - EC_predicted(t)

Under normal conditions residuals are small and centred near zero.
A large residual (positive or negative) signals that reality diverged
from the model's expectation — the candidate signature of an anomaly.
"""

import numpy as np
import pandas as pd

from src.models.base import ParameterModel


def compute_residuals(
    model: ParameterModel,
    X: pd.DataFrame,
    y_actual: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Return the prediction residuals for every row in X.

    Parameters
    ----------
    model    : a fitted ParameterModel (must have _is_fitted=True).
    X        : feature matrix — same columns as used at training time.
    y_actual : observed target values aligned with X.

    Returns
    -------
    residuals : np.ndarray of shape (n,), dtype float64.
                residual[i] = y_actual[i] - y_predicted[i]
    """
    model.check_fitted()
    y_pred = model.predict(X)
    y_true = np.asarray(y_actual, dtype=float)
    return y_true - y_pred
