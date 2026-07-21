"""
explain.py — Coupled anomaly score + SHAP explanation for EC anomalies.

Original contribution (not present in A-12, N-1, N-2, S-3):
  For each detected anomaly we return both the Isolation Forest score AND
  the SHAP explanation of the underlying prediction, answering two questions:
    (1) *How anomalous is this point?*  → anomaly_score
    (2) *Why did the model NOT predict this value?* → shap_top_features
        (i.e. which features drove the prediction away from the actual value)

Output format — AnomalyExplanation dataclass — extends the Prediction
dataclass from src/models/base.py without breaking its contract.

Typical usage
-------------
    result = explain_anomaly(
        model       = fitted_ec_model,       # ECModelXGBoost
        detector    = fitted_ec_detector,    # ECAnomalyDetector
        X_row       = X_test.iloc[[i]],      # single-row DataFrame
        y_actual    = float(y_test.iloc[i]),
        timestamp   = "2026-01-10",
        top_k       = 3,
    )
    # result.is_anomaly, result.anomaly_score, result.shap_top_features
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.models.base import ParameterModel
from src.anomaly.residual import compute_residuals
from src.anomaly.detector import ECAnomalyDetector
from src.xai.shap_wrapper import compute_shap_explanation


@dataclass
class AnomalyExplanation:
    """
    Unified output for a single timestep: prediction + anomaly + explanation.

    Fields mirror the Prediction dataclass (src/models/base.py) and extend it
    with anomaly detection and SHAP attribution.
    """
    # --- Identification ---
    parameter: str
    timestamp: str
    model_version: str

    # --- Prediction ---
    predicted_value: float
    actual_value: float
    residual: float                        # actual - predicted

    # --- Anomaly detection ---
    anomaly_score: float                   # [0, 1], 1 = highly anomalous
    is_anomaly: bool                       # anomaly_score >= detector.threshold

    # --- SHAP (always computed, even for non-anomalous points) ---
    shap_top_features: List[dict] = field(default_factory=list)
    # Format: [{"feature": str, "shap_value": float, "direction": str}, ...]


def explain_anomaly(
    model: ParameterModel,
    detector: ECAnomalyDetector,
    X_row: pd.DataFrame,
    y_actual: float,
    timestamp: str,
    top_k: int = 3,
) -> AnomalyExplanation:
    """
    Compute a full AnomalyExplanation for a single observation.

    Parameters
    ----------
    model     : fitted ParameterModel (e.g. ECModelXGBoost).
    detector  : fitted ECAnomalyDetector.
    X_row     : single-row DataFrame with the 9 EC features.
    y_actual  : the observed EC value for this timestep.
    timestamp : date string (ISO format recommended).
    top_k     : number of SHAP features to return.

    Returns
    -------
    AnomalyExplanation with all fields populated.
    """
    # Prediction
    y_pred    = float(model.predict(X_row)[0])
    residual  = y_actual - y_pred

    # Anomaly score
    residual_arr  = np.array([residual])
    anomaly_score = float(detector.score(residual_arr)[0])
    is_anomaly    = bool(detector.predict(residual_arr)[0])

    # SHAP explanation of the prediction
    shap_feats = compute_shap_explanation(
        model, X_row, top_k=top_k, explainer_type="tree"
    )

    return AnomalyExplanation(
        parameter       = model.parameter_name,
        timestamp       = timestamp,
        model_version   = model.model_version,
        predicted_value = round(y_pred, 2),
        actual_value    = round(y_actual, 2),
        residual        = round(residual, 2),
        anomaly_score   = round(anomaly_score, 4),
        is_anomaly      = is_anomaly,
        shap_top_features = shap_feats,
    )


def batch_explain(
    model: ParameterModel,
    detector: ECAnomalyDetector,
    X: pd.DataFrame,
    y_actual: pd.Series | np.ndarray,
    timestamps: list[str],
    top_k: int = 3,
    anomalies_only: bool = False,
) -> List[AnomalyExplanation]:
    """
    Run explain_anomaly() over an entire dataset and return a list of results.

    Parameters
    ----------
    anomalies_only : if True, return only rows where is_anomaly=True.
                     Useful for production alerting where only flagged rows
                     need a full SHAP explanation.
    """
    results = []
    y_arr = np.asarray(y_actual, dtype=float)

    for i, ts in enumerate(timestamps):
        result = explain_anomaly(
            model     = model,
            detector  = detector,
            X_row     = X.iloc[[i]],
            y_actual  = float(y_arr[i]),
            timestamp = ts,
            top_k     = top_k,
        )
        if not anomalies_only or result.is_anomaly:
            results.append(result)

    return results
