"""
shap_wrapper.py — Standardized SHAP explanation for any ParameterModel.

Takes any already-fitted ParameterModel (see src/models/base.py) and returns
the top-k most influential features as a plain list of dicts.

Works with tree-based models (RF, XGBoost) via shap.TreeExplainer.
For SVR or neural networks, pass explainer_type="kernel".
"""

from typing import List
import numpy as np
import pandas as pd
import shap

from src.models.base import ParameterModel


def compute_shap_explanation(
    model: ParameterModel,
    X_row: pd.DataFrame,
    top_k: int = 3,
    explainer_type: str = "tree",
) -> List[dict]:
    """
    Compute SHAP values for ONE row and return the top_k most influential
    features as a list of dicts: [{feature, shap_value, direction}, ...].

    model.explain(X_row) must return the underlying fitted model object
    (e.g. self.model — a RandomForestRegressor or XGBRegressor).
    """
    model.check_fitted()
    underlying_model = model.explain(X_row)

    if explainer_type == "tree":
        explainer = shap.TreeExplainer(underlying_model)
    elif explainer_type == "kernel":
        explainer = shap.KernelExplainer(underlying_model.predict, X_row)
    else:
        raise ValueError(f"Unknown explainer_type: {explainer_type}")

    shap_values = explainer.shap_values(X_row)
    values = np.array(shap_values)[0] if np.ndim(shap_values) == 2 else np.array(shap_values)

    feature_names = X_row.columns.tolist()
    pairs = sorted(zip(feature_names, values), key=lambda p: abs(p[1]), reverse=True)[:top_k]

    return [
        {
            "feature": name,
            "shap_value": round(float(val), 4),
            "direction": "positive" if val >= 0 else "negative",
        }
        for name, val in pairs
    ]
