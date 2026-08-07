"""
shap_wrapper.py — Standardized SHAP explanation for any ParameterModel.

Takes any already-fitted ParameterModel (see src/models/base.py) and returns
the top-k most influential features as a plain list of dicts.

Explainer selection
-------------------
By default (explainer_type="auto") the wrapper inspects the object returned
by model.explain() and picks the fastest exact explainer available:

  - XGBRegressor / XGBClassifier → shap.TreeExplainer   (exact, ~ms)
  - RandomForest / GradientBoosting / DecisionTree
                                 → shap.TreeExplainer   (exact, ~ms)
  - Any other object (SVR, ANN, …)
                                 → shap.KernelExplainer (approximate, ~s)

For the non-tree path, a ``background`` DataFrame must be provided so that
KernelExplainer has a reference distribution. A concise background (10–50
representative rows, e.g. shap.kmeans(X_train, 20)) balances accuracy and
speed. If ``background`` is None the wrapper falls back to the single
explanation row, which is valid but less accurate and triggers a warning.

Output contract
---------------
Always returns List[dict] with fields {feature, shap_value, direction}.
Callers (explain_anomaly, monitor pipeline, …) are not affected by which
explainer ran internally.
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np
import pandas as pd
import shap

from src.models.base import ParameterModel

logger = logging.getLogger(__name__)

# Tree-type detection — guarded imports so the module loads even without
# optional tree libraries installed.
_TREE_TYPES: list = []

try:
    from xgboost import XGBModel
    _TREE_TYPES.append(XGBModel)
except ImportError:
    pass

try:
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier,
    )
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    _TREE_TYPES.extend([
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier,
        DecisionTreeRegressor, DecisionTreeClassifier,
    ])
except ImportError:
    pass

_TREE_TYPES_TUPLE = tuple(_TREE_TYPES)


def _is_tree_based(obj: object) -> bool:
    """Return True if obj is a known tree-based estimator."""
    if not _TREE_TYPES_TUPLE:
        return False
    return isinstance(obj, _TREE_TYPES_TUPLE)


def compute_shap_explanation(
    model: ParameterModel,
    X_row: pd.DataFrame,
    top_k: int = 3,
    explainer_type: str = "auto",
    background: pd.DataFrame | None = None,
) -> List[dict]:
    """
    Compute SHAP values for one or more rows and return the top_k most
    influential features ranked by mean |SHAP| across the provided rows.

    Parameters
    ----------
    model          : Fitted ParameterModel. model.explain(X_row) must return
                     the underlying estimator (or a wrapper with .predict()).
    X_row          : Feature matrix to explain (typically 1 row from the
                     monitor pipeline; can be multiple rows for batch analysis).
    top_k          : Number of features to return.
    explainer_type : "auto"   — detect from model type (default).
                     "tree"   — force shap.TreeExplainer (tree models only).
                     "kernel" — force shap.KernelExplainer (any model).
    background     : Reference dataset for KernelExplainer. Required when
                     explainer_type="kernel" or when "auto" selects the kernel
                     path. Use shap.kmeans(X_train, k) for a compact summary.
                     If None, falls back to X_row itself (less accurate).

    Returns
    -------
    List[dict] with keys {feature, shap_value, direction}.
    ``shap_value`` is the mean signed SHAP value across all rows in X_row.
    ``direction`` is "positive" if shap_value >= 0, else "negative".
    """
    model.check_fitted()
    underlying = model.explain(X_row)

    # ── Explainer selection ────────────────────────────────────────────────────
    if explainer_type == "auto":
        use_tree = _is_tree_based(underlying)
        resolved = "tree" if use_tree else "kernel"
        logger.debug(
            "shap auto-select: %s → %s",
            type(underlying).__name__, resolved,
        )
    elif explainer_type in ("tree", "kernel"):
        resolved = explainer_type
    else:
        raise ValueError(
            f"explainer_type must be 'auto', 'tree', or 'kernel'; got {explainer_type!r}"
        )

    # ── Build explainer ────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    if resolved == "tree":
        explainer = shap.TreeExplainer(underlying)
        shap_values = explainer.shap_values(X_row)

    else:  # kernel
        if background is None:
            import warnings as _warnings
            _warnings.warn(
                "KernelExplainer: no background dataset provided — "
                "using X_row itself as reference (less accurate). "
                "Pass background=shap.kmeans(X_train, 20) for better results.",
                UserWarning,
                stacklevel=2,
            )
            bg = X_row
        else:
            bg = background

        # Wrap predict in a plain lambda so shap's internal model-conversion
        # doesn't try to introspect XGBRegressor attributes (feature_names_in_
        # has no setter on XGBoost, causing an AttributeError).
        predict_fn = (
            (lambda X: underlying.predict(X))
            if hasattr(underlying, "predict")
            else underlying
        )
        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_values = explainer.shap_values(X_row, silent=True)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.debug(
        "shap_wrapper: explainer=%s  rows=%d  elapsed=%.1f ms",
        resolved, len(X_row), elapsed_ms,
    )

    # ── Aggregate and rank ─────────────────────────────────────────────────────
    arr = np.array(shap_values)
    if arr.ndim == 1:
        # single row → shape (n_features,)
        mean_values = arr
    else:
        # multiple rows → shape (n_rows, n_features); take mean across rows
        mean_values = arr.mean(axis=0)

    feature_names = X_row.columns.tolist()
    pairs = sorted(
        zip(feature_names, mean_values),
        key=lambda p: abs(p[1]),
        reverse=True,
    )[:top_k]

    return [
        {
            "feature":     name,
            "shap_value":  round(float(val), 4),
            "direction":   "positive" if val >= 0 else "negative",
            "explainer":   resolved,
        }
        for name, val in pairs
    ]
