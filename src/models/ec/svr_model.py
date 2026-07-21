"""
svr_model.py — EC prediction model using Support Vector Regression.

SVR is sensitive to feature scale, unlike tree-based models.
Both X and y are standardized (fit on train only) because SVR's
RBF kernel optimizes distances in a space where all dimensions
should be on a comparable scale — including the output.

X scaling: StandardScaler inside a Pipeline (fit in fit(), applied
  transparently in pipeline.predict()).
y scaling: separate StandardScaler (self.y_scaler) fit on y_train;
  predict() inverse-transforms back to the original EC unit (µS/cm).

explain() returns a wrapper whose .predict() accepts raw X and returns
raw y, so that shap_wrapper.py can build a KernelExplainer on it
(explainer_type="kernel").

Note: KernelExplainer is significantly slower than TreeExplainer.
For SHAP on large datasets, consider using a background summary
(e.g. shap.kmeans) instead of passing the full training set.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import ParameterModel


class _PredictWrapper:
    """Thin wrapper so explain() exposes .predict(raw_X) → raw_y for KernelExplainer."""
    def __init__(self, pipeline, y_scaler):
        self._pipeline = pipeline
        self._y_scaler = y_scaler

    def predict(self, X):
        y_scaled = self._pipeline.predict(X)
        return self._y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


class ECModelSVR(ParameterModel):
    def __init__(self):
        super().__init__(parameter_name="EC", model_version="svr_ec_v1")
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ])
        self.y_scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        y_scaled = self.y_scaler.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel()
        self.pipeline.fit(X_train, y_scaled)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame):
        self.check_fitted()
        y_scaled = self.pipeline.predict(X)
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def explain(self, X: pd.DataFrame):
        # Returns a wrapper whose .predict() goes raw_X → raw_y (EC in µS/cm).
        # shap_wrapper.py must use explainer_type="kernel".
        self.check_fitted()
        return _PredictWrapper(self.pipeline, self.y_scaler)
