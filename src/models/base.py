"""
base.py — Shared contract for all per-parameter models.

Every parameter (pH, EC, turbidity) must have a class that inherits from
ParameterModel and implements fit / predict / explain, regardless of the
algorithm chosen internally (RF, XGBoost, SVR, ...).

This contract ensures that shap_wrapper.py works the same way across all
parameters, even if team members make different technical choices within their
own parameter scope.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class Prediction:
    """Standardized model output, before passing through the SHAP wrapper."""
    parameter: str
    timestamp: str
    predicted_value: float
    model_version: str


class ParameterModel(ABC):
    """
    Abstract base class to inherit for each parameter.

    Usage example (in src/models/ph/train.py):

        class PHModel(ParameterModel):
            def __init__(self):
                super().__init__(parameter_name="pH", model_version="rf_ph_v1")
                self.model = RandomForestRegressor(...)

            def fit(self, X_train, y_train):
                self.model.fit(X_train, y_train)

            def predict(self, X):
                return self.model.predict(X)

            def explain(self, X):
                # must return an object compatible with shap.Explainer
                return self.model  # shap_wrapper.py handles the rest
    """

    def __init__(self, parameter_name: str, model_version: str):
        self.parameter_name = parameter_name
        self.model_version = model_version
        self._is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model. Must set self._is_fitted = True when done."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted values for X."""
        ...

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> Any:
        """
        Return the object needed for SHAP computation (typically the underlying
        model itself, e.g. self.model). shap_wrapper.py converts this into a
        standardized explanation.
        """
        ...

    def check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"Model for {self.parameter_name} has not been trained yet (call fit() first)."
            )
