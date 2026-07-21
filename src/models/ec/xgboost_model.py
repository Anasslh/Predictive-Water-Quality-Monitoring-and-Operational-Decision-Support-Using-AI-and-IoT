"""
xgboost_model.py — EC prediction model using XGBoost (final production version).

Hyperparameters are frozen from the tuning pipeline (grid search + Bayesian
optimisation, selected on val set, confirmed on test):
  max_depth=3, n_estimators=100, learning_rate=0.01

explain() returns the fitted XGBRegressor so that shap_wrapper.py can apply
shap.TreeExplainer on it (explainer_type="tree").
"""

import pandas as pd
from xgboost import XGBRegressor

from src.models.base import ParameterModel


class ECModelXGBoost(ParameterModel):
    def __init__(self):
        super().__init__(parameter_name="EC", model_version="xgb_ec_v1_final")
        # Hyperparameters frozen — do not change without re-running the full
        # tuning pipeline in src/evaluation/ and a new test-set evaluation.
        self.model = XGBRegressor(
            max_depth=3,
            n_estimators=100,
            learning_rate=0.01,
            random_state=42,
            verbosity=0,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame):
        self.check_fitted()
        return self.model.predict(X)

    def explain(self, X: pd.DataFrame):
        # XGBoost is tree-based: return the model directly.
        # shap_wrapper.py will apply shap.TreeExplainer on it (explainer_type="tree").
        self.check_fitted()
        return self.model
