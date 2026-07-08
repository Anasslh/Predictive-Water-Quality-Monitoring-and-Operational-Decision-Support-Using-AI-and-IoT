"""
train.py — EC (Electrical Conductivity) prediction model.

Baseline implementation using RandomForest. Replace or extend with other
algorithms (XGBoost, SVR, ANN) for the multi-model comparison.
Any algorithm works as long as fit/predict/explain respect the ParameterModel contract.
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from src.models.base import ParameterModel


class ECModel(ParameterModel):
    def __init__(self):
        super().__init__(parameter_name="EC", model_version="rf_ec_v0")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame):
        self.check_fitted()
        return self.model.predict(X)

    def explain(self, X: pd.DataFrame):
        # For tree-based models, return the model directly:
        # shap_wrapper.py will apply shap.TreeExplainer on it.
        self.check_fitted()
        return self.model
