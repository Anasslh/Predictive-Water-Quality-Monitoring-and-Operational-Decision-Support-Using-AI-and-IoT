"""
recursive_forecaster.py — Generic multi-step recursive forecaster.

RECURSIVE STRATEGY
------------------
At each step the model predicts the next value. That prediction is then
appended to the historical window (in place of the unknown true value) and
used to compute features for the following step. This repeats n_steps times.

Consequence: prediction errors compound. Step-k error ≥ step-(k-1) error.
This is a known trade-off:
  - Advantage: requires only the trained one-step model — no retraining.
  - Disadvantage: error accumulates with horizon depth.
Use this for approximate trend surveillance (night / weekend monitoring),
NOT for precise point estimates at distant horizons.

FEATURE FUNCTION CONTRACT
--------------------------
feature_fn must be the SAME callable used to train the model:
    (df: pd.DataFrame) -> (X: pd.DataFrame, y: pd.Series)

The forecaster uses an "append-and-extract" pattern:
  1. Append a synthetic next row to the historical window (target = placeholder).
  2. Run feature_fn on the extended window.
  3. Extract the LAST row of X — this is the feature vector for the next step.
  4. The placeholder target value does not affect the features of the synthetic
     row (because build_features computes features via shift(), which uses
     PREVIOUS rows' values, not the current row's target).

This approach is model-agnostic: any feature_fn that follows the contract
works, regardless of which columns it produces.

CO-VARIABLE HANDLING
--------------------
Co-variables (pH, Turbidity when predicting EC) are NOT forecasted recursively.
Their lag-1 values at each future step are held constant at the LAST KNOWN
value from the historical window. This is a simplifying assumption: in a real
operational setting, co-variable forecasts could be injected here if available.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd

from src.models.base import ParameterModel


@dataclass
class ForecastResult:
    """
    Output of MultiStepForecaster.forecast().

    Attributes
    ----------
    predictions    : Predicted values at steps 1, 2, …, n_steps.
    uncertainty_std: Empirical std of residuals at each horizon step.
                     Computed across multiple historical starting points.
                     None at a given index if not enough data to estimate.
    target         : Name of the predicted variable (e.g. "EC").
    frequency      : Measurement interval used to generate the forecast.
    n_steps        : Number of steps forecast.
    """
    predictions:     list[float]
    uncertainty_std: list[float | None]
    target:          str
    frequency:       timedelta
    n_steps:         int


class MultiStepForecaster:
    """
    Recursive multi-step forecaster wrapping any fitted ParameterModel.

    Parameters
    ----------
    model      : Fitted ParameterModel (any subclass: XGBoost, RF, SVR, …).
    target     : Name of the target column (must match model training target).
    feature_fn : Feature engineering callable — same function used at training.
                 Signature: (df: pd.DataFrame) -> (X: pd.DataFrame, y: pd.Series).
    frequency  : Measurement interval (timedelta from detect_frequency()).
    residual_std_by_step : Pre-computed std of residuals per forecast step.
                 Index 0 = step-1 uncertainty, index 1 = step-2, etc.
                 If None, uncertainty bands cannot be drawn.
    """

    def __init__(
        self,
        model:        ParameterModel,
        target:       str,
        feature_fn:   Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
        frequency:    timedelta,
        residual_std_by_step: list[float] | None = None,
    ) -> None:
        self.model                 = model
        self.target                = target
        self.feature_fn            = feature_fn
        self.frequency             = frequency
        self.residual_std_by_step  = residual_std_by_step or []

    # ── Main API ───────────────────────────────────────────────────────────────

    def forecast(
        self,
        historical_window: pd.DataFrame,
        n_steps: int,
    ) -> ForecastResult:
        """
        Produce a recursive n-step ahead forecast.

        Parameters
        ----------
        historical_window : pd.DataFrame
            Raw data up to (and including) the last known timestep.
            Must contain 'Date', the target column, and all co-variable columns.
            Must have at least max_lag + max_rolling_window rows of valid history
            (typically 7 rows for the default EC feature set).
        n_steps : int
            Number of future steps to predict.

        Returns
        -------
        ForecastResult
        """
        window = historical_window.copy().sort_values("Date").reset_index(drop=True)
        predictions = []
        uncertainty = []

        for step_idx in range(n_steps):
            # Build features for the next step without printing verbose output
            X_next = self._get_next_step_features(window)
            pred   = float(self.model.predict(X_next)[0])
            predictions.append(pred)

            std = (
                self.residual_std_by_step[step_idx]
                if step_idx < len(self.residual_std_by_step)
                else None
            )
            uncertainty.append(std)

            # Extend window: copy last row, advance Date, set target to prediction.
            # Co-variable columns keep their last known value (see module docstring).
            new_row           = window.iloc[-1].copy()
            new_row["Date"]   = new_row["Date"] + self.frequency
            new_row[self.target] = pred
            window = pd.concat(
                [window, pd.DataFrame([new_row])], ignore_index=True
            )

        return ForecastResult(
            predictions     = predictions,
            uncertainty_std = uncertainty,
            target          = self.target,
            frequency       = self.frequency,
            n_steps         = n_steps,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_next_step_features(self, window: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the feature vector for the step AFTER the last row in `window`.

        Appends a synthetic row (target = 0.0 placeholder, Date = last + frequency)
        and runs feature_fn on the extended window. Extracts the last row of X.

        Because feature_fn computes features via shift() (using PREVIOUS rows'
        values), the synthetic row's placeholder target does NOT affect its own
        feature computation — only the co-variable columns of the synthetic row
        could (those come from the last row of window, already correct).
        """
        synthetic      = window.iloc[-1].copy()
        synthetic["Date"]       = synthetic["Date"] + self.frequency
        synthetic[self.target]  = 0.0   # placeholder — shifted, so not used

        extended = pd.concat(
            [window, pd.DataFrame([synthetic])], ignore_index=True
        )

        # Suppress the verbose print from build_features during recursive calls
        with contextlib.redirect_stdout(io.StringIO()):
            X, _ = self.feature_fn(extended)

        return X.iloc[[-1]].reset_index(drop=True)


# ── Persistence helpers ────────────────────────────────────────────────────────

_FORECASTER_CONFIG_FILENAME = "forecaster_config.json"


def save_forecaster_config(
    target: str,
    frequency: timedelta,
    lag_hours: list[int],
    rolling_hours: list[int],
    co_variables: list[str],
    residual_std_by_step: list[float],
    models_store_path: "str | Path",
    n_steps: int = 7,
) -> "Path":
    """
    Persist the forecaster configuration to <models_store_path>/forecaster_config.json.

    The model itself is not stored here (it lives in current_model.json).
    This file only stores the feature engineering parameters and uncertainty bands
    needed to reconstruct a MultiStepForecaster at load time.
    """
    import json
    from pathlib import Path as _Path

    out = _Path(models_store_path) / _FORECASTER_CONFIG_FILENAME
    cfg = {
        "target":                target,
        "frequency_seconds":     int(frequency.total_seconds()),
        "lag_hours":             lag_hours,
        "rolling_hours":         rolling_hours,
        "co_variables":          co_variables,
        "residual_std_by_step":  [round(s, 6) for s in residual_std_by_step],
        "n_steps":               n_steps,
    }
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def load_forecaster(
    models_store_path: "str | Path",
    prediction_model: "ParameterModel",
) -> "MultiStepForecaster | None":
    """
    Reconstruct a MultiStepForecaster from the saved config and a fitted model.

    Returns None if forecaster_config.json does not exist (forecaster not yet
    trained for this parameter — caller should log and skip forecasting).
    """
    import json
    from pathlib import Path as _Path
    from src.forecasting.feature_engineering_generic import build_features_time_aware

    p = _Path(models_store_path) / _FORECASTER_CONFIG_FILENAME
    if not p.exists():
        return None

    raw           = json.loads(p.read_text(encoding="utf-8"))
    target        = raw["target"]
    frequency     = timedelta(seconds=raw["frequency_seconds"])
    lag_hours     = raw["lag_hours"]
    rolling_hours = raw["rolling_hours"]
    co_variables  = raw.get("co_variables", [])
    residual_std  = raw.get("residual_std_by_step", [])

    def _feature_fn(df: pd.DataFrame) -> "tuple[pd.DataFrame, pd.Series]":
        return build_features_time_aware(
            df,
            target        = target,
            frequency     = frequency,
            lag_hours     = lag_hours,
            rolling_hours = rolling_hours,
            co_variables  = co_variables if co_variables else None,
        )

    forecaster = MultiStepForecaster(
        model                = prediction_model,
        target               = target,
        feature_fn           = _feature_fn,
        frequency            = frequency,
        residual_std_by_step = residual_std,
    )
    # Expose the saved n_steps so callers can pass it as forecast_steps
    forecaster._config_n_steps: int = raw.get("n_steps", 7)
    return forecaster
