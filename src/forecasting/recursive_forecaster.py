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

    # ── Utility: compute per-step residual std ─────────────────────────────────

    def compute_residual_std_by_step(
        self,
        full_df: pd.DataFrame,
        starting_indices: list[int],
        n_steps: int,
    ) -> list[float]:
        """
        Estimate empirical uncertainty at each forecast step.

        For each starting index i (a row index within full_df), run a recursive
        n-step forecast using everything BEFORE row i as the historical window,
        and record the residual (true_value - predicted_value) at each step.

        Returns the per-step standard deviation of these residuals.

        Parameters
        ----------
        full_df          : Complete dataset (must include all rows needed as
                           both history and true values).
        starting_indices : List of row indices (within full_df) where each
                           forecast begins. The forecast at index i uses
                           full_df[:i] as history and full_df[i:i+n_steps]
                           as ground truth.
        n_steps          : Forecast horizon (number of steps).

        Returns
        -------
        list[float]  — length = n_steps, residual std at step 1, 2, …, n_steps.
        """
        residuals_by_step: list[list[float]] = [[] for _ in range(n_steps)]

        for i in starting_indices:
            if i + n_steps > len(full_df):
                continue

            history     = full_df.iloc[:i].copy()
            true_values = full_df[self.target].iloc[i: i + n_steps].values

            # Temporarily remove residual_std to avoid recursion in sub-calls
            saved = self.residual_std_by_step
            self.residual_std_by_step = []
            try:
                result = self.forecast(history, n_steps)
            finally:
                self.residual_std_by_step = saved

            for step, (true_val, pred_val) in enumerate(
                zip(true_values, result.predictions)
            ):
                residuals_by_step[step].append(float(true_val) - pred_val)

        return [
            float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
            for r in residuals_by_step
        ]

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
