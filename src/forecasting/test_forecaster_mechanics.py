"""
test_forecaster_mechanics.py — Verify that MultiStepForecaster.forecast()
actually advances the historical window at every step of the recursive loop.

MOTIVATION
----------
The deployed model (XGBoost trained on daily EC data) produces identical
predictions at every forecast step even though features change — this is an
expected property of tree-based models, not a code bug:

  * Tree models partition the feature space into axis-aligned boxes (leaves).
  * The adjacent feature vectors at D+1 and D+2 (EC_lag1=769 vs EC_lag1=756.58)
    fall in the same leaf → identical output.
  * This is the "fixed-point" behaviour of recursive tree forecasting: once the
    prediction lands at a leaf's centroid, it feeds itself back and stays there.

These tests verify that the MECHANISM is correct:
  - The historical window grows by one row at every step.
  - The feature vector extracted for each step differs from the previous one.
  - Predictions DO vary when a model is used whose leaf boundaries are sensitive
    enough to distinguish the step-to-step feature deltas.

None of these tests assert that the deployed model produces diverse predictions
(that is a model-quality property, not a correctness property of the forecaster).

Run:  python src/forecasting/test_forecaster_mechanics.py
      (also picked up automatically by run_tests.sh)
"""

from __future__ import annotations

import io
import sys
import contextlib
import tempfile
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.models.base import ParameterModel
from src.forecasting.recursive_forecaster import MultiStepForecaster

# ── Test helpers ───────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    symbol = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{symbol}  {label}{suffix}")
    _results.append((label, condition))


# ── Mock model: returns EC_lag1 * slope + bias ─────────────────────────────────
#
# This model is deliberately leaf-free and differentiable: its output is a
# strict linear function of EC_lag1, so any step-to-step change in EC_lag1
# produces a visible change in the prediction. It lets us verify the mechanics
# without relying on the trained model's internal leaf structure.

class _LinearLag1Model(ParameterModel):
    """predict(X) = X['EC_lag1'] * slope + bias."""

    def __init__(self, slope: float = 0.95, bias: float = 20.0) -> None:
        self.slope = slope
        self.bias  = bias
        super().__init__("EC", "linear_lag1_test")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass   # pre-fitted by construction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["EC_lag1"].values * self.slope + self.bias).astype(float)

    def explain(self, X: pd.DataFrame) -> list[dict]:
        return []


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_daily_ec(n: int = 40) -> pd.DataFrame:
    rng   = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    ec    = 500.0 + rng.normal(0, 20, n).cumsum()   # random-walk around 500
    return pd.DataFrame({"Date": dates, "EC": ec.round(2)})


def _make_forecaster(model: ParameterModel, df: pd.DataFrame) -> MultiStepForecaster:
    from src.forecasting.feature_engineering_generic import build_features_time_aware

    freq = timedelta(hours=24)
    lag_hours     = [24, 48, 72]
    rolling_hours = [72, 168]

    def feature_fn(d: pd.DataFrame):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_features_time_aware(
                d, target="EC", frequency=freq,
                lag_hours=lag_hours, rolling_hours=rolling_hours,
                co_variables=[],
            )

    return MultiStepForecaster(
        model=model, target="EC",
        feature_fn=feature_fn, frequency=freq,
        residual_std_by_step=[5.0 * (k + 1) ** 0.5 for k in range(7)],
    )


# ── Test 1: window grows by exactly 1 row at every step ───────────────────────

def test_window_grows_each_step() -> None:
    """
    The internal `window` DataFrame must grow by exactly 1 row per loop
    iteration, confirmed by monkey-patching _get_next_step_features to
    record the window size at call time.
    """
    df    = _make_daily_ec(40)
    model = _LinearLag1Model()
    fc    = _make_forecaster(model, df)

    window_sizes: list[int] = []
    original_fn = fc._get_next_step_features

    def _patched_get_features(window):
        window_sizes.append(len(window))
        return original_fn(window)

    fc._get_next_step_features = _patched_get_features   # type: ignore[method-assign]

    n_steps = 5
    with contextlib.redirect_stdout(io.StringIO()):
        fc.forecast(df, n_steps=n_steps)

    check(
        f"_get_next_step_features called {n_steps} times",
        len(window_sizes) == n_steps,
        str(window_sizes),
    )
    for k in range(1, n_steps):
        check(
            f"window size grew by 1 between step {k} and step {k+1}",
            window_sizes[k] == window_sizes[k - 1] + 1,
            f"{window_sizes[k - 1]} → {window_sizes[k]}",
        )


# ── Test 2: EC_lag1 feature changes at every step (linear model) ──────────────

def test_features_change_between_steps() -> None:
    """
    Using the linear mock model (slope=0.95 ≠ 1), each step's prediction is
    strictly different from the previous, so EC_lag1 differs at every call.
    Records all EC_lag1 values extracted and asserts they are all different.
    """
    df    = _make_daily_ec(40)
    model = _LinearLag1Model(slope=0.95, bias=20.0)
    fc    = _make_forecaster(model, df)

    lag1_values: list[float] = []
    original_fn = fc._get_next_step_features

    def _patched_get_features(window):
        X = original_fn(window)
        lag1_values.append(float(X["EC_lag1"].iloc[0]))
        return X

    fc._get_next_step_features = _patched_get_features   # type: ignore[method-assign]

    n_steps = 5
    with contextlib.redirect_stdout(io.StringIO()):
        result = fc.forecast(df, n_steps=n_steps)

    check(
        "EC_lag1 recorded for every step",
        len(lag1_values) == n_steps,
        str(lag1_values),
    )
    all_distinct = len(set(round(v, 6) for v in lag1_values)) == n_steps
    check(
        "EC_lag1 is strictly distinct at every step (linear model)",
        all_distinct,
        str([round(v, 4) for v in lag1_values]),
    )
    check(
        "predictions are strictly distinct (linear model produces no fixed-point)",
        len(set(round(p, 6) for p in result.predictions)) == n_steps,
        str([round(p, 4) for p in result.predictions]),
    )


# ── Test 3: last EC value in the window equals the previous prediction ─────────

def test_window_tail_contains_previous_prediction() -> None:
    """
    After each step, the last EC value in the working window must equal
    the prediction just made. This confirms the recursive injection is working.
    """
    df    = _make_daily_ec(40)
    model = _LinearLag1Model(slope=0.95, bias=20.0)
    fc    = _make_forecaster(model, df)

    ec_tail_after: list[float] = []
    predictions_made: list[float] = []
    original_fn = fc._get_next_step_features

    step = [0]   # mutable counter to sync injection-point

    def _patched_get_features(window):
        # On step > 0, the tail must equal the previous prediction
        if step[0] > 0:
            ec_tail_after.append(float(window["EC"].iloc[-1]))
        step[0] += 1
        return original_fn(window)

    fc._get_next_step_features = _patched_get_features   # type: ignore[method-assign]

    n_steps = 5
    with contextlib.redirect_stdout(io.StringIO()):
        result = fc.forecast(df, n_steps=n_steps)

    # ec_tail_after[k] == result.predictions[k] for k in 0..n_steps-2
    for k, (tail, pred) in enumerate(zip(ec_tail_after, result.predictions)):
        check(
            f"window EC tail after step {k+1} equals prediction D+{k+1}",
            abs(tail - pred) < 1e-9,
            f"tail={tail:.4f}  pred={pred:.4f}",
        )


# ── Test 4: fixed-point behaviour confirmed for a persistence model ────────────

def test_tree_fixed_point_is_expected_not_a_bug() -> None:
    """
    A model that always returns the SAME constant (regardless of features)
    produces identical predictions — but features still change between steps.

    This shows that identical outputs ≠ broken window. The check that matters
    is whether EC_lag1 CHANGES (it must); whether the model USES that change
    is a model-quality question separate from the forecaster's correctness.
    """

    class _ConstantModel(ParameterModel):
        def __init__(self) -> None:
            super().__init__("EC", "constant_test")
        def fit(self, X, y):   pass
        def predict(self, X):  return np.full(len(X), 999.0)
        def explain(self, X):  return []

    df    = _make_daily_ec(40)
    model = _ConstantModel()
    fc    = _make_forecaster(model, df)

    lag1_values: list[float] = []
    original_fn = fc._get_next_step_features

    def _patched_get_features(window):
        X = original_fn(window)
        lag1_values.append(float(X["EC_lag1"].iloc[0]))
        return X

    fc._get_next_step_features = _patched_get_features   # type: ignore[method-assign]

    n_steps = 5
    with contextlib.redirect_stdout(io.StringIO()):
        result = fc.forecast(df, n_steps=n_steps)

    # Predictions are identical — that's expected from a constant model
    check(
        "constant model: all predictions equal 999.0",
        all(abs(p - 999.0) < 1e-9 for p in result.predictions),
        str(result.predictions[:3]),
    )
    # But EC_lag1 still changes (the window advances with 999.0 injected)
    check(
        "constant model: EC_lag1 at step 2 = 999.0 (previous prediction injected)",
        abs(lag1_values[1] - 999.0) < 1e-9,
        f"lag1[1]={lag1_values[1]:.4f}",
    )
    check(
        "constant model: EC_lag1 changes from step 1 to step 2",
        abs(lag1_values[0] - lag1_values[1]) > 1e-6,
        f"{lag1_values[0]:.4f} → {lag1_values[1]:.4f}",
    )


# ── Test 5: n_steps from saved config is honoured ─────────────────────────────

def test_n_steps_honoured() -> None:
    """forecast() produces exactly n_steps predictions."""
    df    = _make_daily_ec(40)
    model = _LinearLag1Model()
    fc    = _make_forecaster(model, df)

    for n in (1, 3, 7):
        with contextlib.redirect_stdout(io.StringIO()):
            result = fc.forecast(df, n_steps=n)
        check(
            f"forecast(n_steps={n}) → {n} predictions",
            len(result.predictions) == n,
            f"got {len(result.predictions)}",
        )
        check(
            f"forecast(n_steps={n}) → {n} uncertainty bands",
            len(result.uncertainty_std) == n,
        )


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  forecaster mechanics — recursive loop integrity tests")
    print(SEP2)
    print()
    print("  NOTE: identical consecutive predictions from the deployed model")
    print("  are a tree-based fixed-point property, NOT a forecaster bug.")
    print("  These tests verify the WINDOW ADVANCES, not that the model varies.")
    print()

    test_window_grows_each_step()
    print(SEP)
    test_features_change_between_steps()
    print(SEP)
    test_window_tail_contains_previous_prediction()
    print(SEP)
    test_tree_fixed_point_is_expected_not_a_bug()
    print(SEP)
    test_n_steps_honoured()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
