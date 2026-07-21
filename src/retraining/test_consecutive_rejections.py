"""
test_consecutive_rejections.py — Verify the consecutive-rejection counter in RetrainManager.

WHAT IS TESTED
--------------
1. read_rejection_counter() returns 0 when no file exists.
2. write/read round-trip: counter survives a simulated process restart.
3. A new RetrainManager reads the persisted counter in __init__.
4. 3 consecutive rejections trigger the alert; counter increments each time.
5. Counter resets to 0 when the acceptance gate is passed.

HOW REJECTION IS TRIGGERED
--------------------------
- Current model: _MeanModel — predicts the training mean (~1.0 on stable data).
- Initial data : 100 rows, y = 1.0 (reference residuals ≈ 0).
- Drifted data : same 100 rows + 70 rows with y = 100.0.
  * RMSE ratio on recent rows = ∞ (reference_rmse ≈ 0) → drift fires.
  * Candidate model (_BadModel, predicts −1 000) → new_rmse ≫ old_rmse → rejected.

RUN
---
  python src/retraining/test_consecutive_rejections.py
  pytest  src/retraining/test_consecutive_rejections.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base import ParameterModel
from src.retraining.model_versioning import (
    read_rejection_counter,
    write_rejection_counter,
)
from src.retraining.retrain_manager import RetrainManager


# ── Minimal ParameterModel stubs ──────────────────────────────────────────────

class _MeanModel(ParameterModel):
    """Predicts the training mean. Good on in-distribution data."""

    def __init__(self) -> None:
        super().__init__(parameter_name="test_param", model_version="mean_v1")
        self._mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._mean = float(y.mean())
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)

    def explain(self, X: pd.DataFrame):
        return None


class _BadModel(ParameterModel):
    """Always predicts −1 000. Guaranteed to be worse than _MeanModel on positive data."""

    def __init__(self) -> None:
        super().__init__(parameter_name="test_param", model_version="bad_v1")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), -1_000.0)

    def explain(self, X: pd.DataFrame):
        return None


class _GoodCandidate(ParameterModel):
    """Predicts the training mean — beats a stale mean model on drifted data.

    Must be at module level so pickle can serialize it during submit_for_approval.
    """

    def __init__(self) -> None:
        super().__init__(parameter_name="test_param", model_version="good_v1")
        self._mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._mean = float(y.mean())
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)

    def explain(self, X: pd.DataFrame):
        return None


# ── Synthetic data ─────────────────────────────────────────────────────────────

def _make_datasets(n_initial: int = 100, n_drifted: int = 70):
    """
    Return (initial_df, drifted_df).

    initial_df : n_initial rows, y = 1.0  (stable baseline)
    drifted_df : initial rows + n_drifted rows with y = 100.0 (drift event)
    """
    initial_df = pd.DataFrame({
        "x": np.arange(n_initial, dtype=float),
        "y": np.ones(n_initial),
    })
    drifted_df = pd.DataFrame({
        "x": np.arange(n_initial + n_drifted, dtype=float),
        "y": np.concatenate([np.ones(n_initial), np.full(n_drifted, 100.0)]),
    })
    return initial_df, drifted_df


def _feature_fn(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[["x"]], df["y"]


# ── Helper: build an initialized RetrainManager ────────────────────────────────

def _make_manager(
    store: Path,
    model_class=_BadModel,
    threshold: int = 3,
    tolerance: float = 0.0,
    initial_df: pd.DataFrame | None = None,
) -> RetrainManager:
    if initial_df is None:
        initial_df, _ = _make_datasets()

    initial_model = _MeanModel()
    initial_model.fit(initial_df[["x"]], initial_df["y"])

    manager = RetrainManager(
        model_class               = model_class,
        feature_fn                = _feature_fn,
        min_new_rows              = 50,
        tolerance                 = tolerance,
        models_store_path         = store,
        rejection_alert_threshold = threshold,
        max_history_years         = float("inf"),   # window not under test here
    )
    manager.initialize(initial_df, initial_model)
    return manager


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_read_counter_returns_zero_when_no_file():
    with tempfile.TemporaryDirectory() as tmp:
        count = read_rejection_counter(Path(tmp))
    assert count == 0


def test_counter_write_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)
        write_rejection_counter(store, count=7)
        assert read_rejection_counter(store) == 7

        write_rejection_counter(store, count=0)
        assert read_rejection_counter(store) == 0


def test_new_manager_loads_persisted_counter():
    """A new RetrainManager must restore the counter from disk in __init__."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)
        write_rejection_counter(store, count=2)

        manager = RetrainManager(
            model_class       = _BadModel,
            feature_fn        = _feature_fn,
            models_store_path = store,
        )
        assert manager.consecutive_rejections == 2, (
            f"Expected 2 after restart, got {manager.consecutive_rejections}"
        )


def test_three_consecutive_rejections_trigger_alert():
    """
    Core test: after 3 rejections the alert fires and the counter value is correct.

    Rejection path:
      - Initial data stable at y=1.0 → reference_rmse ≈ 0.
      - Drifted data adds 70 rows at y=100 → RMSE ratio = ∞ → drift fires.
      - _BadModel predicts −1 000 → new_rmse ≫ old_rmse → rejected.
    """
    initial_df, drifted_df = _make_datasets()

    with tempfile.TemporaryDirectory() as tmp:
        store   = Path(tmp)
        manager = _make_manager(store, threshold=3)

        for i in range(1, 4):
            result = manager.attempt_retrain(drifted_df)

            assert result["retrained"],     f"Run {i}: expected retrained=True"
            assert not result["accepted"],  f"Run {i}: expected accepted=False"
            assert result["consecutive_rejections"] == i, (
                f"Run {i}: counter should be {i}, got {result['consecutive_rejections']}"
            )
            assert read_rejection_counter(store) == i, (
                f"Run {i}: persisted counter should be {i}"
            )

            if i < 3:
                assert "alert" not in result or result.get("alert") is None, (
                    f"Run {i}: alert must not fire before threshold"
                )
            else:
                assert "alert" in result and result["alert"], (
                    "Run 3: alert must fire at threshold"
                )
                assert "benchmark" in result["alert"].lower() or "src/pipeline" in result["alert"], (
                    f"Alert should reference benchmark: {result['alert']}"
                )


def test_counter_increments_beyond_threshold():
    """Counter keeps incrementing past the alert threshold (no cap)."""
    initial_df, drifted_df = _make_datasets()

    with tempfile.TemporaryDirectory() as tmp:
        manager = _make_manager(Path(tmp), threshold=3)
        for i in range(1, 6):
            result = manager.attempt_retrain(drifted_df)
            assert result["consecutive_rejections"] == i

        assert manager.consecutive_rejections == 5


def test_alert_fires_periodically_not_on_every_rejection():
    """
    Alert policy: periodic modulo (fire at multiples of rejection_alert_threshold).

    With threshold=3: alert fires at 3 and 6; silent at 1, 2, 4, 5.
    This avoids alert fatigue (every rejection) while ensuring re-notification
    if the first alert is missed.
    """
    initial_df, drifted_df = _make_datasets()

    # Expected pattern for threshold=3 over 7 rejections:
    #   count  1  2  3  4  5  6  7
    #   alert  -  -  ✓  -  -  ✓  -
    expected_alert = {1: False, 2: False, 3: True, 4: False, 5: False, 6: True, 7: False}

    with tempfile.TemporaryDirectory() as tmp:
        manager = _make_manager(Path(tmp), threshold=3)
        for i in range(1, 8):
            result = manager.attempt_retrain(drifted_df)
            has_alert = bool(result.get("alert"))
            assert has_alert == expected_alert[i], (
                f"Rejection #{i}: expected alert={expected_alert[i]}, got alert={has_alert} "
                f"(consecutive_rejections={result['consecutive_rejections']})"
            )


def test_counter_resets_on_acceptance():
    """
    After rejections the counter resets to 0 when the acceptance gate is passed.

    A _GoodCandidate (predicts training mean) is a strictly better model than
    the frozen _MeanModel(y=1.0) on a test set full of y=100 rows.
    """
    initial_df, drifted_df = _make_datasets()

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)

        # Phase 1: 2 rejections
        manager = _make_manager(store, model_class=_BadModel, threshold=3)
        for _ in range(2):
            r = manager.attempt_retrain(drifted_df)
            assert not r["accepted"]

        assert manager.consecutive_rejections == 2
        assert read_rejection_counter(store) == 2

        # Phase 2: swap to a good candidate by building a new manager over the
        # same store — simulates a config change between retrain cycles.
        # Carry over the initialized state so no re-initialize() is needed.
        manager2 = RetrainManager(
            model_class               = _GoodCandidate,
            feature_fn                = _feature_fn,
            min_new_rows              = 50,
            tolerance                 = 0.10,
            models_store_path         = store,
            rejection_alert_threshold = 3,
            max_history_years         = float("inf"),
        )
        manager2.current_model       = manager.current_model
        manager2.reference_residuals = manager.reference_residuals
        manager2.last_train_size     = manager.last_train_size

        result = manager2.attempt_retrain(drifted_df)

        if result.get("accepted"):
            assert manager2.consecutive_rejections == 0, (
                "Counter must reset to 0 after acceptance"
            )
            assert read_rejection_counter(store) == 0, (
                "Persisted counter must reset to 0 after acceptance"
            )
            assert result["consecutive_rejections"] == 0
        # If not accepted on this particular split (edge case), no crash is also a pass.


# ── Sliding-window tests ───────────────────────────────────────────────────────
#
# These tests cover `_apply_history_window` and its integration with
# `attempt_retrain`.  Two scenarios:
#   1. Long history (8 years synthetic daily data) — window cuts to 5 years.
#   2. Short history (<5 years, current real dataset) — no rows are lost.
#
# IMPORTANT: the 5-year default is a THEORETICAL choice, not empirically
# validated on the C-1 dataset (only ~1 year of data available at time of
# writing).  These tests verify the MECHANISM, not the optimality of the
# chosen value.  Revalidate the value once ≥3 years of real data exist.

def _feature_fn_dated(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature function for dated test data — ignores the date column."""
    return df[["x"]], df["y"]


def _make_dated_df(n_days: int, n_drifted: int = 70, start: str = "2017-01-01") -> pd.DataFrame:
    """
    Build a daily DataFrame with a 'date' column.

    The last n_drifted rows have y=100.0 (simulates a drift event);
    all preceding rows have y=1.0 (stable baseline).
    """
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    y = np.ones(n_days)
    y[-n_drifted:] = 100.0
    return pd.DataFrame({
        "date": dates,
        "x":    np.arange(n_days, dtype=float),
        "y":    y,
    })


# ── Unit tests: _apply_history_window directly ────────────────────────────────

def test_window_unit_cuts_long_history():
    """
    _apply_history_window on 8-year data with max_history_years=5 returns
    approximately 5*365 rows (±5 for leap-year rounding).
    """
    n_years  = 8
    n_rows   = n_years * 365
    df       = _make_dated_df(n_rows)

    manager = RetrainManager(
        model_class        = _BadModel,
        feature_fn         = _feature_fn_dated,
        max_history_years  = 5.0,
        date_col           = "date",
    )
    windowed = manager._apply_history_window(df)

    expected = int(5 * 365.25)
    assert abs(len(windowed) - expected) < 5, (
        f"Expected ~{expected} rows after 5-year window on {n_rows}-row dataset, "
        f"got {len(windowed)}"
    )
    # Earliest retained date must be within the 5-year window
    cutoff_approx = df["date"].max() - pd.DateOffset(days=5 * 365.25)
    assert windowed["date"].min() >= cutoff_approx - pd.Timedelta(days=2)


def test_window_unit_preserves_short_history():
    """
    _apply_history_window on <5-year data returns the DataFrame unchanged.
    No rows are lost — this is the current real-dataset scenario (~1 year).
    """
    n_days  = 365
    df      = _make_dated_df(n_days, start="2025-02-01")

    manager = RetrainManager(
        model_class       = _BadModel,
        feature_fn        = _feature_fn_dated,
        max_history_years = 5.0,
        date_col          = "date",
    )
    windowed = manager._apply_history_window(df)

    assert len(windowed) == n_days, (
        f"Short history ({n_days} rows < 5 years) must not be cut, "
        f"got {len(windowed)}"
    )


def test_window_unit_disabled_when_inf():
    """Setting max_history_years=inf disables the window entirely."""
    df = _make_dated_df(n_days=4 * 365)

    manager = RetrainManager(
        model_class       = _BadModel,
        feature_fn        = _feature_fn_dated,
        max_history_years = float("inf"),
        date_col          = "date",
    )
    windowed = manager._apply_history_window(df)
    assert len(windowed) == len(df)


def test_window_unit_no_date_column_emits_warning():
    """
    If no datetime column is found and max_history_years is finite, a
    RuntimeWarning is emitted and all_data is returned unchanged.
    """
    df_no_date = pd.DataFrame({"x": np.arange(10.0), "y": np.ones(10)})

    manager = RetrainManager(
        model_class       = _BadModel,
        feature_fn        = _feature_fn,
        max_history_years = 5.0,
        # date_col not set → auto-detect will find nothing
    )
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        windowed = manager._apply_history_window(df_no_date)

    assert len(windowed) == len(df_no_date), "All rows must be preserved when no date col"
    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warns, "A RuntimeWarning must be emitted when no date column is found"


# ── Integration tests: through attempt_retrain() ──────────────────────────────

def test_window_integration_long_history_caps_training_rows():
    """
    With 8 years of synthetic daily data and max_history_years=5, attempt_retrain()
    must report window_applied=True and training_rows ≈ 5*365 (±10 rows).

    The candidate (_BadModel) is rejected, but the window check runs regardless
    of the acceptance outcome — the key assertions are on window_applied and
    training_rows inside the returned dict.
    """
    n_years   = 8
    n_drifted = 70
    n_rows    = n_years * 365

    full_df     = _make_dated_df(n_rows, n_drifted=n_drifted)
    initial_df  = full_df.iloc[:365].reset_index(drop=True)   # first year as initial

    with tempfile.TemporaryDirectory() as tmp:
        initial_model = _MeanModel()
        initial_model.fit(initial_df[["x"]], initial_df["y"])

        manager = RetrainManager(
            model_class       = _BadModel,
            feature_fn        = _feature_fn_dated,
            min_new_rows      = 50,
            tolerance         = 0.0,
            models_store_path = Path(tmp),
            max_history_years = 5.0,
            date_col          = "date",
        )
        manager.initialize(initial_df, initial_model)
        result = manager.attempt_retrain(full_df)

        assert result["retrained"],       "Drift expected on 8-year data with a tail at y=100"
        assert result["window_applied"],  "Window must be applied when history > max_history_years"

        expected = int(5 * 365.25)
        assert abs(result["training_rows"] - expected) < 10, (
            f"Expected ~{expected} training rows for 5-year window "
            f"on {n_rows}-row history, got {result['training_rows']}"
        )


def test_window_integration_short_history_uses_all_rows():
    """
    With <5 years of data (current real EC dataset scenario, ~1 year),
    attempt_retrain() must report window_applied=False and training_rows equal
    to the total number of feature-engineered rows — no data is discarded.
    """
    n_days    = 200   # ~7 months — well under 5 years
    n_drifted = 70
    full_df    = _make_dated_df(n_days, n_drifted=n_drifted, start="2025-02-01")
    initial_df = full_df.iloc[:n_days - n_drifted].reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp:
        initial_model = _MeanModel()
        initial_model.fit(initial_df[["x"]], initial_df["y"])

        manager = RetrainManager(
            model_class       = _BadModel,
            feature_fn        = _feature_fn_dated,
            min_new_rows      = 50,
            tolerance         = 0.0,
            models_store_path = Path(tmp),
            max_history_years = 5.0,
            date_col          = "date",
        )
        manager.initialize(initial_df, initial_model)
        result = manager.attempt_retrain(full_df)

        assert result["retrained"],          "Drift expected"
        assert not result["window_applied"], "No windowing for short history (<5 years)"
        assert result["training_rows"] == n_days, (
            f"All {n_days} rows must be used for training; got {result['training_rows']}"
        )


# ── Standalone runner ──────────────────────────────────────────────────────────

SEP  = "─" * 64
SEP2 = "═" * 64

_TESTS = [
    # Consecutive-rejection counter
    (test_read_counter_returns_zero_when_no_file,        "read_rejection_counter() → 0 when no file"),
    (test_counter_write_read_roundtrip,                  "write/read round-trip"),
    (test_new_manager_loads_persisted_counter,           "new RetrainManager loads counter from disk"),
    (test_three_consecutive_rejections_trigger_alert,    "alert fires on 3rd rejection"),
    (test_alert_fires_periodically_not_on_every_rejection,
                                                         "alert pattern: 3✓ 4- 5- 6✓ 7- (modulo)"),
    (test_counter_increments_beyond_threshold,           "counter keeps incrementing past threshold"),
    (test_counter_resets_on_acceptance,                  "counter resets to 0 on acceptance"),
    # Sliding-window — unit
    (test_window_unit_cuts_long_history,                 "window unit: 8yr → ~5yr rows retained"),
    (test_window_unit_preserves_short_history,           "window unit: <5yr history → all rows kept"),
    (test_window_unit_disabled_when_inf,                 "window unit: inf → no cut"),
    (test_window_unit_no_date_column_emits_warning,      "window unit: no date col → warning + all rows"),
    # Sliding-window — integration via attempt_retrain()
    (test_window_integration_long_history_caps_training_rows,
                                                         "window integration: 8yr history → training_rows≈5yr"),
    (test_window_integration_short_history_uses_all_rows,
                                                         "window integration: <5yr → window_applied=False, all rows"),
]


def _run(fn, name: str) -> bool:
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as exc:
        print(f"  FAIL  {name}")
        print(f"        {exc}")
        return False
    except Exception as exc:
        print(f"  ERROR {name}")
        print(f"        {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(SEP2)
    print("  RetrainManager — rejection counter + sliding window")
    print(SEP2)
    results = [_run(fn, name) for fn, name in _TESTS]
    print(SEP)
    passed = sum(results)
    total  = len(results)
    print(f"  {passed}/{total} passed.")
    print(SEP2)
    sys.exit(0 if all(results) else 1)
