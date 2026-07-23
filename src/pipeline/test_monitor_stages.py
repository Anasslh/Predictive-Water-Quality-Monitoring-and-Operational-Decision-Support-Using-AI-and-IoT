"""
test_monitor_stages.py — Verify that the four monitor stages wire correctly
into ParameterMonitor and that the CLI flags control which stages are active.

Tests
-----
1. Forecaster missing → forecast stage skipped cleanly (no crash).
2. Forecaster present → forecast produces a multi-step trajectory.
3. Retrain check: all_data=None  → Stage 3 skipped (retrain_result is None).
4. Retrain check: all_data=df    → Stage 3 runs (retrain_result is set).
5. Parser smoke test: --check-retrain / --no-anomaly / --no-forecast flags parsed.
6. save_forecaster_config + load_forecaster round-trip preserves parameters.

Run:  python src/pipeline/test_monitor_stages.py
      (also picked up automatically by run_tests.sh)
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── Test helpers ───────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    symbol = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{symbol}  {label}{suffix}")
    _results.append((label, condition))


# ── Shared fixture ─────────────────────────────────────────────────────────────

def _make_synthetic_ec(n: int = 150) -> pd.DataFrame:
    rng   = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    ec    = (200 + 15 * rng.standard_normal(n)).round(2)
    return pd.DataFrame({"Date": dates, "EC": ec})


def _build_model_and_splits(df: pd.DataFrame):
    """Return (model, X_all, y_all, X_te, y_te, frequency) for the EC column."""
    from src.forecasting.feature_engineering_generic import build_features_time_aware
    from src.forecasting.frequency_detector import detect_frequency
    from src.data.split import chronological_split
    from src.pipeline.model_benchmark import _GenericRFModel

    freq = detect_frequency(df, date_col="Date")
    X, y = build_features_time_aware(
        df, target="EC", frequency=freq,
        lag_hours=[24, 48], rolling_hours=[72], co_variables=[],
    )
    X_tr, y_tr, X_v, y_v, X_te, y_te = chronological_split(X, y)
    X_all = pd.concat([X_tr, X_v]).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_v]).reset_index(drop=True)

    model = _GenericRFModel("EC", "rf_test", n_estimators=50)
    model.fit(X_all, y_all)
    return model, X_all, y_all, X_te, y_te, freq


# ── Test 1: forecaster missing → clean skip ────────────────────────────────────

def test_forecast_missing_skips_cleanly() -> None:
    """
    ParameterMonitor with forecaster=None must not crash and must produce
    result.forecast = None even when _historical_window is provided.
    """
    df    = _make_synthetic_ec()
    model, X_all, y_all, X_te, y_te, freq = _build_model_and_splits(df)

    from src.monitor import ParameterMonitor

    with tempfile.TemporaryDirectory(prefix="test_mon_no_fc_") as tmp_str:
        monitor = ParameterMonitor(
            parameter_name    = "EC",
            prediction_model  = model,
            feature_fn        = lambda _df: (None, None),
            forecaster        = None,
            models_store_path = Path(tmp_str),
        )
        monitor._historical_window = df   # set a window — should be ignored

        result = monitor.process_new_measurement(
            X_te.iloc[[-1]].reset_index(drop=True),
        )

    check(
        "result.forecast is None when forecaster=None",
        result.forecast is None,
        str(result.forecast),
    )
    check(
        "result.prediction is populated (Stage 1 still runs)",
        result.prediction is not None,
        str(result.prediction),
    )


# ── Test 2: forecaster present → trajectory produced ──────────────────────────

def test_forecast_produces_trajectory() -> None:
    """
    After save_forecaster_config + load_forecaster, passing a historical window
    to ParameterMonitor must produce a non-empty forecast trajectory.
    """
    from src.forecasting.recursive_forecaster import save_forecaster_config, load_forecaster
    from src.monitor import ParameterMonitor

    df    = _make_synthetic_ec()
    model, X_all, y_all, X_te, y_te, freq = _build_model_and_splits(df)

    n_steps       = 5
    step1_std     = float(np.std(y_all.values - model.predict(X_all)))
    residual_std  = [round(step1_std * (k + 1) ** 0.5, 4) for k in range(n_steps)]

    sensor = {
        "parameter_name": "EC",
        "lag_hours":      [24, 48],
        "rolling_hours":  [72],
        "co_variables":   [],
    }

    with tempfile.TemporaryDirectory(prefix="test_mon_fc_") as tmp_str:
        tmp = Path(tmp_str)

        cfg_path = save_forecaster_config(
            target               = "EC",
            frequency            = freq,
            lag_hours            = sensor["lag_hours"],
            rolling_hours        = sensor["rolling_hours"],
            co_variables         = sensor["co_variables"],
            residual_std_by_step = residual_std,
            models_store_path    = tmp,
            n_steps              = n_steps,
        )
        check(
            "forecaster_config.json created",
            cfg_path.exists(),
            str(cfg_path),
        )

        forecaster = load_forecaster(tmp, model)
        check(
            "load_forecaster returns MultiStepForecaster",
            forecaster is not None,
            str(type(forecaster)),
        )

        monitor = ParameterMonitor(
            parameter_name    = "EC",
            prediction_model  = model,
            feature_fn        = lambda _df: (None, None),
            forecaster        = forecaster,
            models_store_path = tmp,
        )
        # Provide historical window with datetime-parsed dates
        _window = df.copy()
        _window["Date"] = pd.to_datetime(_window["Date"])
        monitor._historical_window = _window

        result = monitor.process_new_measurement(
            X_te.iloc[[-1]].reset_index(drop=True),
            forecast_steps = getattr(forecaster, "_config_n_steps", n_steps),
        )

    check(
        "result.forecast is not None",
        result.forecast is not None,
        str(type(result.forecast)),
    )
    check(
        f"forecast has {n_steps} predictions",
        result.forecast is not None and len(result.forecast.predictions) == n_steps,
        f"got {len(result.forecast.predictions) if result.forecast else 0}",
    )
    check(
        "all forecast values are finite floats",
        result.forecast is not None and all(
            isinstance(v, float) and np.isfinite(v)
            for v in result.forecast.predictions
        ),
        str(result.forecast.predictions if result.forecast else []),
    )
    check(
        "uncertainty bands populated",
        result.forecast is not None and any(
            s is not None for s in result.forecast.uncertainty_std
        ),
    )


# ── Test 3: retrain skipped when all_data=None ────────────────────────────────

def test_retrain_skipped_without_all_data() -> None:
    """
    Stage 3 must not run when all_data=None, even if retrain_manager is set.
    result.retrain_result must remain None.
    """
    from src.monitor import ParameterMonitor

    df    = _make_synthetic_ec()
    model, X_all, y_all, X_te, y_te, freq = _build_model_and_splits(df)

    mock_rm = MagicMock()
    mock_rm.should_check_retrain.return_value = True

    with tempfile.TemporaryDirectory(prefix="test_mon_no_retrain_") as tmp_str:
        monitor = ParameterMonitor(
            parameter_name    = "EC",
            prediction_model  = model,
            feature_fn        = lambda _df: (None, None),
            retrain_manager   = mock_rm,
            models_store_path = Path(tmp_str),
        )
        result = monitor.process_new_measurement(
            X_te.iloc[[-1]].reset_index(drop=True),
            all_data = None,         # ← no historical data → Stage 3 skipped
        )

    check(
        "retrain_result is None when all_data=None",
        result.retrain_result is None,
        str(result.retrain_result),
    )
    check(
        "attempt_retrain was NOT called",
        not mock_rm.attempt_retrain.called,
    )


# ── Test 4: retrain check triggered when all_data provided ────────────────────

def test_retrain_triggered_with_all_data() -> None:
    """
    Stage 3 must run and populate retrain_result when a RetrainManager is set
    AND all_data is passed — AND the volume gate fires.
    """
    from src.monitor import ParameterMonitor

    df    = _make_synthetic_ec()
    model, X_all, y_all, X_te, y_te, freq = _build_model_and_splits(df)

    _fake_retrain_result = {
        "outcome":        "drift_not_detected",
        "drift_detected": False,
        "alert":          None,
    }
    mock_rm = MagicMock()
    mock_rm.should_check_retrain.return_value = True
    mock_rm.attempt_retrain.return_value      = _fake_retrain_result

    with tempfile.TemporaryDirectory(prefix="test_mon_retrain_") as tmp_str:
        monitor = ParameterMonitor(
            parameter_name    = "EC",
            prediction_model  = model,
            feature_fn        = lambda _df: (None, None),
            retrain_manager   = mock_rm,
            models_store_path = Path(tmp_str),
        )
        result = monitor.process_new_measurement(
            X_te.iloc[[-1]].reset_index(drop=True),
            all_data = df,           # ← pass data → Stage 3 must run
        )

    check(
        "should_check_retrain was called",
        mock_rm.should_check_retrain.called,
    )
    check(
        "attempt_retrain was called (volume gate fired)",
        mock_rm.attempt_retrain.called,
    )
    check(
        "retrain_result populated from manager",
        result.retrain_result == _fake_retrain_result,
        str(result.retrain_result),
    )
    check(
        "retrain_needed = False (no drift in this mock)",
        result.retrain_needed is False,
    )


# ── Test 5: parser smoke test ──────────────────────────────────────────────────

def test_parser_flags_parsed_correctly() -> None:
    """
    --check-retrain, --no-anomaly, and --no-forecast must parse without error
    and set the correct attribute values on the Namespace.
    """
    from run import _build_parser

    parser = _build_parser()

    # All three flags present
    args_all = parser.parse_args([
        "monitor",
        "--parameter", "EC",
        "--dataset",   "data.csv",
        "--new-row",   "row.csv",
        "--check-retrain",
        "--no-anomaly",
        "--no-forecast",
    ])
    check("--check-retrain parsed → True",  args_all.check_retrain is True)
    check("--no-anomaly parsed  → True",    args_all.no_anomaly    is True)
    check("--no-forecast parsed → True",    args_all.no_forecast   is True)

    # All three flags absent (defaults)
    args_def = parser.parse_args([
        "monitor",
        "--parameter", "EC",
        "--dataset",   "data.csv",
        "--new-row",   "row.csv",
    ])
    check("--check-retrain absent → False", args_def.check_retrain is False)
    check("--no-anomaly absent    → False", args_def.no_anomaly    is False)
    check("--no-forecast absent   → False", args_def.no_forecast   is False)


# ── Test 6: save/load forecaster config round-trip ────────────────────────────

def test_forecaster_config_roundtrip() -> None:
    """
    save_forecaster_config + load_forecaster must reconstruct a forecaster
    that uses the exact same frequency and produces finite predictions.
    """
    from src.forecasting.recursive_forecaster import (
        save_forecaster_config, load_forecaster, MultiStepForecaster,
    )
    from datetime import timedelta

    df    = _make_synthetic_ec()
    model, X_all, y_all, X_te, y_te, freq = _build_model_and_splits(df)

    with tempfile.TemporaryDirectory(prefix="test_fc_roundtrip_") as tmp_str:
        tmp = Path(tmp_str)
        save_forecaster_config(
            target               = "EC",
            frequency            = timedelta(hours=24),
            lag_hours            = [24, 48],
            rolling_hours        = [72],
            co_variables         = [],
            residual_std_by_step = [3.0, 4.2, 5.2],
            models_store_path    = tmp,
            n_steps              = 3,
        )
        fc = load_forecaster(tmp, model)

    check("load_forecaster returns a MultiStepForecaster",  isinstance(fc, MultiStepForecaster))
    check("frequency preserved (24h)",   fc is not None and fc.frequency == timedelta(hours=24))
    check("3 uncertainty bands loaded",  fc is not None and len(fc.residual_std_by_step) == 3)

    # Produce a forecast and check outputs
    _window = df.copy()
    _window["Date"] = pd.to_datetime(_window["Date"])
    fc_result = fc.forecast(_window, n_steps=3)
    check(
        "forecast produces 3 finite predictions",
        len(fc_result.predictions) == 3
        and all(np.isfinite(v) for v in fc_result.predictions),
        str(fc_result.predictions),
    )

    check(
        "load_forecaster returns None when file missing",
        load_forecaster(Path("/tmp/does_not_exist_xyz"), model) is None,
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  monitor stages — end-to-end tests")
    print(SEP2)

    for lg in ("src.monitor", "src.forecasting.feature_engineering_generic",
               "src.forecasting.frequency_detector", "src.pipeline.orchestrator",
               "src.retraining.model_versioning"):
        logging.getLogger(lg).setLevel(logging.WARNING)

    test_forecast_missing_skips_cleanly()
    print(SEP)
    test_forecast_produces_trajectory()
    print(SEP)
    test_retrain_skipped_without_all_data()
    print(SEP)
    test_retrain_triggered_with_all_data()
    print(SEP)
    test_parser_flags_parsed_correctly()
    print(SEP)
    test_forecaster_config_roundtrip()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
