"""
test_n_steps_calculation.py — Verify that n_steps is derived from
forecast_horizon_hours / detected_frequency_hours (not a hardcoded constant).

Tests
-----
1. EC daily (24 h), horizon 72 h  → n_steps = 3
2. Hourly sensor (1 h), horizon 72 h → n_steps = 72
3. Sub-hourly (15 min = 0.25 h), horizon 72 h → n_steps = 288
4. Horizon < frequency → n_steps = 1 (minimum guard)
5. Default horizon (sensor has no forecast_horizon_hours) = 72 h → 3 steps at daily
6. Custom horizon (168 h = 7 days) at daily frequency → n_steps = 7
7. forecaster_config.json written by _train_and_save_forecaster has correct n_steps
8. load_forecaster returns _config_n_steps matching what was saved

Run:  python src/forecasting/test_n_steps_calculation.py
      (also picked up automatically by run_tests.sh)
"""

from __future__ import annotations

import io
import math
import pickle
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import timedelta
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


# ── Helper: compute n_steps (mirrors the fixed run.py logic) ──────────────────

def _compute_n_steps(forecast_horizon_hours: float, freq_hours: float) -> int:
    return max(1, math.floor(forecast_horizon_hours / freq_hours))


# ── Tests 1-6: n_steps arithmetic ─────────────────────────────────────────────

def test_n_steps_arithmetic() -> None:
    cases = [
        # (label, horizon_h, freq_h, expected_steps)
        ("EC daily 24h / horizon 72h",         72,  24,   3),
        ("Hourly sensor / horizon 72h",         72,   1,  72),
        ("15-min sensor / horizon 72h",         72,  0.25, 288),
        ("Horizon < frequency → minimum 1",      6,  24,   1),
        ("Default horizon 72h at daily",         72,  24,   3),
        ("Custom horizon 168h at daily (7 days)",168,  24,  7),
    ]
    for label, h_hours, f_hours, expected in cases:
        got = _compute_n_steps(h_hours, f_hours)
        check(
            label,
            got == expected,
            f"got={got}  expected={expected}",
        )


# ── Tests 7-8: round-trip through _train_and_save_forecaster ─────────────────

def _make_synthetic_ec(n: int = 150) -> pd.DataFrame:
    rng   = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    ec    = (300 + 20 * rng.standard_normal(n)).round(2)
    return pd.DataFrame({"Date": dates, "EC": ec})


def _build_minimal_model_and_splits(df: pd.DataFrame):
    from src.forecasting.feature_engineering_generic import build_features_time_aware
    from src.forecasting.frequency_detector import detect_frequency
    from src.data.split import chronological_split
    from src.pipeline.model_benchmark import _GenericRFModel

    freq = detect_frequency(df, date_col="Date")
    with redirect_stdout(io.StringIO()):
        X, y = build_features_time_aware(
            df, target="EC", frequency=freq,
            lag_hours=[24, 48], rolling_hours=[72], co_variables=[],
        )
    X_tr, y_tr, X_v, y_v, _, _ = chronological_split(X, y)
    X_all = pd.concat([X_tr, X_v]).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_v]).reset_index(drop=True)

    model = _GenericRFModel("EC", "rf_test", n_estimators=20)
    model.fit(X_all, y_all)
    return model, X_all, y_all, freq


def test_train_and_save_writes_correct_n_steps() -> None:
    """
    _train_and_save_forecaster with sensor forecast_horizon_hours=72
    and daily frequency must write n_steps=3 to forecaster_config.json.
    """
    from run import _train_and_save_forecaster
    from src.forecasting.recursive_forecaster import load_forecaster

    df    = _make_synthetic_ec()
    model, X_all, y_all, freq = _build_minimal_model_and_splits(df)

    sensor = {
        "parameter_name":        "EC",
        "lag_hours":             [24, 48],
        "rolling_hours":         [72],
        "co_variables":          [],
        "forecast_horizon_hours": 72,   # explicit: 72 h / 24 h = 3 steps
    }

    with tempfile.TemporaryDirectory(prefix="test_nsteps_") as tmp:
        store = Path(tmp)
        with redirect_stdout(io.StringIO()):
            _train_and_save_forecaster(model, X_all, y_all, sensor, freq, store)

        cfg_file = store / "forecaster_config.json"
        check(
            "forecaster_config.json created",
            cfg_file.exists(),
        )

        import json
        raw = json.loads(cfg_file.read_text(encoding="utf-8"))
        check(
            "n_steps=3 written for EC daily / horizon 72h",
            raw.get("n_steps") == 3,
            f"got {raw.get('n_steps')}",
        )
        check(
            "3 residual_std values stored (one per step)",
            len(raw.get("residual_std_by_step", [])) == 3,
            str(raw.get("residual_std_by_step")),
        )

        forecaster = load_forecaster(store, model)
        check(
            "load_forecaster returns _config_n_steps=3",
            getattr(forecaster, "_config_n_steps", None) == 3,
            str(getattr(forecaster, "_config_n_steps", None)),
        )


def test_hourly_sensor_writes_72_steps() -> None:
    """
    Sensor with forecast_horizon_hours=72 on a 1-hour frequency dataset
    must produce n_steps=72.
    """
    from run import _train_and_save_forecaster
    import json

    # Build a synthetic hourly dataset (frequency = 1 h)
    n     = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    rng   = np.random.default_rng(99)
    df_h  = pd.DataFrame({"Date": dates, "EC": (300 + 10 * rng.standard_normal(n)).round(2)})

    from src.forecasting.feature_engineering_generic import build_features_time_aware
    from src.forecasting.frequency_detector import detect_frequency
    from src.data.split import chronological_split
    from src.pipeline.model_benchmark import _GenericRFModel

    freq = detect_frequency(df_h, date_col="Date")
    freq_h = freq.total_seconds() / 3600
    check(
        "Detected frequency is 1 hour",
        abs(freq_h - 1.0) < 0.01,
        f"freq_h={freq_h}",
    )

    with redirect_stdout(io.StringIO()):
        X, y = build_features_time_aware(
            df_h, target="EC", frequency=freq,
            lag_hours=[1, 2, 3], rolling_hours=[3, 6], co_variables=[],
        )
    X_tr, y_tr, X_v, y_v, _, _ = chronological_split(X, y)
    X_all = pd.concat([X_tr, X_v]).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_v]).reset_index(drop=True)

    model = _GenericRFModel("EC", "rf_hourly", n_estimators=10)
    model.fit(X_all, y_all)

    sensor = {
        "parameter_name":        "EC",
        "lag_hours":             [1, 2, 3],
        "rolling_hours":         [3, 6],
        "co_variables":          [],
        "forecast_horizon_hours": 72,
    }

    with tempfile.TemporaryDirectory(prefix="test_nsteps_h_") as tmp:
        store = Path(tmp)
        with redirect_stdout(io.StringIO()):
            _train_and_save_forecaster(model, X_all, y_all, sensor, freq, store)

        raw = json.loads((store / "forecaster_config.json").read_text(encoding="utf-8"))
        check(
            "n_steps=72 written for hourly sensor / horizon 72h",
            raw.get("n_steps") == 72,
            f"got {raw.get('n_steps')}",
        )
        check(
            "72 residual_std values stored",
            len(raw.get("residual_std_by_step", [])) == 72,
            f"got {len(raw.get('residual_std_by_step', []))}",
        )


def test_missing_horizon_defaults_to_72h() -> None:
    """
    A sensor entry with no forecast_horizon_hours defaults to 72 h.
    At daily frequency, n_steps must equal 3.
    """
    from run import _train_and_save_forecaster
    import json

    df    = _make_synthetic_ec()
    model, X_all, y_all, freq = _build_minimal_model_and_splits(df)

    # No forecast_horizon_hours key
    sensor = {
        "parameter_name": "EC",
        "lag_hours":       [24, 48],
        "rolling_hours":   [72],
        "co_variables":    [],
    }

    with tempfile.TemporaryDirectory(prefix="test_nsteps_default_") as tmp:
        store = Path(tmp)
        with redirect_stdout(io.StringIO()):
            _train_and_save_forecaster(model, X_all, y_all, sensor, freq, store)

        raw = json.loads((store / "forecaster_config.json").read_text(encoding="utf-8"))
        check(
            "Missing forecast_horizon_hours defaults to 72h → n_steps=3",
            raw.get("n_steps") == 3,
            f"got {raw.get('n_steps')}",
        )


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  n_steps calculation — forecast horizon / frequency tests")
    print(SEP2)

    test_n_steps_arithmetic()
    print(SEP)
    test_train_and_save_writes_correct_n_steps()
    print(SEP)
    test_hourly_sensor_writes_72_steps()
    print(SEP)
    test_missing_horizon_defaults_to_72h()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
