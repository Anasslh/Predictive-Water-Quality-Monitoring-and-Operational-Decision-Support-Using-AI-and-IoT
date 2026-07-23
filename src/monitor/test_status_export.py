"""
test_status_export.py — Tests for write_status_export() and get_dashboard_export_paths().

Three scenarios:
  A) 15 measurements → skill score present and correct
  B)  5 measurements → insufficient_data=True, skill/rmse/mae all null
  C) Per-measurement write verification: status.json updated after EACH call
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Resolve project root ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.monitor import MonitorResult
from src.monitor.export import (
    append_measurement,
    get_dashboard_export_paths,
    write_status_export,
    _MIN_SKILL_MEASUREMENTS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_result(
    prediction: float,
    actual: float,
    ts: datetime,
    parameter_name: str = "TestParam",
) -> MonitorResult:
    """Build a minimal MonitorResult with known prediction/actual pair."""
    r = MonitorResult(parameter_name=parameter_name)
    r.timestamp   = ts.isoformat()
    r.prediction  = prediction
    r.residual    = actual - prediction   # actual = prediction + residual
    r.processed_at = ts
    return r


def _simulate(
    exports_dir: Path,
    parameter_name: str,
    predictions: list[float],
    actuals: list[float],
    interval_hours: int = 1,
    unit: str = "test_unit",
) -> None:
    """
    Append measurements one by one and write status.json after each.
    Timestamps are placed recent-to-now so they fall within the 30-day window.
    """
    n = len(predictions)
    now = datetime.now(timezone.utc)
    # Spread backwards from (now - 1h) so all n measurements stay within 30 days
    base_ts = now - timedelta(hours=n * interval_hours)

    for i, (pred, act) in enumerate(zip(predictions, actuals)):
        ts = base_ts + timedelta(hours=i * interval_hours)
        r  = _make_result(pred, act, ts, parameter_name)
        append_measurement(
            parameter_name,
            r,
            exports_dir    = exports_dir,
            retention_days = 30,
        )
        write_status_export(
            parameter_name,
            unit              = unit,
            models_store_path = None,
            exports_dir       = exports_dir,
            retention_days    = 30,
        )


# ── Test A: 15 measurements → valid skill score ────────────────────────────────

def test_15_measurements_skill_score() -> None:
    """After 15 measurements, skill score must be present and match reference values."""
    actual    = [100, 105, 102, 108, 115, 110, 120, 118, 125, 122, 130, 128, 135, 132, 140]
    predicted = [102, 106, 103, 107, 114, 112, 119, 120, 126, 124, 131, 127, 136, 133, 141]

    # Reference values (computed analytically, see test header):
    #   rmse_model = 1.2817, mae_model = 1.2143, rmse_pers = 5.9522, skill = 78.47%
    expected_skill = 78.47
    expected_rmse  = 1.2817
    expected_mae   = 1.2143

    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp)
        param       = "EC_test"

        _simulate(exports_dir, param, predicted, actual)

        status_file = exports_dir / f"{param}_status.json"
        assert status_file.exists(), "status.json not created"

        status = json.loads(status_file.read_text(encoding="utf-8"))
        perf   = status["performance_30d"]

        assert "insufficient_data" not in perf, (
            f"insufficient_data present unexpectedly: {perf}"
        )
        assert perf["n_measurements"] == 15, (
            f"Expected n=15, got {perf['n_measurements']}"
        )

        skill = perf["skill_vs_persistence_pct"]
        rmse  = perf["rmse"]
        mae   = perf["mae"]

        assert skill is not None, "skill is None but should be computed"
        assert abs(skill - expected_skill) < 0.02, (
            f"skill {skill:.2f} != expected {expected_skill:.2f}"
        )
        assert abs(rmse - expected_rmse) < 0.0002, (
            f"rmse {rmse:.4f} != expected {expected_rmse:.4f}"
        )
        assert abs(mae - expected_mae) < 0.0002, (
            f"mae {mae:.4f} != expected {expected_mae:.4f}"
        )

        # Top-level fields present
        assert status["parameter_name"] == param
        assert status["unit"] == "test_unit"
        assert status["model_version"] is None          # no models_store_path passed
        assert status["consecutive_rejections"] == 0
        assert status["pending_approvals"] == 0
        assert status["last_measurement_at"] is not None


def _run_test(fn) -> bool:
    try:
        fn()
        print(f"PASS  {fn.__name__}")
        return True
    except Exception:
        print(f"FAIL  {fn.__name__}")
        traceback.print_exc()
        return False


# ── Test B: 5 measurements → insufficient_data ────────────────────────────────

def test_5_measurements_insufficient_data() -> None:
    """Fewer than _MIN_SKILL_MEASUREMENTS → insufficient_data=True, metrics null."""
    n = 5
    assert n < _MIN_SKILL_MEASUREMENTS, "sanity: 5 < threshold"

    actual    = [100.0, 102.0, 98.0, 105.0, 101.0]
    predicted = [101.0, 103.0, 99.0, 104.0, 102.0]

    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp)
        param       = "NewParam_test"

        _simulate(exports_dir, param, predicted, actual)

        status_file = exports_dir / f"{param}_status.json"
        assert status_file.exists()

        status = json.loads(status_file.read_text(encoding="utf-8"))
        perf   = status["performance_30d"]

        assert perf.get("insufficient_data") is True, (
            f"Expected insufficient_data=True, got: {perf}"
        )
        assert perf["skill_vs_persistence_pct"] is None, (
            f"Expected skill=None, got {perf['skill_vs_persistence_pct']}"
        )
        assert perf["rmse"] is None, f"Expected rmse=None, got {perf['rmse']}"
        assert perf["mae"]  is None, f"Expected mae=None,  got {perf['mae']}"
        assert perf["n_measurements"] == n, (
            f"Expected n={n}, got {perf['n_measurements']}"
        )


# ── Test C: status.json updated after EVERY measurement ───────────────────────

def test_status_written_after_each_measurement() -> None:
    """queried_at must advance and n_measurements must grow after each call."""
    actual    = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    predicted = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5]

    now = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp)
        param       = "StepTest"
        status_file = exports_dir / f"{param}_status.json"

        prev_n       = -1
        prev_queried = ""

        for i, (pred, act) in enumerate(zip(predicted, actual)):
            ts = now - timedelta(hours=len(predicted) - i)
            r  = _make_result(pred, act, ts, param)
            append_measurement(param, r, exports_dir=exports_dir, retention_days=30)
            write_status_export(
                param,
                unit              = "",
                models_store_path = None,
                exports_dir       = exports_dir,
                retention_days    = 30,
            )

            assert status_file.exists(), f"status.json absent after measurement {i+1}"

            status = json.loads(status_file.read_text(encoding="utf-8"))
            n      = status["performance_30d"]["n_measurements"]
            qt     = status["queried_at"]

            assert n == i + 1, f"step {i+1}: expected n={i+1}, got n={n}"
            assert n > prev_n, f"step {i+1}: n did not grow ({n} <= {prev_n})"
            assert qt != prev_queried or i == 0, (
                f"step {i+1}: queried_at did not change ({qt!r})"
            )

            prev_n       = n
            prev_queried = qt


# ── Test D: get_dashboard_export_paths returns correct paths ───────────────────

def test_get_dashboard_export_paths() -> None:
    """Returned paths must match the actual files written by the export functions."""
    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp)
        param       = "pH_test"
        actual    = [7.0, 7.1, 7.2]
        predicted = [7.05, 7.15, 7.25]

        _simulate(exports_dir, param, predicted, actual)

        paths = get_dashboard_export_paths(param, exports_dir)

        assert "measurements" in paths and "status" in paths
        assert paths["measurements"].exists(), "measurements JSONL not found at returned path"
        assert paths["status"].exists(),       "status JSON not found at returned path"
        assert paths["measurements"].suffix == ".jsonl"
        assert paths["status"].suffix       == ".json"


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_all() -> None:
    tests = [
        test_15_measurements_skill_score,
        test_5_measurements_insufficient_data,
        test_status_written_after_each_measurement,
        test_get_dashboard_export_paths,
    ]
    passed = sum(_run_test(t) for t in tests)
    print(f"\nRESULT: {passed}/{len(tests)} tests passed")
    if passed < len(tests):
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
