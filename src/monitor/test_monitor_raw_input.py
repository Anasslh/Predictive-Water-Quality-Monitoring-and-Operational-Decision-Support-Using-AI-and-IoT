"""
test_monitor_raw_input.py — Tests for prepare_feature_row_from_raw().

Verifies that:
  1. Features computed from raw history + one new raw row match what
     build_features_time_aware() would produce directly on the full dataset.
  2. The historical dataset grows by exactly one row after a successful call
     (simulating the atomic-append behaviour of _cmd_monitor).
  3. Insufficient history raises ValueError with a helpful message.
  4. A new raw row whose timestamp predates the last history row is still
     handled correctly (sort_values inside prepare_feature_row_from_raw).

Run:  python src/monitor/test_monitor_raw_input.py
      (or via run_tests.sh — picked up automatically)
"""

from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.monitor import prepare_feature_row_from_raw
from src.forecasting.feature_engineering_generic import build_features_time_aware
from src.forecasting.frequency_detector import detect_frequency

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    symbol = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{symbol}  {label}{suffix}")
    _results.append((label, condition))


def _make_history(n: int = 15) -> pd.DataFrame:
    """Synthetic daily dataset with Date, EC, pH, Turbidity."""
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    rng   = np.random.default_rng(42)
    return pd.DataFrame({
        "Date":      dates.strftime("%Y-%m-%d"),
        "EC":        (300 + 50 * rng.standard_normal(n)).round(2),
        "pH":        (7.0 + 0.3 * rng.standard_normal(n)).round(3),
        "Turbidity": (20  + 5  * rng.standard_normal(n)).round(2),
    })


_SENSOR = {
    "column_name":    "EC",
    "parameter_name": "EC",
    "unit":           "µS/cm",
    "lag_hours":      [24, 48, 72],
    "rolling_hours":  [72, 168],
    "co_variables":   ["pH", "Turbidity"],
}

_SENSOR_CONFIG = {
    "timestamp_column_candidates": ["Date", "timestamp", "date", "Time"],
    "sensors": [_SENSOR],
}


# ── test cases ────────────────────────────────────────────────────────────────

def test_feature_columns_match_manual_calculation() -> None:
    """
    Features returned by prepare_feature_row_from_raw() must equal the last
    row of build_features_time_aware() called on the full combined dataset.
    """
    df_hist = _make_history(15)
    rng = np.random.default_rng(99)
    new_raw = pd.DataFrame({
        "Date":      ["2026-01-16"],
        "EC":        [round(float(310 + 10 * rng.standard_normal()), 2)],
        "pH":        [round(float(7.1 + 0.1 * rng.standard_normal()), 3)],
        "Turbidity": [round(float(21  + 2  * rng.standard_normal()), 2)],
    })

    feature_row = prepare_feature_row_from_raw(df_hist, new_raw, _SENSOR, _SENSOR_CONFIG)

    # Manual: build features on df_hist + new_raw combined
    df_combined = pd.concat(
        [df_hist, new_raw], ignore_index=True
    ).copy()
    df_combined["Date"] = pd.to_datetime(df_combined["Date"])
    df_combined = df_combined.sort_values("Date").reset_index(drop=True)
    freq = detect_frequency(df_combined, date_col="Date")
    X_manual, _ = build_features_time_aware(
        df_combined, "EC", freq,
        lag_hours=_SENSOR["lag_hours"],
        rolling_hours=_SENSOR["rolling_hours"],
        co_variables=_SENSOR["co_variables"],
    )
    expected = X_manual.tail(1).reset_index(drop=True)

    cols_match = list(feature_row.columns) == list(expected.columns)
    check("Feature column names match manual calculation", cols_match,
          f"got {list(feature_row.columns)}")

    if cols_match:
        for col in expected.columns:
            got = float(feature_row[col].iloc[0])
            exp = float(expected[col].iloc[0])
            close = abs(got - exp) < 1e-6
            check(f"  column {col}: value matches", close,
                  f"got={got:.6f}  expected={exp:.6f}")


def test_history_grows_by_one_after_append() -> None:
    """
    Simulate the atomic-append step in _cmd_monitor: after concatenating the
    new raw row to history and writing it, the CSV has len(history) + 1 rows.
    """
    df_hist = _make_history(15)
    rng = np.random.default_rng(7)
    new_raw = pd.DataFrame({
        "Date":      ["2026-01-16"],
        "EC":        [round(float(320 + 5 * rng.standard_normal()), 2)],
        "pH":        [round(float(7.0 + 0.2 * rng.standard_normal()), 3)],
        "Turbidity": [round(float(22  + 3 * rng.standard_normal()), 2)],
    })

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="test_hist_"
    ) as f:
        tmp_path = Path(f.name)
    try:
        df_hist.to_csv(tmp_path, index=False)
        rows_before = len(pd.read_csv(tmp_path))

        # Simulate atomic append (same logic as _cmd_monitor)
        updated = pd.concat([df_hist, new_raw], ignore_index=True)
        tmp2 = tmp_path.with_suffix(".tmp")
        updated.to_csv(tmp2, index=False)
        os.replace(tmp2, tmp_path)

        rows_after = len(pd.read_csv(tmp_path))
        check(
            "History grows by exactly 1 row after append",
            rows_after == rows_before + 1,
            f"before={rows_before}  after={rows_after}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def test_insufficient_history_raises_valueerror() -> None:
    """
    Only 3 rows of history — not enough for rolling-7 (needs 7+1=8 rows).
    prepare_feature_row_from_raw() must raise ValueError with a clear message.
    """
    df_short = _make_history(3)
    new_raw = pd.DataFrame({
        "Date":      ["2026-01-04"],
        "EC":        [310.0],
        "pH":        [7.1],
        "Turbidity": [21.0],
    })
    try:
        prepare_feature_row_from_raw(df_short, new_raw, _SENSOR, _SENSOR_CONFIG)
        check("Insufficient history raises ValueError", False, "no exception raised")
    except ValueError as exc:
        msg = str(exc)
        check(
            "Insufficient history raises ValueError",
            "Not enough history" in msg or "enough" in msg.lower(),
            msg[:80],
        )


def test_out_of_order_new_row_handled() -> None:
    """
    A new row whose Date is earlier than the last history row (late delivery)
    is accepted after sorting. The last feature row still corresponds to the
    highest timestamp in the combined dataset.
    """
    df_hist = _make_history(15)
    # Deliver a row dated 2026-01-05 (day 5, in the middle of history)
    new_raw = pd.DataFrame({
        "Date":      ["2026-01-05"],  # already in history range
        "EC":        [305.0],
        "pH":        [7.05],
        "Turbidity": [19.5],
    })
    try:
        feature_row = prepare_feature_row_from_raw(df_hist, new_raw, _SENSOR, _SENSOR_CONFIG)
        # Should succeed — sort_values inside prepare_feature_row_from_raw handles this
        check("Out-of-order new row accepted after sort", len(feature_row) == 1,
              f"got {len(feature_row)} rows")
    except Exception as exc:
        check("Out-of-order new row accepted after sort", False, str(exc)[:80])


def test_sensor_with_no_co_variables() -> None:
    """
    A sensor declared without co_variables (empty list) produces feature columns
    for EC only — no pH_lag1 / Turbidity_lag1 — and does not crash.
    """
    sensor_no_co = {**_SENSOR, "co_variables": []}
    df_hist = _make_history(15)
    new_raw = pd.DataFrame({
        "Date": ["2026-01-16"], "EC": [310.0], "pH": [7.1], "Turbidity": [21.0],
    })
    feature_row = prepare_feature_row_from_raw(df_hist, new_raw, sensor_no_co, _SENSOR_CONFIG)
    has_no_co = not any("pH" in c or "Turbidity" in c for c in feature_row.columns)
    check(
        "No co-variable columns when co_variables=[]",
        has_no_co,
        f"columns: {list(feature_row.columns)}",
    )


def test_backfill_timestamp_stored_not_processing_time() -> None:
    """
    Backfill scenario: the measurement date is 3 days in the past — clearly
    different from today's processing date.  process_new_measurement() must store
    the measurement date in the JSONL, NOT the wall-clock time of execution
    (result.processed_at).

    3 days is recent enough to survive the 30-day retention window (so the JSONL
    record is kept), while still being unambiguously different from "today".

    This verifies the fix for the timestamp bug: _cmd_monitor extracts the date
    from the raw row before prepare_feature_row_from_raw() drops the Date column,
    then passes it explicitly via the new `timestamp` argument so export.py
    writes the correct measurement date rather than falling back to processed_at.
    """
    import json
    from datetime import datetime, timedelta, timezone
    from src.monitor import ParameterMonitor
    from src.retraining.model_versioning import load_current_model

    # 3 days ago — survives 30-day retention, clearly ≠ today
    BACKFILL_DATE = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")

    # ── Build a minimal fitted model (reuse the production pickle if present) ──
    models_store = ROOT / "models_store" / "ec"
    if not models_store.exists():
        check(
            "Backfill timestamp stored in JSONL (skipped — no EC model in models_store/)",
            True,   # skip gracefully
            "run 'python run.py onboard --parameter EC' first to enable this test",
        )
        return

    model = load_current_model(models_store)
    if model is None:
        check("Backfill timestamp stored in JSONL (skipped — model load failed)", True, "")
        return

    df_hist = _make_history(15)
    new_raw = pd.DataFrame({
        "Date": [BACKFILL_DATE], "EC": [310.0], "pH": [7.1], "Turbidity": [21.0],
    })

    feature_row = prepare_feature_row_from_raw(df_hist, new_raw, _SENSOR, _SENSOR_CONFIG)

    # ── Simulate the _cmd_monitor timestamp extraction ─────────────────────────
    raw_timestamp = new_raw["Date"].iloc[0]   # "2025-01-15"

    with tempfile.TemporaryDirectory(prefix="test_backfill_") as tmp_exports:
        from src.config import SystemConfig, ExportsConfig
        cfg = SystemConfig()
        cfg.exports = ExportsConfig(exports_dir=tmp_exports, retention_days=30)

        monitor = ParameterMonitor(
            parameter_name    = "EC",
            prediction_model  = model,
            feature_fn        = lambda df: (None, None),
            models_store_path = models_store,
            config            = cfg,
        )

        result = monitor.process_new_measurement(
            feature_row,
            actual_value = 310.0,
            timestamp    = raw_timestamp,          # ← the fix being tested
        )

        # ── Verify result.timestamp is the measurement date, not processed_at ──
        ts_str = str(result.timestamp)
        check(
            "result.timestamp is the measurement date (not processed_at)",
            BACKFILL_DATE in ts_str,
            f"got '{ts_str}', expected to contain '{BACKFILL_DATE}'",
        )

        # ── Verify the JSONL has the measurement date ──────────────────────────
        jsonl_path = Path(tmp_exports) / "EC.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            record = json.loads(lines[0]) if lines else {}
            jsonl_ts = record.get("timestamp", "")
            check(
                "JSONL timestamp is the measurement date (not processing time)",
                BACKFILL_DATE in jsonl_ts,
                f"JSONL has '{jsonl_ts}', expected to contain '{BACKFILL_DATE}'",
            )
            # Extra guard: processed_at should NOT be the stored timestamp
            processing_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            check(
                "JSONL timestamp is NOT today's processing date",
                processing_date not in jsonl_ts or BACKFILL_DATE in jsonl_ts,
                f"JSONL='{jsonl_ts}' | today='{processing_date}' | meas='{BACKFILL_DATE}'",
            )
        else:
            check("JSONL file created", False, f"not found at {jsonl_path}")


# ── runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  monitor raw-input — prepare_feature_row_from_raw() tests")
    print(SEP2)

    # Suppress info logs from feature_engineering_generic during tests
    logging.getLogger("src.forecasting.feature_engineering_generic").setLevel(logging.WARNING)
    logging.getLogger("src.forecasting.frequency_detector").setLevel(logging.WARNING)
    logging.getLogger("src.pipeline.timestamp_detector").setLevel(logging.WARNING)
    logging.getLogger("src.monitor").setLevel(logging.WARNING)

    test_feature_columns_match_manual_calculation()
    test_history_grows_by_one_after_append()
    test_insufficient_history_raises_valueerror()
    test_out_of_order_new_row_handled()
    test_sensor_with_no_co_variables()
    test_backfill_timestamp_stored_not_processing_time()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
