"""
test_onboarding_guard.py — Tests for the minimum dataset size guard in
onboard_new_parameter() (src/pipeline/orchestrator.py).

Verified behaviours
-------------------
Guard 1 (raw row count, before any computation):
  - 20 rows  → ValueError, clear message, no ML computation attempted
  - 99 rows  → ValueError (one below minimum)
  - Exactly MIN_ROWS_DATASET rows → guard 1 does NOT fire

Guard 2 (post-FE partition size):
  - Synthetic dataset where rolling windows drop enough rows to push
    val and test below MIN_PARTITION_ROWS → ValueError after FE

Message quality:
  - Error text mentions the actual row count
  - Error text mentions the minimum threshold
  - Error text mentions val and test
  - Error text is NOT just an internal traceback (it reads like a diagnostic)

Run:
    python -m src.pipeline.test_onboarding_guard
    # or
    python src/pipeline/test_onboarding_guard.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.pipeline.orchestrator import (
    MIN_PARTITION_ROWS,
    MIN_ROWS_DATASET,
    onboard_new_parameter,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

# Minimal sensor config — no real sensors_config.json needed in tests.
_MINIMAL_CONFIG = {
    "timestamp_column_candidates": ["Date"],
    "sensors": [
        {
            "column_name":    "EC",
            "parameter_name": "EC",
            "unit":           "µS/cm",
            "wqi_standard":   400,
            "co_variables":   ["pH", "Turbidity"],
            "lag_hours":      [24, 48, 72],
            "rolling_hours":  [72, 168],
        }
    ],
}


def _make_df(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic daily DataFrame with EC, pH, Turbidity columns.
    Dates start at 2025-01-01 and step one day at a time.
    """
    rng   = np.random.default_rng(seed)
    base  = datetime(2025, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n_rows)]
    return pd.DataFrame({
        "Date":      dates,
        "EC":        rng.uniform(200, 500, n_rows),
        "pH":        rng.uniform(6.0, 9.0, n_rows),
        "Turbidity": rng.uniform(0.0, 100.0, n_rows),
    })


def _assert_size_guard_error(exc: ValueError, n_rows: int) -> None:
    """
    Check that a ValueError is the size-guard error, not some unrelated one.
    """
    msg = str(exc)
    assert str(n_rows) in msg, (
        f"Error message should mention the actual row count ({n_rows}):\n{msg}"
    )
    assert str(MIN_ROWS_DATASET) in msg, (
        f"Error message should mention the minimum threshold ({MIN_ROWS_DATASET}):\n{msg}"
    )
    # Must read as a diagnostic, not a raw traceback line
    assert "too small" in msg.lower() or "minimum" in msg.lower(), (
        f"Error message should contain 'too small' or 'minimum':\n{msg}"
    )


# ── Guard 1 tests ──────────────────────────────────────────────────────────────

def test_guard1_20_rows_raises():
    """
    20 rows → ValueError before any FE or model training.

    This is the canonical case requested: an obviously undersized dataset
    must be rejected with an informative message, not silently produce
    a 3-row test set.
    """
    df = _make_df(20)

    raised = False
    try:
        onboard_new_parameter(_MINIMAL_CONFIG, df, "EC")
    except ValueError as exc:
        raised = True
        _assert_size_guard_error(exc, n_rows=20)
        # Confirm the projected split is shown in the message
        msg = str(exc)
        assert "14" in msg or "train" in msg.lower(), (
            "Message should show the projected train size"
        )
        assert "val" in msg.lower() or "test" in msg.lower(), (
            "Message should mention val / test"
        )
        print(f"  [message shown to user]\n{msg}\n")
    except Exception as exc:
        raise AssertionError(
            f"Expected ValueError (size guard), got {type(exc).__name__}: {exc}"
        ) from exc

    assert raised, "onboard_new_parameter() should raise ValueError for 20 rows"
    print("PASS  test_guard1_20_rows_raises")


def test_guard1_one_below_minimum_raises():
    """99 rows (one below MIN_ROWS_DATASET=100) must also be rejected."""
    n = MIN_ROWS_DATASET - 1
    df = _make_df(n)

    raised = False
    try:
        onboard_new_parameter(_MINIMAL_CONFIG, df, "EC")
    except ValueError as exc:
        raised = True
        _assert_size_guard_error(exc, n_rows=n)
    except Exception as exc:
        raise AssertionError(
            f"Expected ValueError, got {type(exc).__name__}: {exc}"
        ) from exc

    assert raised, f"onboard_new_parameter() should raise for {n} rows"
    print(f"PASS  test_guard1_one_below_minimum_raises  ({n} rows → guard fires)")


def test_guard1_does_not_fire_at_minimum():
    """
    MIN_ROWS_DATASET rows pass the raw-count guard.

    The function will continue into _prepare_data / feature-engineering /
    benchmark — it may raise for other reasons (e.g. insufficient FE rows
    from the rolling window) but must NOT raise the size-guard ValueError
    with the 'too small' / 'minimum' wording.
    """
    n  = MIN_ROWS_DATASET   # exactly 100
    df = _make_df(n)

    try:
        onboard_new_parameter(_MINIMAL_CONFIG, df, "EC")
        # If it completes without error, guard 1 definitely did not fire.
    except ValueError as exc:
        msg = str(exc)
        is_size_guard = (
            ("too small" in msg.lower() or "minimum" in msg.lower())
            and str(n) in msg
            and str(MIN_ROWS_DATASET) in msg
        )
        assert not is_size_guard, (
            f"Guard 1 fired at exactly {n} rows — it must not:\n{msg}"
        )
        # Any other ValueError (e.g. guard 2 on partition size, or a
        # training error) is acceptable here; guard 1 must stay silent.
    except Exception:
        pass   # non-ValueError exceptions are out of scope for this test

    print(f"PASS  test_guard1_does_not_fire_at_minimum  ({n} rows → guard 1 silent)")


# ── Guard 2 test ───────────────────────────────────────────────────────────────

def test_guard2_fires_when_fe_shrinks_partitions():
    """
    Guard 2: dataset passes guard 1 (≥ 100 raw rows) but guard 2 fires
    when the rolling window causes val/test partitions to fall below
    MIN_PARTITION_ROWS.

    We simulate this by using very large rolling_hours relative to the
    dataset size so that NaN-drop removes most rows after FE.
    """
    # 100 raw rows, rolling_hours=[2160] = 90 days rolling window
    # → drops 90 rows from lags/rolling → only ~10 FE rows
    # → val ≈ 1 row, test ≈ 2 rows → guard 2 fires
    config_large_window = {
        "timestamp_column_candidates": ["Date"],
        "sensors": [
            {
                "column_name":    "EC",
                "parameter_name": "EC",
                "unit":           "µS/cm",
                "wqi_standard":   400,
                "co_variables":   ["pH", "Turbidity"],
                "lag_hours":      [24, 48, 72],
                "rolling_hours":  [2160],   # 90-day rolling window on daily data
            }
        ],
    }

    df = _make_df(MIN_ROWS_DATASET)   # exactly 100 raw rows → guard 1 passes

    raised = False
    try:
        onboard_new_parameter(config_large_window, df, "EC")
    except ValueError as exc:
        raised = True
        msg = str(exc)
        # This should be guard 2, not guard 1
        assert "too small" in msg.lower() or "minimum" in msg.lower() or "partition" in msg.lower() or "val" in msg.lower(), (
            f"Expected a size-related guard-2 error, got:\n{msg}"
        )
        # Guard 1 message mentions the raw count; guard 2 message mentions FE rows
        assert "FE" in msg or "NaN" in msg or "rolling" in msg.lower() or "val=" in msg, (
            f"Guard 2 message should mention FE / rolling context:\n{msg}"
        )
        print(f"  [guard 2 message]\n{msg}\n")
    except Exception as exc:
        raise AssertionError(
            f"Expected ValueError (guard 2), got {type(exc).__name__}: {exc}"
        ) from exc

    assert raised, "Guard 2 should fire when rolling window shrinks partitions below minimum"
    print("PASS  test_guard2_fires_when_fe_shrinks_partitions")


# ── Constants sanity check ─────────────────────────────────────────────────────

def test_constants_are_reasonable():
    """
    MIN_ROWS_DATASET and MIN_PARTITION_ROWS must satisfy basic invariants:
      - MIN_PARTITION_ROWS ≤ MIN_ROWS_DATASET × 0.15
        (the val set at minimum size must reach the partition threshold)
      - Both are positive integers
    """
    assert isinstance(MIN_ROWS_DATASET, int)   and MIN_ROWS_DATASET > 0
    assert isinstance(MIN_PARTITION_ROWS, int) and MIN_PARTITION_ROWS > 0
    assert MIN_PARTITION_ROWS <= int(MIN_ROWS_DATASET * 0.15), (
        f"MIN_PARTITION_ROWS={MIN_PARTITION_ROWS} > val size at minimum "
        f"({int(MIN_ROWS_DATASET * 0.15)}) — the guard is internally inconsistent"
    )
    print(
        f"PASS  test_constants_are_reasonable  "
        f"(MIN_ROWS_DATASET={MIN_ROWS_DATASET}, "
        f"MIN_PARTITION_ROWS={MIN_PARTITION_ROWS}, "
        f"val@min≈{int(MIN_ROWS_DATASET * 0.15)})"
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_constants_are_reasonable,
        test_guard1_20_rows_raises,
        test_guard1_one_below_minimum_raises,
        test_guard1_does_not_fire_at_minimum,
        test_guard2_fires_when_fe_shrinks_partitions,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            print(f"FAIL  {t.__name__}:  {exc}")
            failed += 1
        except Exception as exc:
            import traceback
            print(f"ERROR {t.__name__}:  {exc}")
            traceback.print_exc()
            failed += 1

    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"RESULT: {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
