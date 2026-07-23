"""
test_export.py — Regression tests for src/monitor/export.py

Simulates 40 days of daily measurements and verifies the 30-day
rolling-window constraint:
  - No record older than 30 days is ever present after an append.
  - The purge is timestamp-based (field inside the JSON line), not
    based on file size or line count.
  - The oldest eligible records (days 0-8, i.e. 31-39 days ago) are
    gone; the most recent 30+ days are intact.

Run:
    python -m src.monitor.test_export
    # or
    python src/monitor/test_export.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Path bootstrap (works whether run as module or as script) ──────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]   # src/monitor/test_export.py → project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.monitor import MonitorResult
from src.monitor.export import append_measurement, read_export, _export_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_result(day_offset: int, now: datetime) -> MonitorResult:
    """
    Build a fake MonitorResult whose timestamp is (now - day_offset).

    day_offset=0  → today (most recent)
    day_offset=39 → 39 days ago (oldest in a 40-day simulation)
    """
    ts = now - timedelta(days=day_offset)
    return MonitorResult(
        parameter_name   = "EC",
        timestamp        = ts,
        prediction       = 300.0 + day_offset,
        prediction_shap  = [
            {"feature": "EC_lag1", "shap_value": round(0.5 - day_offset * 0.01, 3),
             "feature_value": 290.0}
        ],
        residual         = 5.0 - day_offset * 0.1,
        anomaly_score    = 0.15 if day_offset % 7 != 0 else 0.72,
        anomaly_detected = day_offset % 7 == 0,
        retrain_alert    = None,
    )


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


# ── Test 1: 30-day invariant holds after every single append ──────────────────

def test_window_invariant_on_every_write():
    """
    After each of the 40 daily appends, the file must contain zero records
    whose timestamp predates the 30-day cutoff.
    """
    RETENTION = 30

    with tempfile.TemporaryDirectory() as tmpdir:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=RETENTION)

        for i in range(40):
            # day_offset counts DOWN: i=0 → 39 days ago, i=39 → today
            day_offset = 39 - i
            result = _fake_result(day_offset, now)
            append_measurement("EC", result, exports_dir=tmpdir, retention_days=RETENTION)

            # After every write: verify invariant
            records = read_export("EC", exports_dir=tmpdir, retention_days=RETENTION)
            for rec in records:
                ts = datetime.fromisoformat(rec["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                assert ts >= cutoff, (
                    f"After write {i+1}/40: stale record found  ts={ts}  cutoff={cutoff}"
                )

    print("PASS  test_window_invariant_on_every_write  (40 writes, invariant held each time)")


# ── Test 2: Final content after 40 writes ─────────────────────────────────────

def test_final_content_after_40_days():
    """
    After 40 daily writes (oldest = 39 days ago, newest = today):
      - records with ts < cutoff are purged  → offsets 31-39 (definitely > 30 days old)
      - records with ts >= cutoff are kept   → offsets 0-29 at minimum (definitely within 30d)
      - offset 30 (exactly = cutoff) may be kept or purged depending on sub-millisecond
        timing between timestamp creation and cutoff evaluation; both are valid
      - expected kept count: 30 or 31 (boundary record is on the edge)

    The hard invariants are:
      1. NO record older than 30 days appears in the export.
      2. Offsets 31-39 are definitely purged.
      3. Offsets 0-29 are definitely present.
      4. File line count == read_export count (file is clean, no blank/stale lines).
    """
    RETENTION = 30

    with tempfile.TemporaryDirectory() as tmpdir:
        now = datetime.now(timezone.utc)

        for i in range(40):
            day_offset = 39 - i
            append_measurement(
                "EC", _fake_result(day_offset, now),
                exports_dir=tmpdir, retention_days=RETENTION
            )

        records = read_export("EC", exports_dir=tmpdir, retention_days=RETENTION)
        path    = _export_path("EC", Path(tmpdir))
        n_lines = _count_lines(path)

        # Invariant 1: all returned records must be within the window
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION)
        for rec in records:
            ts = datetime.fromisoformat(rec["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            assert ts >= cutoff, f"Stale record in final export: ts={ts}"

        kept_preds = {rec["predicted_value"] for rec in records}

        # Invariant 2: offsets 31-39 (definitely > 30 days old) must NOT be present
        for offset in range(31, 40):
            expected_pred = 300.0 + offset
            assert expected_pred not in kept_preds, (
                f"Record from offset={offset} days ago should be purged "
                f"but predicted_value={expected_pred} found in export"
            )

        # Invariant 3: offsets 0-29 (definitely within 30 days) must be present
        for offset in range(0, 30):
            expected_pred = 300.0 + offset
            assert expected_pred in kept_preds, (
                f"Record from offset={offset} days ago should be kept "
                f"but predicted_value={expected_pred} not found in export"
            )

        # Invariant 4: file is clean (no blank / stale lines)
        assert n_lines == len(records), (
            f"File has {n_lines} lines but read_export returned {len(records)} records"
        )

        # Count is 30 (offset-30 purged due to sub-ms drift) or 31 (offset-30 kept)
        assert 30 <= len(records) <= 31, (
            f"Expected 30 or 31 records, got {len(records)}"
        )

        n_purged = 40 - len(records)

    print(
        f"PASS  test_final_content_after_40_days  "
        f"({len(records)} records kept, {n_purged} purged — "
        "purge is timestamp-based, not line-count-based)"
    )


# ── Test 3: read_export without prior writes returns empty list ────────────────

def test_read_export_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = read_export("EC", exports_dir=tmpdir)
        assert result == [], f"Expected [], got {result}"
    print("PASS  test_read_export_empty")


# ── Test 4: multiple parameters are independent ───────────────────────────────

def test_multi_parameter_isolation():
    """
    Writes for "EC" must not appear in "pH" export and vice-versa.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        now = datetime.now(timezone.utc)
        for param in ("EC", "pH"):
            r = MonitorResult(
                parameter_name=param,
                timestamp=now,
                prediction=1.0 if param == "EC" else 7.0,
                prediction_shap=[],
                residual=0.1,
            )
            append_measurement(param, r, exports_dir=tmpdir)

        ec_records  = read_export("EC",  exports_dir=tmpdir)
        ph_records  = read_export("pH",  exports_dir=tmpdir)
        all_records = read_export("Turbidity", exports_dir=tmpdir)

        assert len(ec_records)  == 1
        assert len(ph_records)  == 1
        assert len(all_records) == 0
        assert ec_records[0]["predicted_value"] == 1.0
        assert ph_records[0]["predicted_value"] == 7.0

    print("PASS  test_multi_parameter_isolation")


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_window_invariant_on_every_write,
        test_final_content_after_40_days,
        test_read_export_empty,
        test_multi_parameter_isolation,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            print(f"FAIL  {t.__name__}:  {exc}")
            failed += 1
        except Exception as exc:
            print(f"ERROR {t.__name__}:  {exc}")
            failed += 1

    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"RESULT: {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
