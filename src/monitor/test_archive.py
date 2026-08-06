"""
test_archive.py — Tests for src/monitor/archive.py

Covers:
  1. Single-month append: records accumulate in the correct plain .jsonl file.
  2. Month rotation: a file for a past month is compressed to .jsonl.gz and
     the original removed.
  3. read_archive() spans compressed + plain files correctly.
  4. read_archive() respects start_date / end_date filters.
  5. Hot export (exports/) keeps purging at 30 days while the archive grows.
  6. Volume estimation: records-per-month and compressed file size.
  7. list_archive_files() returns correct metadata.
  8. Idempotency: rotation called twice does not re-compress an already-gz file.

Run:
    python src/monitor/test_archive.py
    (or via run_tests.sh — picked up automatically)
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.monitor import MonitorResult
from src.monitor.archive import (
    append_to_archive,
    export_archive_range,
    list_archive_files,
    read_archive,
    _rotate_old_months,
    _month_tag,
    _plain_path,
    _gz_path,
)
from src.monitor.export import append_measurement, read_export

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    sym = PASS if condition else FAIL
    suf = f"  ({detail})" if detail else ""
    print(f"  {sym}  {label}{suf}")
    _results.append((label, condition))


# ── Month helpers (relative to real current date) ─────────────────────────────

def _months_ago(n: int) -> tuple[int, int]:
    """Return (year, month) for the calendar month that is n months before now."""
    now = datetime.now(timezone.utc)
    m = now.month - n
    y = now.year
    while m <= 0:
        m += 12
        y -= 1
    return y, m


def _month_str(n: int) -> str:
    """'YYYY-MM' string for the month that is n months before now."""
    y, m = _months_ago(n)
    return f"{y:04d}-{m:02d}"


def _month_first_day(n: int) -> datetime:
    """First day of the month that is n months before now (UTC midnight)."""
    y, m = _months_ago(n)
    return datetime(y, m, 1, tzinfo=timezone.utc)


def _days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _fake_result(ts: datetime, predicted: float = 200.0) -> MonitorResult:
    """Build a minimal MonitorResult with a specific timestamp."""
    r = MonitorResult(
        parameter_name   = "EC",
        timestamp        = ts,
        prediction       = predicted,
        prediction_shap  = [{"feature": "EC_lag1", "shap_value": 0.5}],
        residual         = 5.0,
        anomaly_score    = 0.1,
        anomaly_detected = False,
        anomaly_shap     = None,
        retrain_needed   = False,
        retrain_result   = None,
        retrain_alert    = None,
        forecast         = None,
        processed_at     = ts,
    )
    return r


# ── Test 1: Single-month plain file ───────────────────────────────────────────

def test_single_month_append() -> None:
    print("\n  [1] Single-month append")
    with tempfile.TemporaryDirectory() as tmp:
        d    = Path(tmp)
        # Use the CURRENT calendar month so rotation never triggers
        cur  = _month_str(0)
        base = _month_first_day(0)

        for i in range(5):
            r = _fake_result(base + timedelta(days=i))
            append_to_archive("EC", r, archive_dir=d)

        plain = _plain_path("EC", d, cur)
        check("plain .jsonl file created",          plain.exists())
        lines = [ln for ln in plain.read_text().splitlines() if ln.strip()]
        check("5 records written",                  len(lines) == 5, f"got {len(lines)}")
        gz    = _gz_path("EC", d, cur)
        check("no premature compression",            not gz.exists())

        obj = json.loads(lines[0])
        check("record has 'timestamp' field",       "timestamp" in obj)
        check("record has 'predicted_value' field", "predicted_value" in obj)
        check("record has 'is_anomaly' field",      "is_anomaly" in obj)


# ── Test 2: Month rotation (compression) ─────────────────────────────────────

def test_month_rotation() -> None:
    print("\n  [2] Month rotation / compression")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # Use 2 months ago so the file is always a past month regardless of run date
        old_month  = _month_str(2)
        cur_month  = _month_str(0)
        y_old, m_old = _months_ago(2)
        old_base   = datetime(y_old, m_old, 1, tzinfo=timezone.utc)

        plain_old = _plain_path("EC", d, old_month)
        for i in range(3):
            ts   = old_base + timedelta(days=i)
            line = json.dumps({
                "timestamp":        ts.isoformat(),
                "predicted_value":  200 + i,
                "actual_value":     205 + i,
                "is_anomaly":       False,
                "anomaly_score":    0.1,
                "shap_top_features": [],
                "retrain_alert":    None,
                "parameter_name":   "EC",
            })
            with open(plain_old, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        check(f"Old-month ({old_month}) plain file exists before rotation",
              plain_old.exists())

        # Write a current-month record — rotation should compress the old-month file
        cur_ts = _month_first_day(0)
        append_to_archive("EC", _fake_result(cur_ts), archive_dir=d)

        gz_old    = _gz_path("EC", d, old_month)
        plain_cur = _plain_path("EC", d, cur_month)
        check(f"Old-month ({old_month}) file compressed after new write", gz_old.exists())
        check("Old-month plain file removed after rotation",              not plain_old.exists())
        check(f"Current-month ({cur_month}) plain file created",          plain_cur.exists())

        # Verify compressed content is readable and contains 3 records
        content = gzip.decompress(gz_old.read_bytes()).decode("utf-8")
        lines   = [ln for ln in content.splitlines() if ln.strip()]
        check("Compressed old-month file has 3 records",
              len(lines) == 3, f"got {len(lines)}")


# ── Test 3: read_archive across compressed + plain ───────────────────────────

def test_read_archive_cross_month() -> None:
    print("\n  [3] read_archive() spans compressed + plain files")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # Use 2 months ago as the compressed past month
        old_month    = _month_str(2)
        cur_month    = _month_str(0)
        y_old, m_old = _months_ago(2)
        y_cur, m_cur = _months_ago(0)
        old_base     = datetime(y_old, m_old, 1, tzinfo=timezone.utc)

        # Build old-month plain file with 4 records (written directly)
        plain_old = _plain_path("EC", d, old_month)
        for i in range(4):
            ts  = old_base + timedelta(days=i)
            rec = {"timestamp": ts.isoformat(), "parameter_name": "EC",
                   "predicted_value": float(200 + i), "actual_value": float(205 + i),
                   "shap_top_features": [], "is_anomaly": False,
                   "anomaly_score": 0.0, "retrain_alert": None}
            with open(plain_old, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        # Manually compress old-month file (simulate past rotation)
        gz_old = _gz_path("EC", d, old_month)
        gz_old.write_bytes(gzip.compress(plain_old.read_bytes(), compresslevel=9))
        plain_old.unlink()

        # Write a current-month record via append_to_archive
        cur_ts = _month_first_day(0)
        append_to_archive("EC", _fake_result(cur_ts, predicted=210.0), archive_dir=d)

        # Confirm state: old gz + current plain
        check(f"Old month ({old_month}) .gz exists",      gz_old.exists())
        check(f"Current month ({cur_month}) .jsonl exists",
              _plain_path("EC", d, cur_month).exists())

        # read_archive with no filter: should return 5 records (4 old + 1 current)
        all_recs = read_archive("EC", archive_dir=d)
        check("read_archive() returns 5 records total",
              len(all_recs) == 5, f"got {len(all_recs)}")

        # Filter: only old-month records
        old_start = f"{y_old:04d}-{m_old:02d}-01"
        days_old  = _days_in_month(y_old, m_old)
        old_end   = f"{y_old:04d}-{m_old:02d}-{days_old:02d}T23:59:59Z"
        old_recs  = read_archive("EC", archive_dir=d, start_date=old_start, end_date=old_end)
        check("Date-filtered read returns 4 old records",
              len(old_recs) == 4, f"got {len(old_recs)}")

        # Filter: only current-month records
        cur_start = f"{y_cur:04d}-{m_cur:02d}-01"
        cur_recs  = read_archive("EC", archive_dir=d, start_date=cur_start)
        check("Date-filtered read returns 1 current record",
              len(cur_recs) == 1, f"got {len(cur_recs)}")


# ── Test 4: Hot export purges / archive does not ──────────────────────────────

def test_hot_purges_archive_grows() -> None:
    print("\n  [4] Hot export purges at 30 days; archive never purges")
    with tempfile.TemporaryDirectory() as exports_tmp:
        with tempfile.TemporaryDirectory() as archive_tmp:
            exports_dir = Path(exports_tmp)
            archive_dir = Path(archive_tmp)
            param       = "EC"
            now         = datetime.now(timezone.utc)
            n_days      = 45   # simulate 45 daily measurements

            for i in range(n_days):
                ts = now - timedelta(days=n_days - 1 - i)
                r  = _fake_result(ts)
                # Hot export (30-day retention)
                append_measurement(param, r, exports_dir=exports_dir, retention_days=30)
                # Cold archive (no purge)
                append_to_archive(param, r, archive_dir=archive_dir)

            hot_records     = read_export(param, exports_dir=exports_dir, retention_days=30)
            archive_records = read_archive(param, archive_dir=archive_dir)

            check("Hot export has ≤ 31 records (30-day window)",
                  len(hot_records) <= 31, f"got {len(hot_records)}")
            check("Hot export has fewer records than total simulated",
                  len(hot_records) < n_days, f"hot={len(hot_records)} total={n_days}")
            check("Archive has all 45 records (no purge)",
                  len(archive_records) == n_days, f"got {len(archive_records)}")


# ── Test 5: Volume estimation ─────────────────────────────────────────────────

def test_volume_estimation() -> None:
    print("\n  [5] Volume estimation (3 simulated months)")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        param = "EC"
        # Simulate 3 months of daily measurements (Jan, Feb, Mar 2026)
        # with rich SHAP to approximate real record size
        months = [
            ("2026-01", 31),
            ("2026-02", 28),
            ("2026-03", 31),
        ]
        for month, n_days in months:
            year, mo = map(int, month.split("-"))
            for day in range(1, n_days + 1):
                ts  = datetime(year, mo, day, 12, 0, tzinfo=timezone.utc)
                r   = MonitorResult(
                    parameter_name  = param,
                    timestamp       = ts,
                    prediction      = 215.5,
                    prediction_shap = [
                        {"feature": "EC_lag1",       "shap_value": 0.45},
                        {"feature": "EC_roll3_mean", "shap_value": 0.22},
                        {"feature": "pH_lag1",       "shap_value": -0.10},
                    ],
                    residual        = 3.2,
                    anomaly_score   = 0.05,
                    anomaly_detected = False,
                    anomaly_shap    = None,
                    retrain_needed  = False,
                    retrain_result  = None,
                    retrain_alert   = None,
                    forecast        = None,
                    processed_at    = ts,
                )
                plain = _plain_path(param, d, month)
                line  = json.dumps({
                    "timestamp":         ts.isoformat(),
                    "parameter_name":    param,
                    "predicted_value":   r.prediction,
                    "actual_value":      r.prediction + (r.residual or 0),
                    "shap_top_features": r.prediction_shap,
                    "is_anomaly":        r.anomaly_detected,
                    "anomaly_score":     r.anomaly_score,
                    "retrain_alert":     r.retrain_alert,
                })
                with open(plain, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

        # Compress Jan and Feb (past months), leave March plain
        import gzip as _gzip
        for month in ("2026-01", "2026-02"):
            plain = _plain_path(param, d, month)
            data  = plain.read_bytes()
            gz    = _gz_path(param, d, month)
            gz.write_bytes(_gzip.compress(data, compresslevel=9))
            plain.unlink()

        files = list_archive_files(param, archive_dir=d)
        check("3 archive files listed", len(files) == 3, str([f["month"] for f in files]))

        total_gz_bytes   = sum(f["size_bytes"] for f in files if f["compressed"])
        total_plain_bytes= sum(f["size_bytes"] for f in files if not f["compressed"])
        total_bytes      = total_gz_bytes + total_plain_bytes
        total_records    = sum(m[1] for m in months)

        bytes_per_record = total_bytes / total_records
        yearly_daily_est = bytes_per_record * 365 / 1024          # KB/year daily
        yearly_hourly_est= bytes_per_record * 365 * 24 / (1024**3) # GB/year hourly

        print(f"    Records simulated    : {total_records}")
        print(f"    Total size (mixed)   : {total_bytes:,} bytes  "
              f"({total_bytes/1024:.1f} KB)")
        print(f"    Compressed Jan+Feb   : {total_gz_bytes:,} bytes")
        print(f"    Plain March          : {total_plain_bytes:,} bytes")
        print(f"    Bytes per record     : {bytes_per_record:.0f} B")
        print(f"    Yearly estimate @ 1/day  : {yearly_daily_est:.1f} KB/year")
        print(f"    Yearly estimate @ 1/hour : {yearly_hourly_est:.3f} GB/year")

        check("Bytes per record < 500 B (efficient JSON)",
              bytes_per_record < 500, f"got {bytes_per_record:.0f} B")
        check("Yearly @ hourly < 5 GB (reasonable estimate)",
              yearly_hourly_est < 5.0, f"got {yearly_hourly_est:.3f} GB")


# ── Test 6: list_archive_files metadata ──────────────────────────────────────

def test_list_archive_files() -> None:
    print("\n  [6] list_archive_files() metadata")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Create one gz file and one plain file
        gz_path   = _gz_path("pH", d, "2026-05")
        plain_path = _plain_path("pH", d, "2026-06")
        gz_path.write_bytes(gzip.compress(b'{"timestamp":"2026-05-01T00:00:00+00:00"}\n'))
        plain_path.write_text('{"timestamp":"2026-06-01T00:00:00+00:00"}\n', encoding="utf-8")

        files = list_archive_files("pH", archive_dir=d)
        check("2 files listed",                    len(files) == 2, f"got {len(files)}")
        check("Newest first (2026-06 before 05)",  files[0]["month"] == "2026-06")
        check("May file is compressed",             files[1]["compressed"] is True)
        check("June file is not compressed",        files[0]["compressed"] is False)


# ── Test 7: Idempotency of rotation ──────────────────────────────────────────

def test_rotation_idempotency() -> None:
    print("\n  [7] Rotation idempotency")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Create a pre-compressed June file
        gz = _gz_path("EC", d, "2026-06")
        gz.write_bytes(gzip.compress(b'{"timestamp":"2026-06-01T00:00:00+00:00"}\n'))

        size_before = gz.stat().st_size
        # Call rotation multiple times — should not re-compress
        _rotate_old_months("EC", d)
        _rotate_old_months("EC", d)
        size_after = gz.stat().st_size

        check("Second rotation leaves .gz unchanged", size_before == size_after)
        plain = _plain_path("EC", d, "2026-06")
        check("No plain file created by idempotent rotation", not plain.exists())


# ── Test 8: export_archive_range ─────────────────────────────────────────────

def test_export_archive_range() -> None:
    print("\n  [8] export_archive_range()")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # Build 3 months: May + June compressed, July plain
        months_data = [
            ("2026-05", 31),
            ("2026-06", 30),
            ("2026-07", 31),
        ]
        for month, n_days in months_data:
            year, mo = map(int, month.split("-"))
            lines = []
            for day in range(1, n_days + 1):
                ts = datetime(year, mo, day, 12, 0, tzinfo=timezone.utc)
                lines.append(json.dumps({
                    "timestamp":         ts.isoformat(),
                    "parameter_name":    "EC",
                    "predicted_value":   200.0 + day,
                    "actual_value":      202.0 + day,
                    "shap_top_features": [{"feature": "EC_lag1", "shap_value": 0.4}],
                    "is_anomaly":        False,
                    "anomaly_score":     0.05,
                    "retrain_alert":     None,
                }))
            plain = _plain_path("EC", d, month)
            plain.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Compress May and June (past months)
        for month in ("2026-05", "2026-06"):
            plain = _plain_path("EC", d, month)
            gz    = _gz_path("EC", d, month)
            gz.write_bytes(gzip.compress(plain.read_bytes(), compresslevel=9))
            plain.unlink()

        check("[export] May and June are compressed",
              _gz_path("EC", d, "2026-05").exists() and
              _gz_path("EC", d, "2026-06").exists())
        check("[export] July is plain",
              _plain_path("EC", d, "2026-07").exists())

        out = d / "output" / "EC_3months.jsonl"

        # Export May–July (spans 2 compressed + 1 plain)
        n = export_archive_range(
            "EC",
            start_date  = "2026-05-01",
            end_date    = "2026-07-31T23:59:59Z",
            output_path = out,
            archive_dir = d,
        )
        total_days = 31 + 30 + 31
        check("[export] Returns correct record count",
              n == total_days, f"got {n}, expected {total_days}")
        check("[export] Output file exists",
              out.exists())

        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        check("[export] Output has correct line count",
              len(lines) == total_days, f"got {len(lines)}")

        # First record is oldest (2026-05-01)
        first = json.loads(lines[0])
        check("[export] First record is from May 2026",
              first["timestamp"].startswith("2026-05"))
        # Last record is newest (2026-07-31)
        last = json.loads(lines[-1])
        check("[export] Last record is from July 2026",
              last["timestamp"].startswith("2026-07"))

        # Date filter: June only
        out2 = d / "output" / "EC_june.jsonl"
        n2 = export_archive_range(
            "EC",
            start_date  = "2026-06-01",
            end_date    = "2026-06-30T23:59:59Z",
            output_path = out2,
            archive_dir = d,
        )
        check("[export] June-only export has 30 records",
              n2 == 30, f"got {n2}")

        # ValueError when no records match
        raised = False
        try:
            export_archive_range(
                "EC",
                start_date  = "2024-01-01",
                end_date    = "2024-01-31",
                output_path = d / "output" / "empty.jsonl",
                archive_dir = d,
            )
        except ValueError:
            raised = True
        check("[export] ValueError raised when no records found", raised)

        # Atomic write: no .tmp file left after success
        tmp_files = list((d / "output").glob("*.tmp"))
        check("[export] No stray .tmp files after export",
              len(tmp_files) == 0, f"found {tmp_files}")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("═" * 62)
    print("  archive — tests")
    print("═" * 62)

    test_single_month_append()
    test_month_rotation()
    test_read_archive_cross_month()
    test_hot_purges_archive_grows()
    test_volume_estimation()
    test_list_archive_files()
    test_rotation_idempotency()
    test_export_archive_range()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(f"\n{'─' * 62}")
    print(f"  {passed}/{total} passed.")
    print("═" * 62)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
