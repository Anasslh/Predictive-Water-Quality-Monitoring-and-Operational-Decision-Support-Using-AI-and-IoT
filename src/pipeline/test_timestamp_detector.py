"""
test_timestamp_detector.py — Unit tests for detect_timestamp_column() and
parse_timestamp_column() with focus on ISO-strict parsing and day/month
ambiguity handling.

Run:  python src/pipeline/test_timestamp_detector.py
      (or via run_tests.sh — the file is picked up automatically)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.pipeline.timestamp_detector import detect_timestamp_column, parse_timestamp_column

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    symbol = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{symbol}  {label}{suffix}")
    _results.append((label, condition))


# ── test cases ────────────────────────────────────────────────────────────────

def test_iso_date_only() -> None:
    """YYYY-MM-DD (the C-1 format) is accepted without inference warning."""
    df = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02", "2026-01-03"], "v": [1, 2, 3]})
    col = detect_timestamp_column(df, ["Date"])
    check("ISO date-only YYYY-MM-DD accepted", col == "Date")


def test_iso_datetime_with_T() -> None:
    """Full ISO 8601 with T separator is accepted."""
    df = pd.DataFrame({
        "ts": ["2026-01-01T08:00:00", "2026-01-02T08:00:00", "2026-01-03T08:00:00"],
    })
    col = detect_timestamp_column(df, ["ts"])
    check("ISO datetime with T separator accepted", col == "ts")


def test_iso_datetime_with_space() -> None:
    """YYYY-MM-DD HH:MM:SS (space separator) is accepted."""
    df = pd.DataFrame({
        "ts": ["2026-01-01 08:00:00", "2026-01-02 09:00:00", "2026-01-03 10:00:00"],
    })
    col = detect_timestamp_column(df, ["ts"])
    check("ISO datetime space-separated accepted", col == "ts")


def test_already_datetime64() -> None:
    """A column already typed datetime64 is accepted directly."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "v": [1, 2, 3],
    })
    col = detect_timestamp_column(df, ["Date"])
    check("Already-typed datetime64 accepted", col == "Date")


def test_non_monotone_rejected() -> None:
    """A column that parses but is not monotonically increasing is rejected."""
    df = pd.DataFrame({
        "Date": ["2026-01-03", "2026-01-01", "2026-01-02"],
        "v": [1, 2, 3],
    })
    try:
        detect_timestamp_column(df, ["Date"])
        check("Non-monotone column raises ValueError", False, "no exception raised")
    except ValueError as exc:
        check(
            "Non-monotone column raises ValueError",
            "monoton" in str(exc).lower(),
            str(exc)[:60],
        )


def test_column_not_in_df() -> None:
    """A candidate column absent from the DataFrame is skipped gracefully."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    try:
        detect_timestamp_column(df, ["Date", "timestamp"])
        check("Missing column raises ValueError", False, "no exception raised")
    except ValueError as exc:
        check("Missing column raises ValueError", "not found" in str(exc).lower())


def test_fallback_to_second_candidate() -> None:
    """If first candidate is absent, second valid candidate is returned."""
    df = pd.DataFrame({
        "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "v": [1, 2, 3],
    })
    col = detect_timestamp_column(df, ["Date", "timestamp"])
    check("Falls back to second candidate when first absent", col == "timestamp")


def test_ambiguous_ddmmyyyy_triggers_warning() -> None:
    """
    DD/MM/YYYY format with days ≤ 12 is ambiguous (e.g. 05/07/2026 could be
    5 July or 7 May). The new code falls through to inference (since it doesn't
    match any ISO format) and emits a WARNING.

    We verify:
      (a) No exception is raised — the column IS accepted.
      (b) A WARNING was logged with "automatic inference" in the message.
    """
    dates = [
        "05/07/2026",
        "06/07/2026",
        "07/07/2026",
    ]
    df = pd.DataFrame({"Date": dates, "v": [1, 2, 3]})

    warning_messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warning_messages.append(record.getMessage())

    handler = _Capture()
    ts_logger = logging.getLogger("src.pipeline.timestamp_detector")
    ts_logger.addHandler(handler)
    ts_logger.setLevel(logging.WARNING)

    try:
        col = detect_timestamp_column(df, ["Date"])
        accepted = col == "Date"
    except ValueError:
        accepted = False
    finally:
        ts_logger.removeHandler(handler)

    check(
        "DD/MM/YYYY column accepted (inference fallback)",
        accepted,
        "should accept but log warning",
    )
    warned = any("automatic inference" in m for m in warning_messages)
    check(
        "DD/MM/YYYY triggers 'automatic inference' WARNING",
        warned,
        f"warnings captured: {warning_messages}",
    )


def test_parse_iso_correct_date() -> None:
    """parse_timestamp_column() on ISO dates returns correct datetime values."""
    df = pd.DataFrame({
        "Date": ["2026-07-05", "2026-07-06", "2026-07-07"],
        "v": [10, 20, 30],
    })
    result = parse_timestamp_column(df, "Date")
    expected_first = pd.Timestamp("2026-07-05")
    check(
        "parse_timestamp_column ISO: first date is 2026-07-05",
        result["Date"].iloc[0] == expected_first,
        f"got {result['Date'].iloc[0]}",
    )


def test_parse_ambiguous_ddmmyyyy_warns() -> None:
    """
    parse_timestamp_column() on DD/MM/YYYY also emits the inference warning.
    The parsed result may be month/day-swapped (pandas behaviour) but the
    warning makes it non-silent.
    """
    df = pd.DataFrame({
        "Date": ["05/07/2026", "06/07/2026", "07/07/2026"],
        "v": [1, 2, 3],
    })

    warning_messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warning_messages.append(record.getMessage())

    handler = _Capture()
    ts_logger = logging.getLogger("src.pipeline.timestamp_detector")
    ts_logger.addHandler(handler)
    ts_logger.setLevel(logging.WARNING)

    try:
        parse_timestamp_column(df, "Date")
        raised = False
    except Exception:
        raised = True
    finally:
        ts_logger.removeHandler(handler)

    warned = any("automatic inference" in m for m in warning_messages)
    check(
        "parse_timestamp_column DD/MM/YYYY: no exception + warning emitted",
        not raised and warned,
        f"raised={raised}  warned={warned}",
    )


def test_unparseable_column_raises() -> None:
    """A column with garbage values raises ValueError (not silently NaT)."""
    df = pd.DataFrame({
        "Date": ["not-a-date", "also-not", "nope"],
        "v": [1, 2, 3],
    })
    try:
        detect_timestamp_column(df, ["Date"])
        check("Unparseable column raises ValueError", False, "no exception raised")
    except ValueError as exc:
        check("Unparseable column raises ValueError", "parse" in str(exc).lower())


def test_iso_datetime_with_fractional_seconds_and_timezone() -> None:
    """
    ISO timestamps with sub-second precision and UTC offset (e.g.
    '2026-05-25 16:09:38.503752912+00:00') must be recognised as strict ISO
    without triggering the 'automatic inference / ambiguous day/month' warning.

    Verifies:
      (a) The column is accepted.
      (b) No 'automatic inference' warning is emitted.
      (c) The returned series is timezone-naive (UTC-normalised) so it remains
          compatible with the rest of the pipeline.
    """
    dates = [
        "2026-05-25 16:09:38.503752912+00:00",
        "2026-05-26 08:22:11.123456789+00:00",
        "2026-05-27 14:55:00.000000001+00:00",
    ]
    df = pd.DataFrame({"ts": dates, "v": [1, 2, 3]})

    warning_messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warning_messages.append(record.getMessage())

    handler = _Capture()
    ts_logger = logging.getLogger("src.pipeline.timestamp_detector")
    ts_logger.addHandler(handler)
    ts_logger.setLevel(logging.WARNING)

    try:
        col = detect_timestamp_column(df, ["ts"])
        accepted = col == "ts"
    except ValueError:
        accepted = False
    finally:
        ts_logger.removeHandler(handler)

    check(
        "ISO datetime with sub-seconds + timezone accepted",
        accepted,
        "column should be detected without exception",
    )
    check(
        "No 'automatic inference' warning for unambiguous ISO+TZ format",
        not any("automatic inference" in m for m in warning_messages),
        f"warnings: {warning_messages}" if warning_messages else "no warnings (correct)",
    )

    # Also verify parse_timestamp_column returns tz-naive timestamps
    if accepted:
        result = parse_timestamp_column(df, "ts")
        tz_naive = not hasattr(result["ts"].dtype, "tz") or result["ts"].dt.tz is None
        check(
            "parse_timestamp_column returns tz-naive (UTC-normalised) series",
            tz_naive,
            f"dtype={result['ts'].dtype}",
        )


# ── runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  timestamp_detector — ISO-strict + ambiguity-warning tests")
    print(SEP2)

    test_iso_date_only()
    test_iso_datetime_with_T()
    test_iso_datetime_with_space()
    test_already_datetime64()
    test_non_monotone_rejected()
    test_column_not_in_df()
    test_fallback_to_second_candidate()
    test_ambiguous_ddmmyyyy_triggers_warning()
    test_parse_iso_correct_date()
    test_parse_ambiguous_ddmmyyyy_warns()
    test_unparseable_column_raises()
    test_iso_datetime_with_fractional_seconds_and_timezone()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
