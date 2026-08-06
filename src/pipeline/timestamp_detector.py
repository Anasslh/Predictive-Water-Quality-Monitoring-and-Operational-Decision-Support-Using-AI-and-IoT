"""
timestamp_detector.py — Identify the timestamp column in an unknown DataFrame.

STRATEGY
--------
Try each candidate column name in order. A column qualifies if:
  1. It exists in the DataFrame.
  2. Its values parse as datetimes (see _try_parse below for format logic).
  3. After parsing, the values are strictly monotonically increasing.

The function returns the FIRST candidate that passes all three tests.

PARSING LOGIC (_try_parse)
--------------------------
To avoid the silent day/month inversion risk of unconstrained pd.to_datetime(),
parsing follows a two-step protocol:

  Step 1 — Strict ISO formats tried first (in order):
      "%Y-%m-%dT%H:%M:%S"  full ISO 8601 with time
      "%Y-%m-%d %H:%M:%S"  ISO with space separator
      "%Y-%m-%d"           date-only ISO (the recommended format, used in C-1)

      These are unambiguous: year-first, day/month never swapped.
      If any succeeds with zero NaT values, it is used silently.

  Step 2 — Automatic pandas inference as fallback:
      If none of the ISO formats work, pd.to_datetime(col) is tried without
      a format constraint. If it produces zero NaT values, it is accepted BUT
      a WARNING is logged explicitly:

          "timestamp column '<col>' parsed using automatic inference —
           ambiguous day/month formats (e.g. 05/07/2026) may be misread as
           month/day without warning; ISO format YYYY-MM-DD is strongly
           recommended"

      This ensures the fallback is never silent.

  Failure — clear error:
      If no candidate column passes all three checks, ValueError is raised
      listing each candidate and the exact reason it was rejected.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)

# ISO formats tried in order — unambiguous, year first.
# Variants with %z (timezone offset) come first so they match before the
# timezone-naive equivalents.  %f matches 1–9 digit sub-second fractions
# (pandas internal handling), so "2026-05-25 16:09:38.503752912+00:00" is
# recognised as strict ISO without falling through to the inference path.
_ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",   # ISO 8601: T-sep, sub-seconds, timezone offset
    "%Y-%m-%d %H:%M:%S.%f%z",   # same with space separator
    "%Y-%m-%dT%H:%M:%S%z",      # ISO 8601: T-sep, no sub-seconds, timezone offset
    "%Y-%m-%d %H:%M:%S%z",      # same with space separator
    "%Y-%m-%dT%H:%M:%S",        # ISO 8601: T-sep, no timezone
    "%Y-%m-%d %H:%M:%S",        # ISO with space separator, no timezone
    "%Y-%m-%d",                  # date-only ISO (C-1 dataset format)
)


def _try_parse(col: pd.Series) -> tuple[pd.Series, bool]:
    """
    Attempt to parse a Series as datetime, ISO-first then fallback.

    Returns
    -------
    (parsed, used_inference)
        parsed        : datetime64 Series (no NaT values on success).
        used_inference: True if the automatic-inference fallback was used.

    Raises
    ------
    ValueError  if every strategy produces at least one NaT or a parse error.
    """
    # Already datetime — accept without re-parsing
    if pd.api.types.is_datetime64_any_dtype(col):
        return col, False

    # Step 1: try each ISO format with strict matching
    for fmt in _ISO_FORMATS:
        try:
            parsed = pd.to_datetime(col, format=fmt, errors="coerce")
        except Exception:
            continue
        if parsed.notna().all():
            # Formats with %z produce tz-aware series.  Normalise to UTC-naive
            # so downstream feature engineering (shift, diff, merge) stays
            # compatible with the rest of the pipeline which expects naive timestamps.
            if isinstance(parsed.dtype, pd.DatetimeTZDtype):
                parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
            return parsed, False

    # Step 2: pandas automatic inference as fallback.
    # Suppress the pandas UserWarning about dateutil fallback — we emit our
    # own explicit logger.warning() at the call site instead.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(col, errors="coerce")
    except Exception as exc:
        raise ValueError(f"automatic inference also failed: {exc}") from exc

    if not parsed.notna().all():
        n_nat = int(parsed.isna().sum())
        bad = col[parsed.isna()].head(5).tolist()
        raise ValueError(
            f"{n_nat} value(s) could not be parsed as datetime "
            f"(first bad values: {bad})"
        )

    return parsed, True   # succeeded via inference


def detect_timestamp_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """
    Return the name of the first column that is a valid, ordered timestamp.

    Parameters
    ----------
    df         : Raw DataFrame to inspect.
    candidates : Ordered list of column names to try (from sensors_config.json's
                 "timestamp_column_candidates" field).

    Returns
    -------
    str  — name of the detected timestamp column.

    Raises
    ------
    ValueError  if no candidate passes all three checks, with a per-candidate
                failure reason so the caller knows exactly what went wrong.
    """
    failures: list[str] = []

    for col in candidates:
        # Check 1: column exists
        if col not in df.columns:
            failures.append(f"  '{col}': not found in DataFrame columns.")
            continue

        # Check 2: parses as datetime (ISO-first, inference fallback)
        try:
            parsed, used_inference = _try_parse(df[col])
        except ValueError as exc:
            failures.append(f"  '{col}': datetime parse failed — {exc}.")
            continue

        # Check 3: strictly monotonically increasing
        if not parsed.is_monotonic_increasing:
            n_violations = int((parsed.diff().dropna() <= pd.Timedelta(0)).sum())
            failures.append(
                f"  '{col}': parses as datetime but is NOT monotonically increasing "
                f"({n_violations} non-increasing step(s)). Sort the DataFrame by "
                f"this column before calling detect_timestamp_column()."
            )
            continue

        if used_inference:
            logger.warning(
                "timestamp column '%s' parsed using automatic inference — "
                "ambiguous day/month formats (e.g. 05/07/2026) may be misread "
                "as month/day without warning; ISO format YYYY-MM-DD is strongly "
                "recommended",
                col,
            )

        return col

    # Nothing worked
    tried = ", ".join(f"'{c}'" for c in candidates)
    detail = "\n".join(failures) if failures else "  (no candidates provided)"
    raise ValueError(
        f"No valid timestamp column found among candidates: {tried}.\n"
        f"Per-candidate failures:\n{detail}\n"
        f"DataFrame columns available: {list(df.columns)}"
    )


def parse_timestamp_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parse the detected timestamp column and sort the DataFrame.

    Uses the same ISO-first / inference-fallback protocol as detect_timestamp_column()
    to guarantee consistent parsing between detection and actual conversion.

    Parameters
    ----------
    df  : Raw DataFrame.
    col : Column name returned by detect_timestamp_column().

    Returns
    -------
    pd.DataFrame  — sorted copy with col as datetime64.
    """
    df = df.copy()
    try:
        parsed, used_inference = _try_parse(df[col])
    except ValueError as exc:
        raise ValueError(
            f"parse_timestamp_column: column '{col}' could not be parsed — {exc}"
        ) from exc

    if used_inference:
        logger.warning(
            "parse_timestamp_column: column '%s' parsed using automatic inference — "
            "ambiguous day/month formats (e.g. 05/07/2026) may be misread "
            "as month/day without warning; ISO format YYYY-MM-DD is strongly "
            "recommended",
            col,
        )

    df[col] = parsed
    df = df.sort_values(col).reset_index(drop=True)
    return df
