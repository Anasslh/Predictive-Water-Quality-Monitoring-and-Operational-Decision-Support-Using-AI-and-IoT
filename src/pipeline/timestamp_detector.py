"""
timestamp_detector.py — Identify the timestamp column in an unknown DataFrame.

STRATEGY
--------
Try each candidate column name in order. A column qualifies if:
  1. It exists in the DataFrame.
  2. Its values parse as datetimes without errors (using pd.to_datetime with
     errors="raise", so a single unparseable value disqualifies the whole column).
  3. After parsing, the values are strictly monotonically increasing — a basic
     sanity check that rules out columns whose integer values happen to parse
     but clearly are not ordered timestamps.

The function returns the FIRST candidate that passes all three tests.
It never guesses or infers from dtype alone; it always validates on real values.

If no candidate qualifies, a ValueError is raised with a clear explanation of
what was tried and why each candidate failed — no silent fallbacks.
"""

from __future__ import annotations

import pandas as pd


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

        # Check 2: parses as datetime without error
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
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
    Parse the detected timestamp column in-place and sort the DataFrame.

    Convenience wrapper: converts col to datetime64, sorts ascending, and
    resets the index. Returns a new DataFrame (does not modify in place).

    Parameters
    ----------
    df  : Raw DataFrame.
    col : Column name returned by detect_timestamp_column().

    Returns
    -------
    pd.DataFrame  — sorted copy with col as datetime64.
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df = df.sort_values(col).reset_index(drop=True)
    return df
