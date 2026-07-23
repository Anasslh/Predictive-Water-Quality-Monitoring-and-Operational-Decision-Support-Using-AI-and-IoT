"""
frequency_detector.py — Infer measurement frequency from timestamps.

Robust to sparse gaps (sensor outages, missing records): uses the MEDIAN of
consecutive time differences, not the mean, so a few long gaps don't inflate
the detected period.

Example
-------
    df = pd.read_csv("c1_clean.csv", parse_dates=["Date"])
    freq = detect_frequency(df)          # timedelta(hours=24)
    n    = compute_n_steps(48, freq)     # 2
    n    = compute_n_steps(72, freq)     # 3
"""

import logging
import math
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def detect_frequency(df: pd.DataFrame, date_col: str = "Date") -> timedelta:
    """
    Infer the dominant measurement interval from a time-indexed DataFrame.

    Algorithm
    ---------
    1. Sort by date_col, drop duplicates.
    2. Compute all consecutive time differences (N-1 values for N rows).
    3. Return the MEDIAN difference as the representative frequency.
       Using the median instead of the mean ensures robustness: a handful of
       multi-day sensor gaps or duplicate timestamps don't skew the result.

    Parameters
    ----------
    df       : DataFrame containing a datetime column.
    date_col : Name of the datetime column (default "Date").

    Returns
    -------
    timedelta
        Median inter-measurement interval.

    Raises
    ------
    ValueError  if df has fewer than 2 rows (cannot compute a difference).
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")

    raw_dates = pd.to_datetime(df[date_col])
    dates = raw_dates.drop_duplicates().sort_values().reset_index(drop=True)
    n_dup = len(raw_dates) - len(dates)
    if n_dup > 0:
        logger.warning(
            "detect_frequency: %d duplicate timestamp(s) removed before frequency computation",
            n_dup,
        )

    if len(dates) < 2:
        raise ValueError(
            f"Need at least 2 timestamps to detect a frequency; got {len(dates)}."
        )

    diffs = dates.diff().dropna()          # Series of timedelta
    median_diff = diffs.median()           # pandas returns a Timedelta
    return median_diff.to_pytimedelta()    # convert to stdlib timedelta


def compute_n_steps(horizon_hours: float, frequency: timedelta) -> int:
    """
    Compute how many prediction steps are needed to cover a given horizon.

    Formula
    -------
    n_steps = floor(horizon_hours / frequency_in_hours)

    Uses floor division: if the horizon is not an exact multiple of the
    measurement interval, the last partial step is dropped (conservative).

    Parameters
    ----------
    horizon_hours : float
        Desired forecast horizon in hours (e.g. 48, 72).
    frequency     : timedelta
        Measurement interval returned by detect_frequency().

    Returns
    -------
    int  ≥ 1

    Raises
    ------
    ValueError  if frequency ≤ 0 or horizon_hours ≤ 0.

    Examples
    --------
    >>> from datetime import timedelta
    >>> compute_n_steps(48, timedelta(hours=24))
    2
    >>> compute_n_steps(72, timedelta(hours=24))
    3
    >>> compute_n_steps(48, timedelta(hours=1))
    48
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive; got {horizon_hours}.")

    freq_hours = frequency.total_seconds() / 3600.0
    if freq_hours <= 0:
        raise ValueError(f"frequency must be positive; got {frequency}.")

    return max(1, math.floor(horizon_hours / freq_hours))


def frequency_summary(frequency: timedelta) -> str:
    """Human-readable description of a detected frequency."""
    total_h = frequency.total_seconds() / 3600.0
    if total_h < 1:
        return f"~{int(frequency.total_seconds() / 60)} minutes"
    if total_h < 24:
        return f"~{total_h:.1f} hours"
    return f"~{total_h / 24:.1f} days"
