"""
feature_engineering_generic.py — Frequency-aware feature engineering.

This module generalises src/data/feature_engineering.py so that lag windows
and rolling windows are expressed in PHYSICAL HOURS rather than row counts.
Row counts are derived automatically from the dataset's measurement frequency:

    n_rows = floor(window_hours / frequency_hours)

Example: "rolling over last 72 hours"
    Daily data  (freq = 24 h) → 72 / 24 = 3  rows  → column EC_roll3_mean
    Hourly data (freq =  1 h) → 72 /  1 = 72 rows  → column EC_roll72_mean

COLUMN NAMING CONVENTION
-------------------------
Columns are named by the NUMBER OF STEPS (rows), not by hours, so that:
  - Models trained on daily data with lag_hours=[24,48,72] get the familiar
    columns EC_lag1, EC_lag2, EC_lag3 (compatible with existing trained models).
  - Models trained on hourly data with the same lag_hours get EC_lag24, EC_lag48,
    EC_lag72 (correct for that frequency).

USE CASES
---------
1.  Building a new model on a dataset with a different measurement frequency
    (e.g. hourly sensor → use build_features_time_aware instead of build_features).
2.  The recursive forecaster's _get_next_step_features() method uses the
    original build_features (injected as feature_fn) via an append-and-extract
    pattern that is frequency-agnostic (see recursive_forecaster.py).

BACKWARD COMPATIBILITY
----------------------
Calling build_features_time_aware(df, "EC", timedelta(hours=24),
    lag_hours=[24, 48, 72], rolling_hours=[72, 168])
produces the IDENTICAL column set as build_features(df, "EC") from
src/data/feature_engineering.py, because:
    24/24=1, 48/24=2, 72/24=3  → lag1, lag2, lag3
    72/24=3, 168/24=7           → roll3_mean/std, roll7_mean/std
"""

import logging
import math
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def build_features_time_aware(
    df: pd.DataFrame,
    target: str,
    frequency: timedelta,
    lag_hours: list[int] | None = None,
    rolling_hours: list[int] | None = None,
    diff: bool = False,
    co_variables: list[str] | None = None,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix using hour-based windows, independent of data frequency.

    Parameters
    ----------
    df           : Cleaned dataset. Must contain the timestamp column and the target.
    target       : "EC", "pH", or "Turbidity".
    frequency    : Measurement interval (from detect_frequency()).
                   Used to convert hour-based windows to row counts.
    lag_hours    : Lag depths in hours. Default [24, 48, 72] → steps 1, 2, 3
                   for daily data; steps 24, 48, 72 for hourly data.
    rolling_hours: Rolling window durations in hours. Default [72, 168].
                   72 h / 24 h/day = 3 rows (daily), 72 h / 1 h = 72 rows (hourly).
    diff         : If True, predict first difference instead of raw value.
    co_variables : List of co-variable column names to add as lag-1 features.
                   Pass None (or omit) only when no co-variables are declared
                   in sensors_config.json; a warning is logged and cross-
                   parameter features are skipped. Pass [] explicitly to
                   silence the warning and skip them intentionally.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (NaN rows dropped, Date excluded).
    y : pd.Series     — target aligned with X.

    Notes on NaN dropping
    ---------------------
    The number of rows dropped at the start equals max(lag_steps, rolling_rows).
    For daily data with lag_hours=[24,48,72] and rolling_hours=[72,168]:
    max = max(3, 7) = 7 rows dropped (same as build_features baseline).
    """
    lag_hours     = lag_hours     or [24, 48, 72]
    rolling_hours = rolling_hours or [72, 168]

    freq_h = frequency.total_seconds() / 3600.0

    # Convert hours → row counts (floor division, minimum 1)
    lag_steps     = [max(1, math.floor(h / freq_h)) for h in lag_hours]
    rolling_rows  = [max(1, math.floor(h / freq_h)) for h in rolling_hours]

    df   = df.copy().sort_values(date_col).reset_index(drop=True)
    feat = pd.DataFrame(index=df.index)

    # Lag features — named by step count (not hours) for column-name stability
    for n_steps in lag_steps:
        feat[f"{target}_lag{n_steps}"] = df[target].shift(n_steps)

    # Rolling features — shifted by 1 to avoid leakage, named by row count
    shifted = df[target].shift(1)
    for n_rows in rolling_rows:
        feat[f"{target}_roll{n_rows}_mean"] = (
            shifted.rolling(n_rows, min_periods=n_rows).mean()
        )
        feat[f"{target}_roll{n_rows}_std"] = (
            shifted.rolling(n_rows, min_periods=n_rows).std()
        )

    # Cross-variable lag-1 features (always 1 step = 1 measurement interval).
    # co_variables must be passed by the caller (from sensors_config.json).
    # None means no co_variables were declared for this parameter in config.
    if co_variables is None:
        logger.warning(
            "no co_variables defined for '%s' in sensors_config.json"
            " — proceeding without cross-parameter features",
            target,
        )
        co_variables = []
    for col in co_variables:
        if col in df.columns:
            feat[f"{col}_lag1"] = df[col].shift(1)

    # Target (raw or differenced)
    feat[target] = df[target] - df[target].shift(1) if diff else df[target]

    n_before = len(feat)
    feat = feat.dropna().reset_index(drop=True)
    n_dropped = n_before - len(feat)

    freq_label = f"{freq_h:.0f}h"
    print(
        f"[feature_engineering_generic] target={target} | freq={freq_label} | "
        f"lags={lag_steps} rows | rolls={rolling_rows} rows | "
        f"{n_before}→{len(feat)} rows ({n_dropped} dropped)"
    )

    X = feat.drop(columns=[target])
    y = feat[target].rename(target)
    return X, y
