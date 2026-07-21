"""
feature_engineering.py — Time-series feature generation for per-parameter prediction.

Builds lag features, rolling statistics, and cross-variable lags from the
cleaned C-1 dataset. Parameterized by target column so the same function
works for pH, EC, and Turbidity without modification.

All features are constructed from t-1 and earlier only — no data leakage
from the current timestep into the feature matrix.
"""

import pandas as pd

# Co-variables to include as lag-1 features for each target.
# These cross-parameter lags are useful for SHAP: they reveal whether
# changes in one parameter help predict another.
_COVARIATES: dict[str, list[str]] = {
    "EC":        ["pH", "Turbidity"],
    "pH":        ["EC", "Turbidity"],
    "Turbidity": ["pH", "EC"],
}


def build_features(
    df: pd.DataFrame,
    target: str,
    diff: bool = False,
    ewma_spans: list[int] | None = None,
    extra_lags: list[int] | None = None,
    rolling_slope: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and target vector y for a given parameter.

    Base features (always generated)
    ---------------------------------
    - {target}_lag1, _lag2, _lag3       : target at t-1, t-2, t-3
    - {target}_roll3_mean / _roll3_std  : 3-day rolling mean/std (from t-1)
    - {target}_roll7_mean / _roll7_std  : 7-day rolling mean/std (from t-1)
    - {covar}_lag1 for each co-variable : cross-parameter lag at t-1

    Optional features
    -----------------
    ewma_spans  : EWMA of target (shifted to avoid leakage). e.g. [3], [5], [3,5].
    extra_lags  : additional target lags. e.g. [7, 14].
                  ⚠ lag-14 shifts the NaN drop boundary to 14 rows (from 7).
    rolling_slope : (target_lag1 - target_lag7) / 6 — 7-day linear trend proxy.
                    Requires at least shift(7) → same NaN boundary as lag7.

    Target transform
    ----------------
    diff=False (default): y = target(t)
    diff=True           : y = target(t) - target(t-1)
      Reconstruct after prediction: pred_abs(t) = X["{target}_lag1"] + delta_pred

    Parameters
    ----------
    df            : cleaned dataset with 'Date' and target columns.
    target        : "EC", "pH", or "Turbidity".
    diff          : predict first-difference instead of raw value.
    ewma_spans    : list of EWMA spans to add (e.g. [3, 5]).
    extra_lags    : list of additional lag steps (e.g. [7, 14]).
    rolling_slope : add 7-day slope feature.

    Returns
    -------
    X : pd.DataFrame  — feature matrix, NaN rows dropped, Date excluded.
    y : pd.Series     — target (raw or differenced) aligned with X.
    """
    ewma_spans = ewma_spans or []
    extra_lags = extra_lags or []

    df = df.copy().sort_values("Date").reset_index(drop=True)
    feat = pd.DataFrame(index=df.index)

    # --- Base lags of the target ---
    for lag in [1, 2, 3]:
        feat[f"{target}_lag{lag}"] = df[target].shift(lag)

    # --- Extra lags (e.g. lag7, lag14) ---
    for lag in extra_lags:
        feat[f"{target}_lag{lag}"] = df[target].shift(lag)

    # --- Rolling statistics (shift first to avoid leakage) ---
    shifted = df[target].shift(1)
    for window in [3, 7]:
        feat[f"{target}_roll{window}_mean"] = shifted.rolling(window, min_periods=window).mean()
        feat[f"{target}_roll{window}_std"]  = shifted.rolling(window, min_periods=window).std()

    # --- EWMA features (shift to avoid leakage) ---
    for span in ewma_spans:
        feat[f"{target}_ewma{span}"] = shifted.ewm(span=span, adjust=False).mean()

    # --- Rolling slope proxy: (lag1 - lag7) / 6 ---
    if rolling_slope:
        feat[f"{target}_slope7"] = (df[target].shift(1) - df[target].shift(7)) / 6

    # --- Lag t-1 of co-variables ---
    covariates = _COVARIATES.get(target, [c for c in ["pH", "EC", "Turbidity"] if c != target])
    for col in covariates:
        feat[f"{col}_lag1"] = df[col].shift(1)

    # --- Target (raw or differenced) ---
    feat[target] = df[target] - df[target].shift(1) if diff else df[target]

    n_before = len(feat)
    feat = feat.dropna().reset_index(drop=True)
    n_dropped = n_before - len(feat)

    tag_parts = []
    if diff:          tag_parts.append("diff")
    if ewma_spans:    tag_parts.append(f"ewma{ewma_spans}")
    if extra_lags:    tag_parts.append(f"lags{extra_lags}")
    if rolling_slope: tag_parts.append("slope7")
    tag = ",".join(tag_parts) if tag_parts else "baseline"
    print(f"[feature_engineering] target={target} | {tag} | {n_before}→{len(feat)} rows ({n_dropped} dropped)")

    X = feat.drop(columns=[target])
    y = feat[target].rename(target)
    return X, y
