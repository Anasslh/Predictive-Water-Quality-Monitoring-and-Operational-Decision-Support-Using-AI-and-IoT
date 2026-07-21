"""
split.py — Chronological train / validation / test split for time-series data.

IMPORTANT: never use a random split on temporal data. Shuffling would leak
future values into training and produce optimistic, unreliable metrics.
All splits here preserve temporal order strictly.
"""

import pandas as pd


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split X and y into three consecutive, non-overlapping blocks.

    Block boundaries are computed from ratios applied to the total number of
    rows. The test set takes whatever remains after train + val, so the three
    ratios always sum to 1 regardless of rounding.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix. Must already be sorted chronologically (guaranteed
        by build_features in feature_engineering.py).
    y : pd.Series
        Target vector aligned with X.
    train_ratio : float
        Fraction of rows assigned to the training set (default 0.70).
    val_ratio : float
        Fraction of rows assigned to the validation set (default 0.15).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    X_train, y_train = X.iloc[:train_end].copy(),        y.iloc[:train_end].copy()
    X_val,   y_val   = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end].copy()
    X_test,  y_test  = X.iloc[val_end:].copy(),          y.iloc[val_end:].copy()

    print(
        f"[split] n={n} | "
        f"train={len(X_train)} ({len(X_train)/n:.0%}) | "
        f"val={len(X_val)} ({len(X_val)/n:.0%}) | "
        f"test={len(X_test)} ({len(X_test)/n:.0%})"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
