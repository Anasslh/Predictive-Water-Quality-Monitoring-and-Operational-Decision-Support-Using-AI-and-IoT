"""
data_loader.py — Isolated data ingestion for the retraining pipeline.

PURPOSE OF ISOLATION
--------------------
This module is the single point of contact with the raw data source.
The rest of the retraining pipeline (drift_detector, retrain_manager) only
ever receives a pd.DataFrame — they never know whether the data came from a
single growing CSV, multiple files split by period, a database query, or an
IoT streaming endpoint.

If the data source changes in the future, this is the ONLY file to modify.

Current implementation: one CSV file that grows over time (C-1 station format).
"""

import pandas as pd
from pathlib import Path


def load_available_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load all currently available data from a single CSV file.

    The file is assumed to follow the C-1 station format:
      - A 'Date' column parseable as datetime.
      - No missing values (contract: data has been pre-cleaned before storage).
      - Rows in any order (sorted here by Date for downstream safety).

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file that accumulates new rows over time.

    Returns
    -------
    pd.DataFrame
        All rows, sorted by Date ascending, index reset.
        Downstream code must never depend on the original file's row order.

    Raises
    ------
    FileNotFoundError  if csv_path does not exist.
    ValueError         if 'Date' column is absent or unparseable.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])

    if "Date" not in df.columns:
        raise ValueError(f"'Date' column missing in {path}. Expected C-1 station format.")

    df = df.sort_values("Date").reset_index(drop=True)
    return df
