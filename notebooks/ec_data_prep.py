"""
ec_data_prep.py — Data preparation pipeline for the EC parameter.

Loads the cleaned C-1 dataset, builds features, applies the chronological
split, and saves the six resulting CSVs to data/processed/ so that
src/models/ec/train.py can load them directly without re-running this script.

Run from the repo root:
    python notebooks/ec_data_prep.py
"""

import sys
from pathlib import Path

# Allow imports from src/ when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.feature_engineering import build_features
from src.data.split import chronological_split

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_RAW   = ROOT / "data" / "processed" / "c1_clean.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_RAW, parse_dates=["Date"])
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ── Feature engineering ────────────────────────────────────────────────────
print("\nBuilding features for EC...")
X, y = build_features(df, target="EC")

print(f"\nFeatures ({len(X.columns)}):")
for col in X.columns:
    print(f"  {col}")

print(f"\nSample (first 3 rows):")
print(X.head(3).to_string())
print(y.head(3).to_string())

# ── Chronological split ────────────────────────────────────────────────────
print("\nSplitting...")
X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

# ── Save splits ────────────────────────────────────────────────────────────
print("\nSaving splits to data/processed/...")
splits = {
    "ec_X_train": X_train,
    "ec_y_train": y_train.to_frame(),
    "ec_X_val":   X_val,
    "ec_y_val":   y_val.to_frame(),
    "ec_X_test":  X_test,
    "ec_y_test":  y_test.to_frame(),
}
for name, data in splits.items():
    path = OUTPUT_DIR / f"{name}.csv"
    data.to_csv(path, index=False)
    print(f"  Saved {path.name} — {data.shape}")

print("\nDone. All splits saved.")
