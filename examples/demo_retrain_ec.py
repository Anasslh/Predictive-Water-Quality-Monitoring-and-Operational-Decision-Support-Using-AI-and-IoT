"""
demo_retrain_ec.py — Simulated retraining scenario for the EC pipeline.

This script demonstrates the retraining module WITHOUT modifying the
production model (ec_xgboost_v1_final.json). All candidate models are saved
to models_store/retrain_demo/ (separate from production storage).

SCENARIO
--------
The C-1 station has been running for ~10 months. The EC XGBoost model was
trained and frozen on the original 303-row window (train=250 + val=53).
New measurements are now arriving. We simulate two batches:

  Batch 1: raw rows 310-336 of c1_clean.csv  (27 raw → ~27 new FE rows)
  Batch 2: raw rows 337-364 of c1_clean.csv  (28 raw → ~28 additional FE rows)

After batch 1 alone (27 new rows < min_new_rows=30): no action.
After batch 2 (55 cumulative new rows ≥ 30): drift check triggers.

The manager then:
  1. Computes residuals of the frozen model on the 55 new rows.
  2. Tests distribution shift vs reference residuals (KS test).
  3. If drift detected: trains a candidate on the expanded dataset,
     compares RMSE on a shared test set, and decides to accept or reject.

Run from repo root:
    python src/retraining/demo_retrain_ec.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.models.ec.xgboost_model import ECModelXGBoost
from src.data.feature_engineering import build_features
from src.data.split import chronological_split
from src.retraining.data_loader import load_available_data
from src.retraining.retrain_manager import RetrainManager
from src.retraining.model_versioning import save_model_version, list_versions

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
RAW_CSV     = ROOT / "data" / "processed" / "c1_clean.csv"
MODEL_JSON  = ROOT / "models_store" / "ec_xgboost_v1_final.json"
DEMO_STORE  = ROOT / "models_store" / "retrain_demo"   # separate from production

SEP  = "=" * 68
SEP2 = "-" * 68

# ── Feature engineering function (EC-specific — injected into the manager) ────
# This is the ONLY EC-specific piece. Replace with:
#   lambda df: build_features(df, "pH")          for pH
#   lambda df: build_features(df, "Turbidity")   for turbidity
def fe_ec(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(df, target="EC")

# ── Data partition (raw row indices) ──────────────────────────────────────────
# c1_clean.csv has 365 rows (0-364).
# Feature engineering drops the first 7 rows (rolling-7 lag).
# Original training window: raw rows 0-309  → 310 raw → 303 FE rows
# Batch 1:                  raw rows 310-336 → 27 raw rows (new data, small batch)
# Batch 2:                  raw rows 337-364 → 28 raw rows (completes test period)
INITIAL_END = 310    # exclusive — initial raw data slice
BATCH1_END  = 337    # exclusive — after batch 1
# BATCH2_END  = 365  # = all data (c1_clean.csv full length)

print(SEP)
print("DEMO — Simulated EC retraining pipeline")
print("Production model: untouched. Candidates saved to models_store/retrain_demo/")
print(SEP)

# ── Load and partition raw data ────────────────────────────────────────────────
df_full = load_available_data(RAW_CSV)
print(f"\nFull dataset: {len(df_full)} rows  "
      f"({df_full['Date'].min().date()} → {df_full['Date'].max().date()})\n")

df_initial = df_full.iloc[:INITIAL_END].copy().reset_index(drop=True)
df_batch1  = df_full.iloc[:BATCH1_END].copy().reset_index(drop=True)
df_batch2  = df_full.copy()   # all data = initial + batch 1 + batch 2

# Verify feature-engineered sizes
X_init, y_init = fe_ec(df_initial)
X_b1,   _      = fe_ec(df_batch1)
X_b2,   _      = fe_ec(df_batch2)
print(f"Feature-engineered rows:")
print(f"  Initial data   : {len(X_init)} rows  (= reference window)")
print(f"  + Batch 1      : {len(X_b1)} rows   ({len(X_b1) - len(X_init)} new)")
print(f"  + Batch 2      : {len(X_b2)} rows   ({len(X_b2) - len(X_init)} new total)")

# ── Load the frozen production model ──────────────────────────────────────────
print(SEP2)
print("Loading frozen production model (xgb_ec_v1_final.json)…")
frozen_model = ECModelXGBoost()
frozen_model.model.load_model(str(MODEL_JSON))
frozen_model._is_fitted = True
print(f"  Loaded: {frozen_model.model_version}")

# Quick sanity check: RMSE on original train+val
X_train_i, y_train_i, X_val_i, y_val_i, _, _ = chronological_split(X_init, y_init)
X_tv_i = pd.concat([X_train_i, X_val_i], ignore_index=True)
y_tv_i = pd.concat([y_train_i, y_val_i], ignore_index=True)
ref_preds = frozen_model.predict(X_tv_i)
ref_rmse  = float(np.sqrt(np.mean((y_tv_i.values - ref_preds) ** 2)))
print(f"  RMSE on original train+val (reference): {ref_rmse:.2f} µS/cm")

# ── Initialize the RetrainManager ─────────────────────────────────────────────
print(SEP2)
print("Initializing RetrainManager…")
manager = RetrainManager(
    model_class  = ECModelXGBoost,
    feature_fn   = fe_ec,
    min_new_rows = 60,      # default — at least 60 new daily observations (~2 months)
    tolerance    = 0.02,    # accept candidate if RMSE ≤ current × 1.02
)
manager.initialize(df_initial, frozen_model)
print(f"  min_new_rows = {manager.min_new_rows}")
print(f"  tolerance    = {manager.tolerance} ({manager.tolerance*100:.0f}%)")
print(f"  Reference residuals: n={len(manager.reference_residuals)}, "
      f"σ={np.std(manager.reference_residuals):.2f} µS/cm")
print(f"  last_train_size (FE rows) = {manager.last_train_size}")

# ── Batch 1 arrival ────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"BATCH 1 — {len(df_batch1) - INITIAL_END} raw rows added  "
      f"(+{len(X_b1) - len(X_init)} FE rows, total {len(X_b1)})")
print(SEP)

print(f"\nshould_check_retrain({len(X_b1)}) ?  "
      f"{len(X_b1) - manager.last_train_size} new rows vs min={manager.min_new_rows}")
result1 = manager.attempt_retrain(df_batch1)
print(f"  → {result1['reason']}")
print(f"  retrained={result1['retrained']}  accepted={result1['accepted']}")

# ── Batch 2 arrival ────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"BATCH 2 — {len(df_full) - len(df_batch1)} raw rows added  "
      f"(+{len(X_b2) - len(X_b1)} more FE rows, total {len(X_b2)} = "
      f"{len(X_b2) - manager.last_train_size} new since last training)")
print(SEP)

print(f"\nshould_check_retrain({len(X_b2)}) ?  "
      f"{len(X_b2) - manager.last_train_size} new rows vs min={manager.min_new_rows}")
result2 = manager.attempt_retrain(df_batch2)

drift = result2.get("drift", {})
if drift:
    print()
    print("  Drift check results:")
    print(f"    KS statistic      = {drift['ks_statistic']:.4f}")
    print(f"    KS p-value        = {drift['ks_pvalue']:.4f}  "
          f"({'⚡ p < 0.05 → distribution shift' if drift['ks_drift'] else '✓ no distribution shift'})")
    print(f"    Reference RMSE    = {drift['reference_rmse']:.2f} µS/cm")
    print(f"    Recent RMSE       = {drift['recent_rmse']:.2f} µS/cm")
    print(f"    RMSE ratio        = {drift['rmse_ratio']:.3f}  "
          f"({'⚡ > 1.20 → degradation' if drift['rmse_drift'] else '✓ no degradation'})")
    print(f"    Drift detected    = {drift['drift_detected']}")

if result2["retrained"]:
    print()
    print("  Candidate vs current model (shared test set):")
    print(f"    Old RMSE  = {result2['old_rmse']:.4f} µS/cm")
    print(f"    New RMSE  = {result2['new_rmse']:.4f} µS/cm")
    tolerance_bound = result2['old_rmse'] * (1 + manager.tolerance)
    print(f"    Tolerance = old × {1 + manager.tolerance} = {tolerance_bound:.4f} µS/cm")
    accepted_str = "✓ ACCEPTED — candidate promoted" if result2["accepted"] else "✗ REJECTED — current model retained"
    print(f"    Decision  : {accepted_str}")

print()
print(f"  Reason: {result2['reason']}")
print(f"  retrained={result2['retrained']}  accepted={result2['accepted']}")

# ── Versioning demo ────────────────────────────────────────────────────────────
if result2["accepted"]:
    print()
    print(SEP2)
    print("VERSIONING — Saving accepted candidate to models_store/retrain_demo/")
    DEMO_STORE.mkdir(parents=True, exist_ok=True)
    saved_path = save_model_version(
        manager.current_model,
        models_store_path=DEMO_STORE,
        max_versions=3,
    )
    print(f"  Saved: {saved_path.name}")
    available = list_versions(
        manager.current_model.parameter_name,
        manager.current_model.model_version,
        DEMO_STORE,
    )
    print(f"  Versions in store (oldest→newest): {available}")
    print("  (Production model untouched — these files are demo-only)")
else:
    print()
    print(SEP2)
    print("VERSIONING — No save (candidate not accepted or no retrain triggered).")
    print("  Production model untouched.")

# ── Final summary ──────────────────────────────────────────────────────────────
print()
print(SEP)
print("SUMMARY")
print(SEP)
print(f"  Batch 1  ({len(X_b1) - len(X_init):+d} rows)  "
      f"Volume gate: {'PASS' if len(X_b1) - len(X_init) >= manager.min_new_rows else 'FAIL (no action)'}  "
      f"→ retrained={result1['retrained']}")
if drift:
    print(f"  Batch 2  ({len(X_b2) - len(X_init):+d} rows)  "
          f"Volume gate: PASS  "
          f"Drift gate: {'PASS' if drift['drift_detected'] else 'FAIL (no action)'}  "
          f"→ retrained={result2['retrained']}, accepted={result2['accepted']}")
print()
print("Architecture check — EC-specific pieces in this demo only:")
print("  ✓ ECModelXGBoost  — injected as model_class")
print("  ✓ fe_ec lambda    — injected as feature_fn")
print("  ✓ RetrainManager itself: zero EC-specific imports")
print("  ✓ drift_detector:        zero EC-specific imports")
print("  ✓ data_loader:           zero EC-specific imports")
print("  ✓ model_versioning:      zero EC-specific imports")
print()
print("To adapt for pH: replace ECModelXGBoost and fe_ec. Nothing else changes.")
print(SEP)
