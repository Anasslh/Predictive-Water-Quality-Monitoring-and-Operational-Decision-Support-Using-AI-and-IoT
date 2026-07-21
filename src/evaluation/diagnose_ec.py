"""
diagnose_ec.py — Diagnostic on EC model results before hyperparameter tuning.

Checks:
  1. EC target distribution per split (train / val / test) — extrapolation risk.
  2. SVR y-scaling: compares old version (X-only scaling) vs new (X + y scaling).
  3. First 10 val predictions for RF, XGBoost, SVR vs ground truth.

Run from repo root:
    python src/evaluation/diagnose_ec.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.ec.train         import ECModel
from src.models.ec.xgboost_model import ECModelXGBoost
from src.models.ec.svr_model     import ECModelSVR

# ── Load splits ─────────────────────────────────────────────────────────────
PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()
X_test  = pd.read_csv(PROC / "ec_X_test.csv")
y_test  = pd.read_csv(PROC / "ec_y_test.csv").squeeze()

SEP = "=" * 60

# ── 1. EC distribution per split ────────────────────────────────────────────
print(SEP)
print("1. EC TARGET DISTRIBUTION PER SPLIT  (µS/cm)")
print(SEP)

splits_info = [
    ("Train", y_train, len(y_train)),
    ("Val",   y_val,   len(y_val)),
    ("Test",  y_test,  len(y_test)),
]

print(f"{'Split':<8} {'n':>5} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
print("-" * 50)
for label, y, n in splits_info:
    print(f"{label:<8} {n:>5} {y.min():>8.1f} {y.max():>8.1f} {y.mean():>8.1f} {y.std():>8.1f}")

# Extrapolation check
val_below_train = y_val.min() < y_train.min()
val_above_train = y_val.max() > y_train.max()
print()
if val_below_train:
    print(f"  ⚠  Val min ({y_val.min():.1f}) < Train min ({y_train.min():.1f})"
          f"  → extrapolation below train range")
if val_above_train:
    print(f"  ⚠  Val max ({y_val.max():.1f}) > Train max ({y_train.max():.1f})"
          f"  → extrapolation above train range")
if not val_below_train and not val_above_train:
    print("  ✓  Val EC range fully contained in Train EC range — no extrapolation.")

val_pct_outside = (
    ((y_val < y_train.min()) | (y_val > y_train.max())).sum() / len(y_val) * 100
)
print(f"  Val rows outside train range: {val_pct_outside:.1f}%")


# ── 2. SVR y-scaling — before vs after ──────────────────────────────────────
print()
print(SEP)
print("2. SVR  —  y-scaling: before (X-only) vs after (X + y)")
print(SEP)

# --- Old SVR: X scaled, y raw ---
old_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.1)),
])
old_pipeline.fit(X_train, y_train)
old_preds = old_pipeline.predict(X_val)

old_rmse = np.sqrt(mean_squared_error(y_val, old_preds))
old_mae  = mean_absolute_error(y_val, old_preds)
old_r2   = r2_score(y_val, old_preds)

# --- New SVR: X scaled + y scaled (ECModelSVR v1) ---
new_svr = ECModelSVR()
new_svr.fit(X_train, y_train)
new_preds = new_svr.predict(X_val)

new_rmse = np.sqrt(mean_squared_error(y_val, new_preds))
new_mae  = mean_absolute_error(y_val, new_preds)
new_r2   = r2_score(y_val, new_preds)

print(f"{'Version':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 50)
print(f"{'SVR  (X scaled only)':<22} {old_rmse:>8.2f} {old_mae:>8.2f} {old_r2:>8.4f}")
print(f"{'SVR  (X + y scaled)':<22} {new_rmse:>8.2f} {new_mae:>8.2f} {new_r2:>8.4f}")
print()

# Check if SVR predicts quasi-constant (sign of mean-regression)
old_pred_std = np.std(old_preds)
new_pred_std = np.std(new_preds)
true_std     = float(y_val.std())
print(f"  y_val std      : {true_std:.2f}")
print(f"  Old SVR pred std: {old_pred_std:.2f}  (ratio to true: {old_pred_std/true_std:.2f})")
print(f"  New SVR pred std: {new_pred_std:.2f}  (ratio to true: {new_pred_std/true_std:.2f})")
print("  (ratio << 1 → model predicts near-constant / barely follows signal)")


# ── 3. First 10 val predictions: RF, XGBoost, SVR ───────────────────────────
print()
print(SEP)
print("3. FIRST 10 VAL ROWS — true vs predicted (µS/cm)")
print(SEP)

rf  = ECModel();      rf.fit(X_train, y_train)
xgb = ECModelXGBoost(); xgb.fit(X_train, y_train)
# new_svr already fitted above

rf_preds  = rf.predict(X_val)
xgb_preds = xgb.predict(X_val)
svr_preds = new_svr.predict(X_val)

print(f"{'#':>3}  {'True':>8}  {'RF':>8}  {'XGBoost':>8}  {'SVR (v1)':>10}")
print("-" * 46)
for i in range(10):
    true_val = float(y_val.iloc[i])
    print(
        f"{i+1:>3}  {true_val:>8.1f}  "
        f"{rf_preds[i]:>8.1f}  "
        f"{xgb_preds[i]:>8.1f}  "
        f"{svr_preds[i]:>10.1f}"
    )

print()
print("  Error on first 10 rows (|pred - true|):")
print(f"  {'#':>3}  {'RF':>8}  {'XGBoost':>8}  {'SVR (v1)':>10}")
print("  " + "-" * 36)
for i in range(10):
    true_val = float(y_val.iloc[i])
    print(
        f"  {i+1:>3}  "
        f"{abs(rf_preds[i]  - true_val):>8.1f}  "
        f"{abs(xgb_preds[i] - true_val):>8.1f}  "
        f"{abs(svr_preds[i] - true_val):>10.1f}"
    )

print()
print(SEP)
print("Reminder: test set not used — reserved for final model selection.")
print(SEP)
