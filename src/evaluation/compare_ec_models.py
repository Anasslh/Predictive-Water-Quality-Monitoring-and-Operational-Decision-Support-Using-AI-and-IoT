"""
compare_ec_models.py — Out-of-the-box comparison of EC prediction models.

Trains RF, XGBoost, and SVR on the EC training split and evaluates each
on the validation set. Test set is intentionally not touched here —
it is reserved for the final evaluation of the selected model.

Run from the repo root:
    python src/evaluation/compare_ec_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.ec.train         import ECModel
from src.models.ec.xgboost_model import ECModelXGBoost
from src.models.ec.svr_model     import ECModelSVR

# ── Load splits ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}\n")

# ── Naive baselines ────────────────────────────────────────────────────────
# Persistence: predict EC_lag1 (= last observed value, already in X_val).
# Mean train:  predict y_train mean for every val row.
# Both are fit-free — they define the floor any learned model must beat.
persistence_preds = X_val["EC_lag1"].values
mean_train_preds  = np.full(len(y_val), y_train.mean())

results = []
for name, preds in [
    ("Persistence (lag1)", persistence_preds),
    ("Mean train",         mean_train_preds),
]:
    results.append({
        "Model":   name,
        "RMSE":    np.sqrt(mean_squared_error(y_val, preds)),
        "MAE":     mean_absolute_error(y_val, preds),
        "R²":      r2_score(y_val, preds),
        "baseline": True,
    })

# ── Models to compare ──────────────────────────────────────────────────────
models = [
    ("Random Forest", ECModel()),
    ("XGBoost",       ECModelXGBoost()),
    ("SVR",           ECModelSVR()),
]

for name, model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    results.append({
        "Model":    name,
        "RMSE":     np.sqrt(mean_squared_error(y_val, preds)),
        "MAE":      mean_absolute_error(y_val, preds),
        "R²":       r2_score(y_val, preds),
        "baseline": False,
    })
    print(f"[{name}] trained.")

# ── Results table ──────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 58)

# Baselines first (fixed order), then learned models sorted by RMSE
baselines = [r for r in results if r["baseline"]]
learned   = sorted([r for r in results if not r["baseline"]], key=lambda x: x["RMSE"])

best_rmse      = min(r["RMSE"] for r in results)
persistence_rmse = next(r["RMSE"] for r in baselines if "Persistence" in r["Model"])

for r in baselines + learned:
    marker = " ◄ best" if r["RMSE"] == best_rmse else ""
    print(f"{r['Model']:<22} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R²']:>8.4f}{marker}")

print("=" * 58)

# Summary verdict
best_learned_rmse = min(r["RMSE"] for r in learned)
if best_learned_rmse < persistence_rmse:
    gap = persistence_rmse - best_learned_rmse
    print(f"✓  Best model beats persistence by {gap:.2f} RMSE — tuning warranted.")
else:
    gap = best_learned_rmse - persistence_rmse
    print(f"✗  No model beats persistence (gap: +{gap:.2f} RMSE).")
    print("   Consider richer trend features before hyperparameter tuning.")

print("Val set only — test set reserved for final evaluation.")
