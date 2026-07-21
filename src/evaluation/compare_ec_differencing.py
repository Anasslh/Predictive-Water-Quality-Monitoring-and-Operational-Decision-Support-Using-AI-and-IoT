"""
compare_ec_differencing.py — Test whether differencing the EC target improves predictions.

Hypothesis: the train/val mean gap (783 vs 698 µS/cm) caused R² < 0 on val
because models learned the absolute level of train. Predicting the first
difference (delta EC = EC(t) - EC(t-1)) removes this level bias — the model
learns to predict change, then we reconstruct the absolute value with:
    EC_pred(t) = EC_lag1(t) + delta_pred(t)
where EC_lag1 is already a feature in X.

Variants tested (all with XGBoost depth=3, n=100, lr=0.01):
  A. Baseline      — predict EC(t) directly               (current approach)
  B. Diff only     — predict delta EC, reconstruct abs
  C. Diff + EWMA   — same as B but with EWMA(3,5) added to X (only if B helps)

Evaluation: val set only. Test set is untouched.

Run from repo root:
    python src/evaluation/compare_ec_differencing.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.data.feature_engineering import build_features
from src.data.split import chronological_split

SEP = "=" * 68

DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "c1_clean.csv"
df   = pd.read_csv(DATA, parse_dates=["Date"])

XGB_PARAMS = dict(max_depth=3, n_estimators=100, learning_rate=0.01,
                  random_state=42, verbosity=0)


def metrics_abs(y_true, y_pred):
    return (
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
        float(r2_score(y_true, y_pred)),
    )


def run_variant(label: str, X_train, y_train, X_val, y_val, diff: bool):
    """Fit XGBoost, reconstruct absolute predictions if diff=True, return metrics.

    When diff=True:
      y_val contains delta values; to compare in absolute space we compute
        y_val_abs(t) = EC_lag1(t) + delta_true(t)  = EC(t)   [true]
        preds_abs(t) = EC_lag1(t) + delta_pred(t)             [predicted]
      Both sides shift by the same EC_lag1, so the absolute metric is correct.
    """
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    ec_lag1 = X_val["EC_lag1"].values

    if diff:
        preds_abs = ec_lag1 + preds            # predicted absolute EC
        y_abs     = ec_lag1 + y_val.values     # true absolute EC  (= EC(t))
        preds_delta = preds
    else:
        preds_abs = preds
        y_abs     = y_val.values
        preds_delta = None

    rmse, mae, r2 = metrics_abs(y_abs, preds_abs)
    return model, preds_abs, preds_delta, rmse, mae, r2


# ── A. Baseline (EC brute) ────────────────────────────────────────────────────
print(SEP)
print("VARIANT A — EC(t) direct  [baseline]")
print(SEP)

X_a, y_a = build_features(df, "EC", diff=False, ewma=False)
X_train_a, y_train_a, X_val_a, y_val_a, _, _ = chronological_split(X_a, y_a)

_, _, _, rmse_a, mae_a, r2_a = run_variant("A", X_train_a, y_train_a, X_val_a, y_val_a, diff=False)

print(f"  RMSE={rmse_a:.2f}  MAE={mae_a:.2f}  R²={r2_a:.4f}")
print(f"  y_val : mean={y_val_a.mean():.1f}  std={y_val_a.std():.1f}")
print(f"  y_train: mean={y_train_a.mean():.1f}  std={y_train_a.std():.1f}")
print(f"  Level gap (train mean − val mean): {y_train_a.mean()-y_val_a.mean():+.1f} µS/cm")


# ── B. Differencing — predict delta EC ────────────────────────────────────────
print()
print(SEP)
print("VARIANT B — ΔEC(t) = EC(t) - EC(t-1)  [differenced, reconstructed]")
print(SEP)

X_b, y_b = build_features(df, "EC", diff=True, ewma=False)
X_train_b, y_train_b, X_val_b, y_val_b, _, _ = chronological_split(X_b, y_b)

# y_b is the delta; val absolute values come from X_val_b["EC_lag1"] + delta_pred
# True absolute values for val (for metrics):
y_val_abs_b = X_val_b["EC_lag1"].values + y_val_b.values

_, preds_abs_b, preds_delta_b, rmse_b, mae_b, r2_b = run_variant(
    "B", X_train_b, y_train_b, X_val_b, y_val_b, diff=True
)

# Delta metrics (in differenced space)
true_delta_val = y_val_b.values
rmse_delta = float(np.sqrt(mean_squared_error(true_delta_val, preds_delta_b)))
mae_delta  = float(mean_absolute_error(true_delta_val, preds_delta_b))
r2_delta   = float(r2_score(true_delta_val, preds_delta_b))

print(f"  Delta space : RMSE={rmse_delta:.2f}  MAE={mae_delta:.2f}  R²={r2_delta:.4f}")
print(f"    y_train delta: mean={y_train_b.mean():.2f}  std={y_train_b.std():.1f}")
print(f"    y_val   delta: mean={y_val_b.mean():.2f}  std={y_val_b.std():.1f}")
print(f"    Level gap (delta): {y_train_b.mean()-y_val_b.mean():+.2f}  (should be near 0 if differencing works)")
print(f"  Absolute reconstructed: RMSE={rmse_b:.2f}  MAE={mae_b:.2f}  R²={r2_b:.4f}")


# ── C. Diff + EWMA (only if B improves over A) ───────────────────────────────
DIFF_HELPS = rmse_b < rmse_a

print()
print(SEP)
if DIFF_HELPS:
    print("VARIANT C — ΔEC + EWMA(3,5) features  [differenced + EWMA]")
    print(SEP)

    X_c, y_c = build_features(df, "EC", diff=True, ewma=True)
    X_train_c, y_train_c, X_val_c, y_val_c, _, _ = chronological_split(X_c, y_c)

    _, preds_abs_c, preds_delta_c, rmse_c, mae_c, r2_c = run_variant(
        "C", X_train_c, y_train_c, X_val_c, y_val_c, diff=True
    )
    print(f"  Absolute reconstructed: RMSE={rmse_c:.2f}  MAE={mae_c:.2f}  R²={r2_c:.4f}")
    print(f"  Gain vs B (diff only):  {rmse_b - rmse_c:+.2f} RMSE")
else:
    print("VARIANT C — SKIPPED (differencing did not improve baseline)")
    print(SEP)
    rmse_c = mae_c = r2_c = float("nan")


# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("COMPARISON TABLE — val set, absolute EC (µS/cm)")
print(SEP)

rows = [
    ("A — EC direct (baseline)",      rmse_a, mae_a, r2_a),
    ("B — ΔEC reconstructed",         rmse_b, mae_b, r2_b),
]
if DIFF_HELPS:
    rows.append(("C — ΔEC + EWMA(3,5)",  rmse_c, mae_c, r2_c))

best_rmse = min(r[1] for r in rows)

print(f"{'Approach':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8}  vs A (RMSE)")
print("-" * 68)
for name, rmse, mae, r2 in rows:
    flag = " ◄" if rmse == best_rmse else "  "
    delta_vs_a = rmse - rmse_a
    print(f"{name:<30} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}{flag}  {delta_vs_a:>+7.2f}")

print(SEP)

# Persistence for reference
pers_rmse = float(np.sqrt(mean_squared_error(y_val_a, X_val_a["EC_lag1"].values)))
print(f"Reference — Persistence (lag1): RMSE={pers_rmse:.2f}")
print(SEP)

# ── Structural diagnosis ──────────────────────────────────────────────────────
print()
print("STRUCTURAL DIAGNOSIS")
print("-" * 68)
print(f"Train/val level gap  : {y_train_a.mean()-y_val_a.mean():+.1f} µS/cm")
print(f"Train/val delta gap  : {y_train_b.mean()-y_val_b.mean():+.2f} µS/cm  (delta space)")
print(f"Val EC std           : {y_val_a.std():.1f} µS/cm")
print(f"Best RMSE (A)        : {rmse_a:.2f}  →  RMSE/std ratio: {rmse_a/y_val_a.std():.2f}x")
if DIFF_HELPS:
    print(f"Best RMSE (B or C)   : {min(rmse_b, rmse_c):.2f}  →  RMSE/std ratio: {min(rmse_b, rmse_c)/y_val_a.std():.2f}x")
    improvement = rmse_a - min(rmse_b, rmse_c)
    print(f"Improvement          : -{improvement:.2f} RMSE ({improvement/rmse_a*100:.1f}%)")
    if min(rmse_b, rmse_c) < pers_rmse:
        print(f"✓  Best variant beats persistence ({pers_rmse:.2f}) by "
              f"{pers_rmse - min(rmse_b, rmse_c):.2f} RMSE")
    else:
        print(f"✗  Best variant still above persistence ({pers_rmse:.2f})")
else:
    print("Differencing did NOT help — level gap is not the primary error source.")
print(SEP)
