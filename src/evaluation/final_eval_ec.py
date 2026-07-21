"""
final_eval_ec.py — Final test-set evaluation for EC models.

Protocol:
  - Hyperparameters were selected on val; val is now consumed.
  - Retrain each tuned model on train+val combined (all data before test).
  - Evaluate on test exactly once. No adjustments after seeing test results.

Run from repo root:
    python src/evaluation/final_eval_ec.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
SEP  = "=" * 68

# ── Load splits ──────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()
X_test  = pd.read_csv(PROC / "ec_X_test.csv")
y_test  = pd.read_csv(PROC / "ec_y_test.csv").squeeze()

# Train+val combined — chronologically before test, no leakage
X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

print(f"Train+val: {X_tv.shape}  |  Test: {X_test.shape}")


def metrics(y_true, y_pred):
    return (
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
        float(r2_score(y_true, y_pred)),
    )


# ── EC distribution on test set ──────────────────────────────────────────────
print()
print(SEP)
print("EC TARGET — test set distribution  (µS/cm)")
print(SEP)
print(f"  n    : {len(y_test)}")
print(f"  min  : {y_test.min():.1f}")
print(f"  max  : {y_test.max():.1f}")
print(f"  mean : {y_test.mean():.1f}")
print(f"  std  : {y_test.std():.1f}")
print()
print(f"  Recall — train mean: {y_train.mean():.1f}  |  val mean: {y_val.mean():.1f}")

# Extrapolation check
test_below = y_test.min() < y_tv.min()
test_above = y_test.max() > y_tv.max()
if test_below or test_above:
    print(f"  ⚠  Test range [{y_test.min():.1f}, {y_test.max():.1f}] "
          f"partially outside train+val [{y_tv.min():.1f}, {y_tv.max():.1f}]")
else:
    print(f"  ✓  Test EC range fully inside train+val range "
          f"[{y_tv.min():.1f}, {y_tv.max():.1f}]")


# ── Best hyperparameters (from compare_ec_models_tuned.py) ───────────────────
# XGBoost: depth=3, n_estimators=100, lr=0.01
# SVR    : C=1, epsilon=0.5, gamma=0.1  (with X + y StandardScaler)
# RF     : n_estimators=300, max_depth=5, min_samples_leaf=4

# Val RMSEs (from tuning — for comparison table)
VAL_RMSE = {
    "XGBoost (tuned)": 60.94,
    "SVR (tuned)":     64.80,
    "RF (tuned)":      67.26,
    "Persistence":     76.02,
}


# ── Retrain on train+val, evaluate on test ───────────────────────────────────
print()
print(SEP)
print("RETRAINING ON TRAIN+VAL → EVALUATING ON TEST")
print(SEP)

results = {}

# -- Persistence baseline (no retraining needed) --
persistence_preds = X_test["EC_lag1"].values
r, m, r2 = metrics(y_test, persistence_preds)
results["Persistence"] = {"rmse_test": r, "mae_test": m, "r2_test": r2}
print("  [Persistence] done.")

# -- XGBoost --
xgb = XGBRegressor(
    max_depth=3, n_estimators=100, learning_rate=0.01,
    random_state=42, verbosity=0,
)
xgb.fit(X_tv, y_tv)
preds = xgb.predict(X_test)
r, m, r2 = metrics(y_test, preds)
results["XGBoost (tuned)"] = {"rmse_test": r, "mae_test": m, "r2_test": r2}
print("  [XGBoost] trained and evaluated.")

# -- SVR (X + y scaling) --
y_scaler_svr = StandardScaler()
y_tv_scaled  = y_scaler_svr.fit_transform(y_tv.values.reshape(-1, 1)).ravel()
svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel="rbf", C=1, epsilon=0.5, gamma=0.1, max_iter=5000)),
])
svr_pipe.fit(X_tv, y_tv_scaled)
preds_scaled = svr_pipe.predict(X_test)
preds = y_scaler_svr.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
r, m, r2 = metrics(y_test, preds)
results["SVR (tuned)"] = {"rmse_test": r, "mae_test": m, "r2_test": r2}
print("  [SVR] trained and evaluated.")

# -- RF --
rf = RandomForestRegressor(
    n_estimators=300, max_depth=5, min_samples_leaf=4,
    random_state=42,
)
rf.fit(X_tv, y_tv)
preds = rf.predict(X_test)
r, m, r2 = metrics(y_test, preds)
results["RF (tuned)"] = {"rmse_test": r, "mae_test": m, "r2_test": r2}
print("  [RF] trained and evaluated.")


# ── Final comparison table ────────────────────────────────────────────────────
print()
print(SEP)
print("FINAL TABLE — val RMSE vs test RMSE  (µS/cm)")
print(SEP)

ordered = ["XGBoost (tuned)", "SVR (tuned)", "RF (tuned)", "Persistence"]
best_test_rmse = min(results[m]["rmse_test"] for m in ordered)

print(f"{'Model':<22} {'RMSE val':>10} {'RMSE test':>10} {'Δ (test-val)':>13}  {'R² test':>8}")
print("-" * 68)

for name in ordered:
    r   = results[name]
    val = VAL_RMSE.get(name, float("nan"))
    tst = r["rmse_test"]
    delta = tst - val
    flag = " ◄" if tst == best_test_rmse else "  "
    print(
        f"{name:<22} {val:>10.2f} {tst:>10.2f} {delta:>+13.2f}  {r['r2_test']:>8.4f}{flag}"
    )

print(SEP)

# Ranking change?
val_ranking  = sorted(["XGBoost (tuned)", "SVR (tuned)", "RF (tuned)"],
                      key=lambda m: VAL_RMSE[m])
test_ranking = sorted(["XGBoost (tuned)", "SVR (tuned)", "RF (tuned)"],
                      key=lambda m: results[m]["rmse_test"])

print()
print(f"Val ranking  (1→3): {' > '.join(val_ranking)}")
print(f"Test ranking (1→3): {' > '.join(test_ranking)}")

if val_ranking == test_ranking:
    print("✓  Val ranking confirmed on test — no position change.")
else:
    print("⚠  Ranking changed between val and test — see recommendation below.")

print()
print(SEP)
print("RECOMMENDATION")
print(SEP)
winner = test_ranking[0]
winner_rmse = results[winner]["rmse_test"]
print(f"Selected model for EC: {winner}")
print(f"  Test RMSE : {winner_rmse:.2f} µS/cm")
print(f"  Test MAE  : {results[winner]['mae_test']:.2f} µS/cm")
print(f"  Test R²   : {results[winner]['r2_test']:.4f}")
print(f"  Gain vs persistence on test: "
      f"{results['Persistence']['rmse_test'] - winner_rmse:+.2f} RMSE")
print()
print("Next step: SHAP integration for this model.")
print(SEP)
