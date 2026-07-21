"""
compare_ec_models_tuned.py — Hyperparameter tuning for EC models.

Protocol: fit on X_train, score on X_val.
No cross-validation (temporal data — shuffled CV would leak future into past).
Test set is never loaded here.

Priority: SVR > XGBoost > RF (RF already reasonable at defaults).

Run from repo root:
    python src/evaluation/compare_ec_models_tuned.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ── Load splits ──────────────────────────────────────────────────────────────
PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()

SEP = "=" * 70


def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )


# ── Baselines (no tuning) ────────────────────────────────────────────────────
persistence_preds = X_val["EC_lag1"].values
mean_train_preds  = np.full(len(y_val), y_train.mean())


# ── Helper: SVR fit/predict with y-scaling ───────────────────────────────────
def fit_predict_svr(C, epsilon, gamma, X_tr, y_tr, X_v):
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # max_iter=2000: hard limit so a bad (C, gamma) combo can't run forever.
        # ConvergenceWarning may fire but the partial solution is still usable for comparison.
        ("svr",    SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma, max_iter=2000)),
    ])
    pipe.fit(X_tr, y_scaled)
    y_scaled_pred = pipe.predict(X_v)
    return y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()


# ────────────────────────────────────────────────────────────────────────────
# 1. SVR TUNING
# ────────────────────────────────────────────────────────────────────────────
print(SEP)
print("1. SVR TUNING  (fit=train, eval=val)")
print(SEP)

# Grid: 3 × 3 × 2 = 18 configs
# Constraints from empirical testing on this 250-sample dataset:
#   - C > 100 combined with small epsilon causes QP convergence to stall (>10 min/config)
#   - gamma=0.01 (very flat kernel) is the main offender: remove it entirely
#   - epsilon is expressed in the standardised y space (y_std ≈ 1 after scaling)
#     so epsilon=2.0 ≈ 2 sigma, a wide but still meaningful tube
svr_grid = {
    "C":       [1, 10, 100],
    "epsilon": [0.5, 1.0, 2.0],
    "gamma":   ["scale", 0.1],
}

svr_best = {"rmse": np.inf, "params": {}}
svr_all  = []

configs = list(product(svr_grid["C"], svr_grid["epsilon"], svr_grid["gamma"]))
print(f"Grid size: {len(configs)} configs")

for C, eps, gam in configs:
    preds = fit_predict_svr(C, eps, gam, X_train, y_train, X_val)
    rmse, mae, r2 = metrics(y_val, preds)
    svr_all.append({"C": C, "epsilon": eps, "gamma": gam, "rmse": rmse})
    if rmse < svr_best["rmse"]:
        svr_best = {
            "rmse": rmse, "mae": mae, "r2": r2,
            "params": {"C": C, "epsilon": eps, "gamma": gam},
            "preds": preds,
        }

# Top 5
top5_svr = sorted(svr_all, key=lambda x: x["rmse"])[:5]
print(f"\nTop 5 SVR configs:")
print(f"  {'C':>8} {'epsilon':>8} {'gamma':>8} {'RMSE':>8}")
print("  " + "-" * 38)
for r in top5_svr:
    print(f"  {r['C']:>8} {r['epsilon']:>8} {str(r['gamma']):>8} {r['rmse']:>8.2f}")

p = svr_best["params"]
print(f"\nBest SVR → C={p['C']}, epsilon={p['epsilon']}, gamma={p['gamma']}")
print(f"  RMSE={svr_best['rmse']:.2f}  MAE={svr_best['mae']:.2f}  R²={svr_best['r2']:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# 2. XGBoost TUNING
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("2. XGBoost TUNING  (fit=train, eval=val)")
print(SEP)

# Grid: 4 × 4 × 4 = 64 configs
xgb_grid = {
    "max_depth":     [3, 4, 5, 6],
    "n_estimators":  [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}

xgb_best = {"rmse": np.inf, "params": {}}
xgb_all  = []

configs = list(product(xgb_grid["max_depth"], xgb_grid["n_estimators"], xgb_grid["learning_rate"]))
print(f"Grid size: {len(configs)} configs")

for depth, n_est, lr in configs:
    model = XGBRegressor(
        max_depth=depth, n_estimators=n_est, learning_rate=lr,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse, mae, r2 = metrics(y_val, preds)
    xgb_all.append({"max_depth": depth, "n_estimators": n_est, "lr": lr, "rmse": rmse})
    if rmse < xgb_best["rmse"]:
        xgb_best = {
            "rmse": rmse, "mae": mae, "r2": r2,
            "params": {"max_depth": depth, "n_estimators": n_est, "learning_rate": lr},
            "preds": preds,
        }

top5_xgb = sorted(xgb_all, key=lambda x: x["rmse"])[:5]
print(f"\nTop 5 XGBoost configs:")
print(f"  {'depth':>6} {'n_est':>6} {'lr':>6} {'RMSE':>8}")
print("  " + "-" * 32)
for r in top5_xgb:
    print(f"  {r['max_depth']:>6} {r['n_estimators']:>6} {r['lr']:>6} {r['rmse']:>8.2f}")

p = xgb_best["params"]
print(f"\nBest XGBoost → max_depth={p['max_depth']}, n_estimators={p['n_estimators']}, lr={p['learning_rate']}")
print(f"  RMSE={xgb_best['rmse']:.2f}  MAE={xgb_best['mae']:.2f}  R²={xgb_best['r2']:.4f}")

# Key check: does tuned XGBoost beat persistence?
persistence_rmse = float(np.sqrt(mean_squared_error(y_val, persistence_preds)))
if xgb_best["rmse"] < persistence_rmse:
    print(f"  ✓  Beats persistence ({persistence_rmse:.2f}) by {persistence_rmse - xgb_best['rmse']:.2f} RMSE")
else:
    print(f"  ✗  Still above persistence ({persistence_rmse:.2f}), gap: +{xgb_best['rmse'] - persistence_rmse:.2f} RMSE")
    print("     → Document as limitation; XGBoost struggles on this dataset even after tuning.")


# ────────────────────────────────────────────────────────────────────────────
# 3. RF LIGHT TUNING
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("3. RF LIGHT TUNING  (fit=train, eval=val)")
print(SEP)

# Light grid: 3 × 4 × 3 = 36 configs
rf_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 5, 10, 15],
    "min_samples_leaf":[1, 2, 4],
}

rf_best = {"rmse": np.inf, "params": {}}
rf_all  = []

configs = list(product(rf_grid["n_estimators"], rf_grid["max_depth"], rf_grid["min_samples_leaf"]))
print(f"Grid size: {len(configs)} configs")

for n_est, depth, msl in configs:
    model = RandomForestRegressor(
        n_estimators=n_est, max_depth=depth, min_samples_leaf=msl,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse, mae, r2 = metrics(y_val, preds)
    rf_all.append({"n_est": n_est, "depth": str(depth), "msl": msl, "rmse": rmse})
    if rmse < rf_best["rmse"]:
        rf_best = {
            "rmse": rmse, "mae": mae, "r2": r2,
            "params": {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": msl},
            "preds": preds,
        }

top5_rf = sorted(rf_all, key=lambda x: x["rmse"])[:5]
print(f"\nTop 5 RF configs:")
print(f"  {'n_est':>6} {'depth':>6} {'msl':>4} {'RMSE':>8}")
print("  " + "-" * 30)
for r in top5_rf:
    print(f"  {r['n_est']:>6} {r['depth']:>6} {r['msl']:>4} {r['rmse']:>8.2f}")

p = rf_best["params"]
print(f"\nBest RF → n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}")
print(f"  RMSE={rf_best['rmse']:.2f}  MAE={rf_best['mae']:.2f}  R²={rf_best['r2']:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# 4. FINAL COMPARISON TABLE
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("FINAL COMPARISON — val set  (baselines + tuned models)")
print(SEP)

_p_rmse, _p_mae, _p_r2 = metrics(y_val, persistence_preds)
_m_rmse, _m_mae, _m_r2 = metrics(y_val, mean_train_preds)

rows = [
    # Baselines
    {
        "model":  "Persistence (lag1)",
        "rmse":   _p_rmse,
        "mae":    _p_mae,
        "r2":     _p_r2,
        "params": "EC_lag1 as-is",
        "tag":    "baseline",
    },
    {
        "model":  "Mean train",
        "rmse":   _m_rmse,
        "mae":    _m_mae,
        "r2":     _m_r2,
        "params": f"mean={y_train.mean():.1f}",
        "tag":    "baseline",
    },
    # Tuned models
    {
        "model":  "SVR (tuned)",
        "rmse":   svr_best["rmse"],
        "mae":    svr_best["mae"],
        "r2":     svr_best["r2"],
        "params": f"C={svr_best['params']['C']}, ε={svr_best['params']['epsilon']}, γ={svr_best['params']['gamma']}",
        "tag":    "model",
    },
    {
        "model":  "RF (tuned)",
        "rmse":   rf_best["rmse"],
        "mae":    rf_best["mae"],
        "r2":     rf_best["r2"],
        "params": f"n={rf_best['params']['n_estimators']}, depth={rf_best['params']['max_depth']}, msl={rf_best['params']['min_samples_leaf']}",
        "tag":    "model",
    },
    {
        "model":  "XGBoost (tuned)",
        "rmse":   xgb_best["rmse"],
        "mae":    xgb_best["mae"],
        "r2":     xgb_best["r2"],
        "params": f"depth={xgb_best['params']['max_depth']}, n={xgb_best['params']['n_estimators']}, lr={xgb_best['params']['learning_rate']}",
        "tag":    "model",
    },
]


best_rmse = min(r["rmse"] for r in rows)

print(f"{'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7}  Best hyperparameters")
print("-" * 70)

baselines = [r for r in rows if r["tag"] == "baseline"]
models    = sorted([r for r in rows if r["tag"] == "model"], key=lambda x: x["rmse"])

for r in baselines + models:
    flag = " ◄" if r["rmse"] == best_rmse else "  "
    print(f"{r['model']:<22} {r['rmse']:>7.2f} {r['mae']:>7.2f} {r['r2']:>7.4f}{flag}  {r['params']}")

print(SEP)
print(f"Persistence RMSE reference: {persistence_rmse:.2f}")
print(f"Best model RMSE           : {best_rmse:.2f}  ({[r['model'] for r in rows if r['rmse']==best_rmse][0]})")
print(f"Gain vs persistence       : {persistence_rmse - best_rmse:+.2f} RMSE")
print(SEP)
print("Test set reserved for final evaluation.")
