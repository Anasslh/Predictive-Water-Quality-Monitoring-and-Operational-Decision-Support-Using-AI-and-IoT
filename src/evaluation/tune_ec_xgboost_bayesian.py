"""
tune_ec_xgboost_bayesian.py — Bayesian hyperparameter search for XGBoost EC model.

Protocol:
  - Fit on X_train, score on X_val (no cross-validation, temporal data).
  - Objective: minimise RMSE on val.
  - 40 Optuna trials (TPE sampler).
  - Current reference: depth=3, n=100, lr=0.01, RMSE val=60.94.

Search space:
  max_depth        : 2–6
  n_estimators     : 50–300
  learning_rate    : 0.005–0.20  (log scale)
  subsample        : 0.60–1.00   (row subsampling — helps on small datasets)
  colsample_bytree : 0.60–1.00   (feature subsampling per tree)

Test set is never loaded here.

Run from repo root:
    python src/evaluation/tune_ec_xgboost_bayesian.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial noise

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
SEP  = "=" * 64

# ── Load splits (baseline 9-feature set) ────────────────────────────────────
X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")
print(f"Features: {X_train.columns.tolist()}\n")

CURRENT_RMSE = 60.94   # reference from manual tuning
N_TRIALS     = 40

# ── Optuna objective ─────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
        "subsample":        trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        "random_state": 42,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, preds)))


# ── Run study ────────────────────────────────────────────────────────────────
print(SEP)
print(f"Bayesian search — {N_TRIALS} trials  (TPE sampler)")
print(SEP)

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

# ── Best trial ───────────────────────────────────────────────────────────────
best = study.best_trial
best_params = best.params
best_rmse   = best.value

# Retrain with best params to get MAE and R²
best_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
best_model.fit(X_train, y_train)
best_preds  = best_model.predict(X_val)
best_mae    = float(mean_absolute_error(y_val, best_preds))
best_r2     = float(r2_score(y_val, best_preds))

# Current reference re-evaluated (same data, deterministic)
ref_model = XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.01,
                         random_state=42, verbosity=0)
ref_model.fit(X_train, y_train)
ref_preds = ref_model.predict(X_val)
ref_rmse  = float(np.sqrt(mean_squared_error(y_val, ref_preds)))
ref_mae   = float(mean_absolute_error(y_val, ref_preds))
ref_r2    = float(r2_score(y_val, ref_preds))

# ── Trial history: show top 10 ───────────────────────────────────────────────
print(f"\nTop 10 trials out of {N_TRIALS}:")
trials_sorted = sorted(study.trials, key=lambda t: t.value)[:10]
print(f"  {'#':>3}  {'RMSE':>7}  {'depth':>5}  {'n_est':>5}  {'lr':>7}  "
      f"{'sub':>5}  {'colsamp':>7}")
print("  " + "-" * 50)
for t in trials_sorted:
    p = t.params
    print(f"  {t.number:>3}  {t.value:>7.2f}  {p['max_depth']:>5}  "
          f"{p['n_estimators']:>5}  {p['learning_rate']:>7.4f}  "
          f"{p['subsample']:>5.2f}  {p['colsample_bytree']:>7.2f}")

# ── Comparison table ─────────────────────────────────────────────────────────
print()
print(SEP)
print("COMPARISON — val set")
print(SEP)
print(f"{'Config':<28} {'RMSE':>7} {'MAE':>7} {'R²':>8}  Hyperparameters")
print("-" * 64)

# Reference
print(f"{'Current (manual tuning)':<28} {ref_rmse:>7.2f} {ref_mae:>7.2f} {ref_r2:>8.4f}  "
      f"depth=3, n=100, lr=0.0100, sub=1.00, col=1.00")

# Bayesian best
p = best_params
print(f"{'Bayesian best':<28} {best_rmse:>7.2f} {best_mae:>7.2f} {best_r2:>8.4f}  "
      f"depth={p['max_depth']}, n={p['n_estimators']}, lr={p['learning_rate']:.4f}, "
      f"sub={p['subsample']:.2f}, col={p['colsample_bytree']:.2f}")

print(SEP)

# ── Gain analysis ─────────────────────────────────────────────────────────────
gain_abs  = ref_rmse - best_rmse          # positive = Bayesian is better
gain_pct  = gain_abs / ref_rmse * 100

print()
print(f"Gain vs current: {gain_abs:+.2f} RMSE  ({gain_pct:+.1f}%)")
print()

THRESHOLD_PCT = 2.5   # below this, consider it noise

if gain_abs <= 0:
    print("✗  Bayesian search did not improve the current config.")
    print("   Recommendation: keep depth=3, n=100, lr=0.01.")
elif gain_pct < THRESHOLD_PCT:
    print(f"⚠  Gain is marginal ({gain_pct:.1f}% < {THRESHOLD_PCT}% threshold).")
    print("   With only 53 val rows, differences < ~1.5 RMSE are within")
    print("   estimation noise — not a reliable signal of true improvement.")
    print(f"   Recommendation: keep current config (depth=3, n=100, lr=0.01).")
else:
    print(f"✓  Meaningful gain ({gain_pct:.1f}% > {THRESHOLD_PCT}% threshold).")
    print(f"   Recommendation: adopt Bayesian config.")
    print(f"   New config: depth={p['max_depth']}, n={p['n_estimators']}, "
          f"lr={p['learning_rate']:.4f}, subsample={p['subsample']:.2f}, "
          f"colsample_bytree={p['colsample_bytree']:.2f}")

print()
print(SEP)
print("Test set reserved — next step: final evaluation on test.")
print(SEP)
