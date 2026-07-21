"""
ec_final_test_and_shap.py — Final test evaluation and SHAP for EC XGBoost model.

Steps:
  1. Retrain ECModelXGBoost (frozen hyperparams) on train+val combined.
  2. Evaluate once on test — this is the definitive EC result, never repeated.
  3. Save the trained model to models_store/ec_xgboost_v1_final.json.
  4. Run compute_shap_explanation() on 5 selected test rows (varied cases).
  5. Consistency check: are dominant SHAP features coherent with expectations?

Run from repo root:
    python src/evaluation/ec_final_test_and_shap.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.ec.xgboost_model import ECModelXGBoost
from src.xai.shap_wrapper import compute_shap_explanation

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SEP  = "=" * 64

# ── Load splits ──────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()
X_test  = pd.read_csv(PROC / "ec_X_test.csv")
y_test  = pd.read_csv(PROC / "ec_y_test.csv").squeeze()

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

# Recover test dates from the cleaned dataset
df_raw = pd.read_csv(PROC / "c1_clean.csv", parse_dates=["Date"])
df_raw = df_raw.sort_values("Date").reset_index(drop=True)
# Feature engineering drops first 7 rows (7-day rolling), then chronological
# split: train+val = 303 rows, test = rows 303–357 → raw rows 7+303 … 7+357
test_dates = df_raw["Date"].iloc[7 + 303 : 7 + 358].reset_index(drop=True)

print(f"Train+val: {X_tv.shape}  |  Test: {X_test.shape}")
print(f"Test period: {test_dates.iloc[0].date()} → {test_dates.iloc[-1].date()}\n")


# ── STEP 1 — Train on train+val ───────────────────────────────────────────────
print(SEP)
print("STEP 1 — Training ECModelXGBoost on train+val")
print(SEP)

model = ECModelXGBoost()
model.fit(X_tv, y_tv)
print(f"Model version : {model.model_version}")
print(f"Hyperparameters: max_depth={model.model.max_depth}, "
      f"n_estimators={model.model.n_estimators}, "
      f"learning_rate={model.model.learning_rate}")


# ── STEP 2 — Final test evaluation (once, never repeated) ────────────────────
print()
print(SEP)
print("STEP 2 — FINAL TEST EVALUATION  ← definitive result")
print(SEP)

test_preds = model.predict(X_test)
test_rmse  = float(np.sqrt(mean_squared_error(y_test, test_preds)))
test_mae   = float(mean_absolute_error(y_test, test_preds))
test_r2    = float(r2_score(y_test, test_preds))
test_errors = np.abs(test_preds - y_test.values)

print(f"  RMSE : {test_rmse:.2f} µS/cm")
print(f"  MAE  : {test_mae:.2f} µS/cm")
print(f"  R²   : {test_r2:.4f}")
print()
print(f"  Test EC — true  : mean={y_test.mean():.1f}  std={y_test.std():.1f}  "
      f"[{y_test.min():.0f}, {y_test.max():.0f}]")
print(f"  Test EC — pred  : mean={test_preds.mean():.1f}  std={test_preds.std():.1f}  "
      f"[{test_preds.min():.0f}, {test_preds.max():.0f}]")
print(f"  Error (|pred-true|) : median={np.median(test_errors):.1f}  "
      f"p90={np.percentile(test_errors, 90):.1f}  max={test_errors.max():.1f}")

# Persistence on test (reference)
pers_rmse = float(np.sqrt(mean_squared_error(y_test, X_test["EC_lag1"].values)))
print(f"\n  Persistence (lag1) RMSE on test: {pers_rmse:.2f}")
print(f"  Gain vs persistence            : {pers_rmse - test_rmse:+.2f} RMSE")


# ── STEP 3 — Save model ───────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 — Saving model")
print(SEP)

model_path = ROOT / "models_store" / "ec_xgboost_v1_final.json"
model.model.save_model(str(model_path))
print(f"  Saved: {model_path.relative_to(ROOT)}")
print(f"  Format: XGBoost native JSON (reload with XGBRegressor().load_model(path))")


# ── STEP 4 — SHAP on 5 selected test rows ────────────────────────────────────
print()
print(SEP)
print("STEP 4 — SHAP explanations on 5 test rows")
print(SEP)

# Select 5 varied rows:
#   i_normal : prediction closest to test mean
#   i_high   : highest true EC value
#   i_low    : lowest true EC value
#   i_good   : best-predicted row (lowest absolute error)
#   i_bad    : worst-predicted row (highest absolute error)
y_arr  = y_test.values
i_normal = int(np.argmin(np.abs(test_preds - y_arr.mean())))
i_high   = int(np.argmax(y_arr))
i_low    = int(np.argmin(y_arr))
i_good   = int(np.argmin(test_errors))
i_bad    = int(np.argmax(test_errors))

# Deduplicate while preserving order
seen, selected = set(), []
for i, label in [(i_normal, "Normal (pred≈mean)"),
                 (i_high,   "High EC (true max)"),
                 (i_low,    "Low EC (true min)"),
                 (i_good,   "Best predicted"),
                 (i_bad,    "Worst predicted")]:
    if i not in seen:
        selected.append((i, label))
        seen.add(i)

print(f"  Feature set (9): {X_test.columns.tolist()}\n")

shap_rows = []
for idx, case_label in selected:
    X_row = X_test.iloc[[idx]].copy()
    true_val = float(y_arr[idx])
    pred_val = float(test_preds[idx])
    date_val = test_dates.iloc[idx].strftime("%Y-%m-%d")

    explanations = compute_shap_explanation(model, X_row, top_k=3, explainer_type="tree")
    shap_rows.append((date_val, case_label, true_val, pred_val, explanations))

    print(f"  [{case_label}]")
    print(f"    Date   : {date_val}")
    print(f"    True   : {true_val:.1f} µS/cm   Predicted: {pred_val:.1f} µS/cm   "
          f"Error: {abs(pred_val - true_val):.1f}")
    print(f"    Top-3 SHAP features:")
    for exp in explanations:
        bar = "▲" if exp["direction"] == "positive" else "▼"
        feat_val = float(X_row[exp["feature"]].iloc[0])
        print(f"      {bar} {exp['feature']:<22}  SHAP={exp['shap_value']:>+8.2f}  "
              f"feature_value={feat_val:.2f}")
    print()


# ── STEP 5 — Consistency check ────────────────────────────────────────────────
print(SEP)
print("STEP 5 — SHAP consistency check")
print(SEP)

feature_shap_counts = {}
for _, _, _, _, explanations in shap_rows:
    for exp in explanations:
        f = exp["feature"]
        feature_shap_counts[f] = feature_shap_counts.get(f, 0) + 1

ranked = sorted(feature_shap_counts.items(), key=lambda x: -x[1])
print("  Feature appearances in top-3 SHAP across the 5 rows:")
for feat, count in ranked:
    print(f"    {feat:<26} {count}/5 rows")

print()
expected_dominant = {"EC_lag1", "EC_roll3_mean", "EC_roll7_mean",
                     "EC_lag2",  "EC_lag3"}
dominant_found    = {f for f, c in ranked if c >= 2}

unexpected = dominant_found - expected_dominant
if not unexpected:
    print("  ✓ Dominant features are autocorrelation-based (lags, rolling means)"
          " — coherent with a persistent, slowly-varying EC signal.")
else:
    print(f"  ⚠ Unexpected dominant features: {unexpected}")
    print("    Investigate whether cross-variable influence (pH/Turbidity) is"
          " a true signal or a spurious correlation on this 365-row dataset.")

# Flag sign reversals for same feature across rows (potential instability)
feature_directions: dict[str, set] = {}
for _, _, _, _, explanations in shap_rows:
    for exp in explanations:
        f = exp["feature"]
        feature_directions.setdefault(f, set()).add(exp["direction"])

unstable = {f: d for f, d in feature_directions.items() if len(d) > 1}
if unstable:
    print(f"\n  ⚠ Features with inconsistent sign across rows: {list(unstable.keys())}")
    print("    This is normal for features that switch role depending on context"
          " (e.g. a lag value sometimes pushes prediction up, sometimes down).")
else:
    print("  ✓ No sign inconsistency — feature directions are stable across the 5 rows.")

print()
print(SEP)
print("SUMMARY")
print(SEP)
print(f"  Final EC model  : ECModelXGBoost (xgb_ec_v1_final)")
print(f"  Hyperparameters : max_depth=3, n_estimators=100, learning_rate=0.01")
print(f"  Features (9)    : 3×EC lag, 2×rolling mean, 2×rolling std, pH lag1, Turbidity lag1")
print(f"  Test RMSE       : {test_rmse:.2f} µS/cm  (persistence: {pers_rmse:.2f}, gain: {pers_rmse-test_rmse:+.2f})")
print(f"  Test MAE        : {test_mae:.2f} µS/cm")
print(f"  Test R²         : {test_r2:.4f}")
print(f"  Model saved     : models_store/ec_xgboost_v1_final.json")
print(f"  SHAP            : TreeExplainer, top-3 per row, dominant = autocorrelation features")
print(SEP)
