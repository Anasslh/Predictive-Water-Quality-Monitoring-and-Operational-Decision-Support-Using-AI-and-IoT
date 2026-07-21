"""
compare_ec_feature_engineering.py — Feature engineering experiments for EC.

Tests, all with XGBoost (depth=3, n=100, lr=0.01) on EC raw target:
  A   : baseline — current feature set (9 features)
  B3  : + EWMA(span=3)
  B5  : + EWMA(span=5)
  B35 : + EWMA(span=3) + EWMA(span=5)
  C   : + lag7 + lag14   (drops 14 rows instead of 7 → train=243)
  D   : + rolling slope (lag1-lag7)/6
  E   : best combination of individually-improving options
  F   : ensemble (XGBoost + SVR + RF) on the best feature set

Evaluation: val set only. Test set is not loaded.

Run from repo root:
    python src/evaluation/compare_ec_feature_engineering.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.data.feature_engineering import build_features
from src.data.split import chronological_split

warnings.filterwarnings("ignore")

SEP  = "=" * 68
DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "c1_clean.csv"
df   = pd.read_csv(DATA, parse_dates=["Date"])

XGB_PARAMS = dict(max_depth=3, n_estimators=100, learning_rate=0.01,
                  random_state=42, verbosity=0)
RF_PARAMS  = dict(n_estimators=300, max_depth=5, min_samples_leaf=4, random_state=42)
SVR_C, SVR_EPS, SVR_GAMMA = 1, 0.5, 0.1


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return (
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
        float(r2_score(y_true, y_pred)),
    )


def train_xgb(X_tr, y_tr, X_v):
    m = XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, y_tr)
    return m, m.predict(X_v)


def train_svr(X_tr, y_tr, X_v):
    y_sc = StandardScaler()
    y_tr_sc = y_sc.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
    pipe = Pipeline([("sc", StandardScaler()),
                     ("svr", SVR(kernel="rbf", C=SVR_C, epsilon=SVR_EPS,
                                 gamma=SVR_GAMMA, max_iter=5000))])
    pipe.fit(X_tr, y_tr_sc)
    preds_sc = pipe.predict(X_v)
    return pipe, y_sc.inverse_transform(preds_sc.reshape(-1, 1)).ravel()


def train_rf(X_tr, y_tr, X_v):
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(X_tr, y_tr)
    return m, m.predict(X_v)


def evaluate(label, X_tr, y_tr, X_v, y_v, *, xgb_only=True, note=""):
    """Train XGBoost (and optionally all 3 models for ensemble) and return results dict."""
    _, xgb_preds = train_xgb(X_tr, y_tr, X_v)
    rmse, mae, r2 = metrics(y_v, xgb_preds)

    row = {"label": label, "rmse": rmse, "mae": mae, "r2": r2,
           "n_feat": X_tr.shape[1], "n_train": len(X_tr), "note": note}

    if not xgb_only:
        _, svr_preds = train_svr(X_tr, y_tr, X_v)
        _, rf_preds  = train_rf(X_tr, y_tr, X_v)
        ens_preds    = (xgb_preds + svr_preds + rf_preds) / 3
        e_rmse, e_mae, e_r2 = metrics(y_v, ens_preds)
        row.update({"ens_rmse": e_rmse, "ens_mae": e_mae, "ens_r2": e_r2,
                    "xgb_preds": xgb_preds, "svr_preds": svr_preds,
                    "rf_preds": rf_preds,   "ens_preds": ens_preds})
    return row


# ── Helper: build + split ─────────────────────────────────────────────────────
def prepare(**kwargs):
    X, y = build_features(df, "EC", **kwargs)
    X_tr, y_tr, X_v, y_v, _, _ = chronological_split(X, y)
    return X_tr, y_tr, X_v, y_v


print(SEP)
print("EC FEATURE ENGINEERING EXPERIMENTS — val set only")
print(SEP)

results = []

# ── A. Baseline ───────────────────────────────────────────────────────────────
print("\n[A] Baseline")
X_tr, y_tr, X_v, y_v = prepare()
r = evaluate("A — Baseline (9 feat)", X_tr, y_tr, X_v, y_v)
results.append(r)
BASELINE_RMSE = r["rmse"]
print(f"    RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R²={r['r2']:.4f}  "
      f"(train={r['n_train']}, features={r['n_feat']})")

# ── B. EWMA variants ──────────────────────────────────────────────────────────
for spans, lbl in [([3], "B3 — +EWMA(3)"), ([5], "B5 — +EWMA(5)"), ([3, 5], "B35 — +EWMA(3,5)")]:
    print(f"\n[{lbl}]")
    X_tr, y_tr, X_v, y_v = prepare(ewma_spans=spans)
    r = evaluate(lbl, X_tr, y_tr, X_v, y_v)
    r["ewma_spans"] = spans
    results.append(r)
    diff = r["rmse"] - BASELINE_RMSE
    print(f"    RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R²={r['r2']:.4f}  "
          f"vs A: {diff:+.2f}")

# ── C. Extra lags: lag7 + lag14 ───────────────────────────────────────────────
print("\n[C] +lag7 +lag14")
X_tr, y_tr, X_v, y_v = prepare(extra_lags=[7, 14])
r = evaluate("C — +lag7+lag14", X_tr, y_tr, X_v, y_v,
             note=f"train={len(X_tr)} (−{250-len(X_tr)} vs baseline)")
results.append(r)
diff = r["rmse"] - BASELINE_RMSE
print(f"    RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R²={r['r2']:.4f}  "
      f"vs A: {diff:+.2f}  (train={r['n_train']}, {r['note']})")

# ── D. Rolling slope ──────────────────────────────────────────────────────────
print("\n[D] +slope7")
X_tr, y_tr, X_v, y_v = prepare(rolling_slope=True)
r = evaluate("D — +slope7", X_tr, y_tr, X_v, y_v)
results.append(r)
diff = r["rmse"] - BASELINE_RMSE
print(f"    RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R²={r['r2']:.4f}  "
      f"vs A: {diff:+.2f}")

# ── E. Best combination ───────────────────────────────────────────────────────
# Select variants that individually beat the baseline
winners = [r for r in results[1:] if r["rmse"] < BASELINE_RMSE]
print(f"\n[E] Best combination")
print(f"    Individual improvements over baseline: {[r['label'][:4] for r in winners]}")

if len(winners) == 0:
    print("    No individual variant improved baseline — skipping combination.")
    r_e = None
elif len(winners) == 1:
    print(f"    Only one winner ({winners[0]['label'][:4]}) — combination = that variant alone.")
    r_e = winners[0].copy()
    r_e["label"] = f"E — Best combo ({winners[0]['label'][:4]})"
    results.append(r_e)
else:
    # Build combined feature set from all winning variants
    ewma_combined    = []
    extra_lags_combo = []
    slope_combo      = False
    for r_w in winners:
        lbl = r_w["label"]
        if "EWMA" in lbl or "ewma" in lbl:
            ewma_combined.extend(r_w.get("ewma_spans", []))
        if "lag7" in lbl or "lag14" in lbl:
            extra_lags_combo.extend([7, 14])
        if "slope" in lbl:
            slope_combo = True
    ewma_combined = sorted(set(ewma_combined))

    X_tr, y_tr, X_v, y_v = prepare(
        ewma_spans=ewma_combined or None,
        extra_lags=extra_lags_combo or None,
        rolling_slope=slope_combo,
    )
    r_e = evaluate(
        f"E — Combo ({'+'.join(r['label'][:4].strip() for r in winners)})",
        X_tr, y_tr, X_v, y_v
    )
    results.append(r_e)
    diff = r_e["rmse"] - BASELINE_RMSE
    print(f"    RMSE={r_e['rmse']:.2f}  MAE={r_e['mae']:.2f}  R²={r_e['r2']:.4f}  "
          f"vs A: {diff:+.2f}")

# ── F. Ensemble: XGBoost + SVR + RF on best feature set ──────────────────────
print(f"\n[F] Ensemble (XGB+SVR+RF) on best feature set")

# Best single-model config so far
best_single = min(results, key=lambda r: r["rmse"])
print(f"    Best single-model so far: {best_single['label']} (RMSE={best_single['rmse']:.2f})")

# Rebuild features matching the best config
if "Combo" in best_single["label"] or best_single is r_e:
    combo_kwargs = dict(
        ewma_spans=ewma_combined or None,
        extra_lags=extra_lags_combo or None,
        rolling_slope=slope_combo,
    )
elif "EWMA" in best_single["label"]:
    combo_kwargs = dict(ewma_spans=best_single.get("ewma_spans", []))
elif "lag7" in best_single["label"]:
    combo_kwargs = dict(extra_lags=[7, 14])
elif "slope" in best_single["label"]:
    combo_kwargs = dict(rolling_slope=True)
else:
    combo_kwargs = {}

X_tr, y_tr, X_v, y_v = prepare(**combo_kwargs)
r_f = evaluate("F — Ensemble (XGB+SVR+RF)", X_tr, y_tr, X_v, y_v, xgb_only=False)
results.append(r_f)

diff_xgb = r_f["rmse"]       - BASELINE_RMSE
diff_ens = r_f["ens_rmse"]   - BASELINE_RMSE
print(f"    XGBoost alone : RMSE={r_f['rmse']:.2f}  MAE={r_f['mae']:.2f}  R²={r_f['r2']:.4f}  vs A: {diff_xgb:+.2f}")
print(f"    Ensemble avg  : RMSE={r_f['ens_rmse']:.2f}  MAE={r_f['ens_mae']:.2f}  R²={r_f['ens_r2']:.4f}  vs A: {diff_ens:+.2f}")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("SUMMARY TABLE — val set, EC µS/cm  (XGBoost unless noted)")
print(SEP)
print(f"{'Approach':<38} {'RMSE':>7} {'MAE':>7} {'R²':>7}  {'vs A':>7}  {'n_feat':>6}")
print("-" * 68)

# Collect rows: single-model variants + ensemble as separate row
display_rows = results.copy()
# Add ensemble as separate entry
ens_row = {
    "label": "F — Ensemble (XGB+SVR+RF)",
    "rmse": r_f["ens_rmse"], "mae": r_f["ens_mae"], "r2": r_f["ens_r2"],
    "n_feat": r_f["n_feat"], "note": "avg of 3 models",
}
display_rows.append(ens_row)

all_rmse = [r["rmse"] for r in display_rows]
best_rmse_overall = min(all_rmse)

for r in display_rows:
    flag = " ◄" if r["rmse"] == best_rmse_overall else "  "
    vs_a = r["rmse"] - BASELINE_RMSE
    print(f"{r['label']:<38} {r['rmse']:>7.2f} {r['mae']:>7.2f} {r['r2']:>7.4f}  {vs_a:>+7.2f}  {r['n_feat']:>6}{flag}")

print(SEP)

# Reference
pers_preds = X_v["EC_lag1"].values  # using last val split (same rows for baseline features)
# Re-get baseline val for persistence
X_tr_a, y_tr_a, X_v_a, y_v_a = prepare()
pers_rmse = float(np.sqrt(mean_squared_error(y_v_a, X_v_a["EC_lag1"].values)))
print(f"Reference — Persistence: RMSE={pers_rmse:.2f}")
print(SEP)

# ── Final recommendation ──────────────────────────────────────────────────────
winner = min(display_rows, key=lambda r: r["rmse"])
gain   = BASELINE_RMSE - winner["rmse"]
print(f"\nBest overall : {winner['label']}")
print(f"RMSE val     : {winner['rmse']:.2f}  (baseline: {BASELINE_RMSE:.2f},  gain: {gain:+.2f})")
