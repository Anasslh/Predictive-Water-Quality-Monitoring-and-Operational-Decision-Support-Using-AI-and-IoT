"""
test_shap_wrapper.py — Unit tests for src/xai/shap_wrapper.py.

Tests:
  1. auto-select dispatches TreeExplainer for XGBoost
  2. auto-select dispatches TreeExplainer for RandomForest
  3. auto-select dispatches KernelExplainer for SVR
  4. output contract: required keys present for all explainer paths
  5. "tree" and "kernel" explicit overrides still work
  6. unknown explainer_type raises ValueError
  7. KernelExplainer with no background falls back to X_row (warning, no crash)
  8. top_k is respected
  9. multiple-row X_row returns mean |SHAP| ranking
  10. "explainer" key in output matches actually used explainer
"""

import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.models.base import ParameterModel
from src.xai.shap_wrapper import compute_shap_explanation

# ── Minimal ParameterModel wrappers for testing ───────────────────────────────

class _XGBWrapper(ParameterModel):
    def __init__(self):
        super().__init__("test_param", "xgb_v0")
        self.model = XGBRegressor(n_estimators=10, max_depth=2,
                                   random_state=42, verbosity=0)
    def fit(self, X, y):
        self.model.fit(X, y); self._is_fitted = True
    def predict(self, X): self.check_fitted(); return self.model.predict(X)
    def explain(self, X): self.check_fitted(); return self.model


class _RFWrapper(ParameterModel):
    def __init__(self):
        super().__init__("test_param", "rf_v0")
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y); self._is_fitted = True
    def predict(self, X): self.check_fitted(); return self.model.predict(X)
    def explain(self, X): self.check_fitted(); return self.model


class _PredWrapper:
    def __init__(self, pipeline, y_scaler):
        self._pipeline = pipeline; self._y_scaler = y_scaler
    def predict(self, X):
        ys = self._pipeline.predict(X)
        return self._y_scaler.inverse_transform(ys.reshape(-1,1)).ravel()


class _SVRWrapper(ParameterModel):
    def __init__(self):
        super().__init__("test_param", "svr_v0")
        self.pipeline = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", C=1))])
        self.y_sc = StandardScaler()
    def fit(self, X, y):
        ys = self.y_sc.fit_transform(y.reshape(-1,1)).ravel()
        self.pipeline.fit(X, ys); self._is_fitted = True
    def predict(self, X):
        self.check_fitted()
        return self.y_sc.inverse_transform(self.pipeline.predict(X).reshape(-1,1)).ravel()
    def explain(self, X): self.check_fitted(); return _PredWrapper(self.pipeline, self.y_sc)


# ── Shared small dataset (30 samples, 4 features) ─────────────────────────────
np.random.seed(0)
N_TRAIN, N_FEAT = 30, 4
FEAT_NAMES = [f"f{i}" for i in range(N_FEAT)]
X_tr_arr = np.random.randn(N_TRAIN, N_FEAT)
y_tr_arr = X_tr_arr[:, 0] * 2 + np.random.randn(N_TRAIN) * 0.1
X_train  = pd.DataFrame(X_tr_arr, columns=FEAT_NAMES)
y_train  = pd.Series(y_tr_arr, name="y")
X_one    = X_train.iloc[[0]]          # single explanation row
X_multi  = X_train.iloc[:5]           # multi-row batch
BG       = X_train.iloc[:10]          # background for KernelExplainer

# Fit all 3 model types
xgb_m = _XGBWrapper(); xgb_m.fit(X_train, y_train)
rf_m  = _RFWrapper();  rf_m.fit(X_train, y_train)
svr_m = _SVRWrapper(); svr_m.fit(X_train, y_train.values)


# ── Test helpers ──────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"feature", "shap_value", "direction", "explainer"}
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")
        FAIL += 1


# ── Tests ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  test_shap_wrapper.py")
print("=" * 60 + "\n")

# 1. XGBoost → TreeExplainer
r = compute_shap_explanation(xgb_m, X_one, top_k=2, explainer_type="auto")
check("1. auto → tree for XGBoost", r[0]["explainer"] == "tree")

# 2. RF → TreeExplainer
r = compute_shap_explanation(rf_m, X_one, top_k=2, explainer_type="auto")
check("2. auto → tree for RandomForest", r[0]["explainer"] == "tree")

# 3. SVR → KernelExplainer
r = compute_shap_explanation(svr_m, X_one, top_k=2, explainer_type="auto", background=BG)
check("3. auto → kernel for SVR", r[0]["explainer"] == "kernel")

# 4. Output contract: required keys present (all paths)
for label, m, etype, bg in [
    ("tree",   xgb_m, "tree",   None),
    ("kernel", xgb_m, "kernel", BG),
    ("auto/svr", svr_m, "auto", BG),
]:
    r = compute_shap_explanation(m, X_one, top_k=2, explainer_type=etype, background=bg)
    ok = all(REQUIRED_KEYS <= set(d.keys()) for d in r)
    check(f"4. contract keys present ({label})", ok)

# 5a. Explicit "tree" override works
r = compute_shap_explanation(xgb_m, X_one, top_k=2, explainer_type="tree")
check("5a. explicit 'tree' override", r[0]["explainer"] == "tree")

# 5b. Explicit "kernel" override works
r = compute_shap_explanation(xgb_m, X_one, top_k=2, explainer_type="kernel", background=BG)
check("5b. explicit 'kernel' override", r[0]["explainer"] == "kernel")

# 6. Unknown explainer_type raises ValueError
try:
    compute_shap_explanation(xgb_m, X_one, top_k=2, explainer_type="badtype")
    check("6. unknown explainer_type raises ValueError", False, "no exception raised")
except ValueError:
    check("6. unknown explainer_type raises ValueError", True)

# 7. No background → falls back without crashing (warning only)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r = compute_shap_explanation(svr_m, X_one, top_k=2, explainer_type="kernel", background=None)
    warned = any("background" in str(w.message).lower() for w in caught)
check("7. no-background fallback: no crash", isinstance(r, list) and len(r) > 0)
check("7. no-background fallback: warning emitted", warned)

# 8. top_k is respected
for k in [1, 2, 4]:
    r = compute_shap_explanation(xgb_m, X_one, top_k=k, explainer_type="tree")
    check(f"8. top_k={k} respected", len(r) == k,
          f"got {len(r)}")

# 9. Multi-row X_row returns a list (mean over rows)
r = compute_shap_explanation(xgb_m, X_multi, top_k=2, explainer_type="tree")
check("9. multi-row returns list", isinstance(r, list) and len(r) == 2)
check("9. multi-row shap_values are floats",
      all(isinstance(d["shap_value"], float) for d in r))

# 10. "explainer" key matches actually used explainer
r_tree   = compute_shap_explanation(xgb_m, X_one, top_k=1, explainer_type="tree")
r_kernel = compute_shap_explanation(xgb_m, X_one, top_k=1, explainer_type="kernel", background=BG)
check("10. explainer key = 'tree' for tree path",   r_tree[0]["explainer"]   == "tree")
check("10. explainer key = 'kernel' for kernel path", r_kernel[0]["explainer"] == "kernel")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  {PASS} passed, {FAIL} failed  (out of {PASS+FAIL} checks)")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
