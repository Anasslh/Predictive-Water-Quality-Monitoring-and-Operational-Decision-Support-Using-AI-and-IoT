"""
validate_ec_forecast.py — Multi-step forecast validation on the EC pipeline.

Applies the recursive forecaster to the C-1 daily dataset and measures how
prediction error evolves step by step (48 h = 2 steps, 72 h = 3 steps).

WHAT THIS SCRIPT DOES
---------------------
1. Detect frequency on c1_clean.csv → confirm ~24 h.
2. Compute n_steps for 48 h and 72 h horizons.
3. Rebuild train / val / test splits (same chronological split as model training).
4. For each starting point in the test set, run a 3-step recursive forecast
   using full context up to that point as the historical window.
5. Report RMSE at step 1, 2, 3 separately — the key metric for characterising
   how quickly recursive error accumulates.

Run from repo root:
    python src/forecasting/validate_ec_forecast.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.feature_engineering import build_features
from src.data.split import chronological_split
from src.forecasting.frequency_detector import (
    detect_frequency,
    compute_n_steps,
    frequency_summary,
)
from src.forecasting.recursive_forecaster import MultiStepForecaster
from src.models.ec.xgboost_model import ECModelXGBoost

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
RAW_CSV   = ROOT / "data" / "processed" / "c1_clean.csv"
MODEL_JSON = ROOT / "models_store" / "ec_xgboost_v1_final.json"

SEP  = "=" * 64
SEP2 = "-" * 64

# ── 1. Load data ──────────────────────────────────────────────────────────────
print(SEP)
print("Validate: Recursive multi-step EC forecast (C-1 daily)")
print(SEP)

df = pd.read_csv(RAW_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
print(f"\nDataset: {len(df)} rows  ({df['Date'].min().date()} → {df['Date'].max().date()})")

# ── 2. Frequency detection ────────────────────────────────────────────────────
freq = detect_frequency(df, date_col="Date")
freq_hours = freq.total_seconds() / 3600.0

print(f"\n{'Frequency detection':─<50}")
print(f"  Detected frequency : {frequency_summary(freq)}  ({freq_hours:.2f} h)")

n_steps_48 = compute_n_steps(48, freq)
n_steps_72 = compute_n_steps(72, freq)
N_STEPS    = n_steps_72   # maximum horizon used for evaluation

print(f"  n_steps for  48 h  : {n_steps_48}")
print(f"  n_steps for  72 h  : {n_steps_72}")

# ── 3. Load model & build feature-engineered split ────────────────────────────
print(f"\n{'Model loading':─<50}")

model = ECModelXGBoost()
model.model.load_model(str(MODEL_JSON))
model._is_fitted = True
print(f"  Loaded: {model.model_version}")

def fe_ec(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(df_raw, target="EC")

X, y = fe_ec(df)
X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

n_trainval = len(X_train) + len(X_val)
print(f"  Train+val FE rows  : {n_trainval}")
print(f"  Test FE rows       : {len(X_test)}")

# ── 4. Rebuild split on raw data (needed for rolling window context) ──────────
# The raw df has 365 rows; build_features drops 7 (NaN boundary).
# Row i in X_test corresponds to raw row i + 7 in df.
# We use df[:n_test_raw_start + i] as the historical window for starting point i.
nan_boundary  = 7   # rows dropped by build_features (lag-7 rolling window)
raw_tv_end    = n_trainval + nan_boundary        # exclusive index in raw df for train+val context
raw_test_start = raw_tv_end                       # first test observation in raw df

print(f"  Raw rows (train+val context): 0 – {raw_tv_end}")
print(f"  Raw test rows              : {raw_test_start} – {len(df)-1}")

# ── 5. Build forecaster ───────────────────────────────────────────────────────
print(f"\n{'Forecaster setup':─<50}")

forecaster = MultiStepForecaster(
    model      = model,
    target     = "EC",
    feature_fn = fe_ec,
    frequency  = freq,
)

# ── 6. Per-step evaluation across test set ────────────────────────────────────
# For starting point i (0-indexed within test set), forecast N_STEPS ahead and
# compare to the true values at positions i, i+1, ..., i+N_STEPS-1 in X_test.
# Maximum usable starting points: len(X_test) - N_STEPS.

n_start_max   = len(X_test) - N_STEPS   # inclusive upper bound (0-indexed)
print(f"  Forecast horizon   : {N_STEPS} steps ({N_STEPS * freq_hours:.0f} h)")
print(f"  Starting points    : {n_start_max} (all usable in test set)")

residuals_by_step: list[list[float]] = [[] for _ in range(N_STEPS)]

for i in range(n_start_max):
    # Historical window = raw data up to (but not including) test row i
    raw_end = raw_test_start + i          # exclusive in raw df
    history = df.iloc[:raw_end].copy()

    # True EC values at the next N_STEPS test positions
    true_vals = y_test.values[i: i + N_STEPS]

    result = forecaster.forecast(history, N_STEPS)

    for step_idx, (true_v, pred_v) in enumerate(zip(true_vals, result.predictions)):
        residuals_by_step[step_idx].append(true_v - pred_v)

# Compute per-step metrics
rmse_by_step = [
    float(np.sqrt(np.mean(np.array(r) ** 2))) for r in residuals_by_step
]
mae_by_step = [
    float(np.mean(np.abs(np.array(r)))) for r in residuals_by_step
]
std_by_step = [
    float(np.std(r, ddof=1)) for r in residuals_by_step
]
bias_by_step = [
    float(np.mean(r)) for r in residuals_by_step
]
n_samples_by_step = [len(r) for r in residuals_by_step]

# Persist std for the figures and for the forecaster's uncertainty bands
forecaster.residual_std_by_step = std_by_step

# ── 7. Report ─────────────────────────────────────────────────────────────────
print()
print(SEP)
print("RESULTS — Error by forecast step")
print(SEP)
print(f"{'Step':>5}  {'Horizon':>8}  {'n':>5}  {'RMSE':>10}  {'MAE':>10}  {'Bias':>10}  {'Std':>10}")
print(SEP2)

for k in range(N_STEPS):
    horizon_label = f"{int((k+1) * freq_hours)}h"
    print(
        f"  {k+1:>3}  {horizon_label:>8}  {n_samples_by_step[k]:>5}  "
        f"{rmse_by_step[k]:>9.2f}  {mae_by_step[k]:>9.2f}  "
        f"{bias_by_step[k]:>9.2f}  {std_by_step[k]:>9.2f}  µS/cm"
    )

rmse_step1 = rmse_by_step[0]
rmse_last  = rmse_by_step[-1]
growth_pct = (rmse_last - rmse_step1) / rmse_step1 * 100 if rmse_step1 > 0 else float("nan")

print(SEP2)
print(f"  RMSE step-1 (24h)  : {rmse_step1:.2f} µS/cm")
print(f"  RMSE step-{N_STEPS} ({N_STEPS*24}h)  : {rmse_last:.2f} µS/cm")
print(f"  RMSE growth step1→{N_STEPS}: +{growth_pct:.1f}%")
print()

if growth_pct < 20:
    verdict = "Gentle degradation — recursive horizon viable for trend surveillance."
elif growth_pct < 50:
    verdict = "Moderate degradation — recursive horizon usable with caution."
else:
    verdict = "Steep degradation — recursive horizon limited; use only for coarse trend direction."

print(f"  Verdict: {verdict}")
print(SEP)

# ── 8. Save intermediate results for figure generation ─────────────────────────
# Store results as module-level variables so generate_figures_forecast.py
# can import this module and reuse the computation.
FORECAST_RESIDUALS_BY_STEP = residuals_by_step
RMSE_BY_STEP               = rmse_by_step
MAE_BY_STEP                = mae_by_step
STD_BY_STEP                = std_by_step
FORECASTER                 = forecaster
DF_RAW                     = df
N_STEPS_RESULT             = N_STEPS
RAW_TEST_START             = raw_test_start
Y_TEST                     = y_test
FREQ                       = freq

if __name__ == "__main__":
    # Print a good starting point for fig_13 (20 rows from test end for clean trajectory)
    example_start_i = min(10, n_start_max - 1)
    raw_end_ex = raw_test_start + example_start_i
    print(f"\n  Suggested fig_13 starting point: test index {example_start_i}")
    print(f"  Date: {df['Date'].iloc[raw_end_ex - 1].date()}")
    print(f"  True EC at t+1,t+2,t+3: "
          f"{y_test.values[example_start_i]:.0f}, "
          f"{y_test.values[example_start_i+1]:.0f}, "
          f"{y_test.values[example_start_i+2]:.0f} µS/cm")
    res = forecaster.forecast(df.iloc[:raw_end_ex].copy(), N_STEPS)
    print(f"  Predicted t+1,t+2,t+3: "
          f"{res.predictions[0]:.0f}, "
          f"{res.predictions[1]:.0f}, "
          f"{res.predictions[2]:.0f} µS/cm")
