"""
validate_ec_anomaly_detection.py — End-to-end validation of the EC anomaly detector.

Since no labelled anomalies exist in C-1, we validate on synthetic anomalies
injected into a copy of the test set (approach from A-12):

  Type 1 — Single-point spike    : one row raised by +3 sigma (~+146 µS/cm)
  Type 2 — U-shaped burst        : 4 consecutive rows raised by +2 sigma
                                   then returned to normal (simulates a
                                   short pollution episode)
  Type 3 — Negative dip          : one row lowered by -3 sigma (sensor drop,
                                   pipe flush, dilution event)

Validation protocol:
  - Fit ECAnomalyDetector on train+val residuals (normal reference).
  - Score the CLEAN test set → baseline FP rate.
  - Score the SYNTHETIC test set → detection rate (TP recall on labelled rows).
  - Print coupled SHAP explanations for 3 detected anomalies.

Run from repo root:
    python src/anomaly/validate_ec_anomaly_detection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd

from src.models.ec.xgboost_model import ECModelXGBoost
from src.anomaly.residual import compute_residuals
from src.anomaly.detector import ECAnomalyDetector
from src.anomaly.explain import explain_anomaly

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SEP  = "=" * 66

# ── Load all splits ──────────────────────────────────────────────────────────
X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()
X_test  = pd.read_csv(PROC / "ec_X_test.csv")
y_test  = pd.read_csv(PROC / "ec_y_test.csv").squeeze()

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

# Recover test dates
df_raw   = pd.read_csv(PROC / "c1_clean.csv", parse_dates=["Date"])
df_raw   = df_raw.sort_values("Date").reset_index(drop=True)
test_dates = [
    df_raw["Date"].iloc[7 + 303 + i].strftime("%Y-%m-%d")
    for i in range(len(X_test))
]

EC_STD = float(y_tv.std())   # ~212 µS/cm from train+val
N_TEST = len(X_test)

print(f"Train+val: {X_tv.shape}  |  Test: {X_test.shape}")
print(f"EC std (train+val reference): {EC_STD:.1f} µS/cm\n")


# ── STEP 1 — Train ECModelXGBoost on train+val ───────────────────────────────
print(SEP)
print("STEP 1 — Train ECModelXGBoost (frozen hyperparams)")
print(SEP)

model = ECModelXGBoost()
model.fit(X_tv, y_tv)
print(f"  Model: {model.model_version}  |  "
      f"max_depth={model.model.max_depth}, "
      f"n_estimators={model.model.n_estimators}, "
      f"lr={model.model.learning_rate}")


# ── STEP 2 — Compute residuals on train+val (normal reference) ───────────────
print()
print(SEP)
print("STEP 2 — Residuals on train+val  (normal reference for detector)")
print(SEP)

res_tv = compute_residuals(model, X_tv, y_tv)
print(f"  Residuals: mean={res_tv.mean():.2f}  std={res_tv.std():.2f}  "
      f"min={res_tv.min():.1f}  max={res_tv.max():.1f}")


# ── STEP 3 — Fit ECAnomalyDetector ──────────────────────────────────────────
print()
print(SEP)
print("STEP 3 — Fit ECAnomalyDetector on train+val residuals")
print(SEP)

detector = ECAnomalyDetector(contamination=0.05, threshold=0.5, random_state=42)
detector.fit(res_tv)
print(f"  contamination={detector.contamination}  threshold={detector.threshold}")
print(f"  IsolationForest n_estimators=100")

# Sanity: score distribution on the training data itself
train_scores = detector.score(res_tv)
tv_flags     = detector.predict(res_tv)
print(f"  Train+val score: mean={train_scores.mean():.3f}  "
      f"p90={np.percentile(train_scores, 90):.3f}  max={train_scores.max():.3f}")
print(f"  Flagged in train+val: {tv_flags.sum()} / {len(tv_flags)} "
      f"({tv_flags.sum()/len(tv_flags)*100:.1f}%)  "
      f"[expected ~{int(detector.contamination*len(tv_flags))} given contamination={detector.contamination}]")


# ── STEP 4 — Baseline: score CLEAN test set ───────────────────────────────────
print()
print(SEP)
print("STEP 4 — Score CLEAN test set  (false-positive baseline)")
print(SEP)

res_test_clean = compute_residuals(model, X_test, y_test)
scores_clean   = detector.score(res_test_clean)
flags_clean    = detector.predict(res_test_clean)

fp_rate = flags_clean.sum() / N_TEST
print(f"  Test residuals: mean={res_test_clean.mean():.2f}  "
      f"std={res_test_clean.std():.2f}  "
      f"min={res_test_clean.min():.1f}  max={res_test_clean.max():.1f}")
print(f"  Anomaly scores: mean={scores_clean.mean():.3f}  "
      f"p90={np.percentile(scores_clean, 90):.3f}  max={scores_clean.max():.3f}")
print(f"  Flagged (clean): {flags_clean.sum()} / {N_TEST}  "
      f"→ false-positive rate = {fp_rate*100:.1f}%")


# ── STEP 5 — Inject synthetic anomalies ─────────────────────────────────────
print()
print(SEP)
print("STEP 5 — Inject synthetic anomalies into test copy  (A-12 protocol)")
print(SEP)

# Only y_test is modified (actual EC values); X_test is unchanged because
# features are lag-based history — exactly what happens with a real sensor
# fault: the model predicts normally, but the sensor sends a bad reading.
y_synth     = y_test.copy().values.astype(float)
synth_labels = np.zeros(N_TEST, dtype=bool)   # True = synthetic anomaly

SIGMA = res_tv.std()   # residual std, used as injection magnitude reference

# -- Type 1: Single-point positive spike (index 10) -------------------------
idx_spike = 10
spike_magnitude = +3 * EC_STD
y_synth[idx_spike] += spike_magnitude
synth_labels[idx_spike] = True
print(f"  Type 1 — Single spike  @ index {idx_spike} ({test_dates[idx_spike]}): "
      f"EC {y_test.iloc[idx_spike]:.0f} → {y_synth[idx_spike]:.0f} µS/cm  "
      f"(+{spike_magnitude:.0f}, +3σ_EC)")

# -- Type 2: U-shaped burst (indices 25–28, 4 consecutive points) ------------
burst_idx = list(range(25, 29))
burst_magnitude = +2 * EC_STD
for i in burst_idx:
    y_synth[i] += burst_magnitude
    synth_labels[i] = True
print(f"  Type 2 — U-shaped burst @ indices {burst_idx[0]}–{burst_idx[-1]} "
      f"({test_dates[burst_idx[0]]}–{test_dates[burst_idx[-1]]}): "
      f"+{burst_magnitude:.0f} µS/cm (+2σ_EC) over 4 days")

# -- Type 3: Negative dip (index 40) ----------------------------------------
idx_dip = 40
dip_magnitude = -3 * EC_STD
y_synth[idx_dip] += dip_magnitude
synth_labels[idx_dip] = True
print(f"  Type 3 — Negative dip  @ index {idx_dip} ({test_dates[idx_dip]}): "
      f"EC {y_test.iloc[idx_dip]:.0f} → {y_synth[idx_dip]:.0f} µS/cm  "
      f"({dip_magnitude:.0f}, -3σ_EC)")

print(f"\n  Total synthetic anomaly points: {synth_labels.sum()} / {N_TEST}")


# ── STEP 6 — Score synthetic test set ────────────────────────────────────────
print()
print(SEP)
print("STEP 6 — Score SYNTHETIC test set")
print(SEP)

res_synth   = compute_residuals(model, X_test, pd.Series(y_synth))
scores_synth = detector.score(res_synth)
flags_synth  = detector.predict(res_synth)

# Detection metrics on synthetic labels
tp = int(( flags_synth &  synth_labels).sum())
fp = int(( flags_synth & ~synth_labels).sum())
fn = int((~flags_synth &  synth_labels).sum())
tn = int((~flags_synth & ~synth_labels).sum())

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"  Synthetic anomaly rows  : {synth_labels.sum()}")
print(f"  Detected as anomaly     : {flags_synth.sum()}")
print()
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  Precision : {precision:.2f}")
print(f"  Recall    : {recall:.2f}")
print(f"  F1        : {f1:.2f}")
print(f"  FP rate on clean rows: {fp}/{N_TEST-synth_labels.sum()} = {fp/(N_TEST-synth_labels.sum())*100:.1f}%")

# Per-anomaly-type breakdown
print()
for label, indices in [
    ("Spike  (idx 10)",       [idx_spike]),
    ("Burst  (idx 25–28)",    burst_idx),
    ("Dip    (idx 40)",       [idx_dip]),
]:
    detected = sum(flags_synth[i] for i in indices)
    scores_  = [f"{scores_synth[i]:.3f}" for i in indices]
    print(f"  {label}: detected {detected}/{len(indices)}  scores={scores_}")


# ── STEP 7 — Score table: all 55 test rows ──────────────────────────────────
print()
print(SEP)
print("STEP 7 — Score table (synthetic test, 55 rows)")
print(SEP)
print(f"  {'#':>3}  {'Date':>12}  {'True':>6}  {'Synth':>6}  "
      f"{'Pred':>6}  {'Resid':>7}  {'Score':>6}  {'Flag':>5}  {'Injected':>8}")
print("  " + "-" * 64)

pred_tv = model.predict(X_test)
for i in range(N_TEST):
    flag   = "ANOM" if flags_synth[i] else "ok"
    injected = "★" if synth_labels[i] else ""
    print(
        f"  {i:>3}  {test_dates[i]:>12}  "
        f"{y_test.iloc[i]:>6.0f}  {y_synth[i]:>6.0f}  "
        f"{pred_tv[i]:>6.1f}  {res_synth[i]:>+7.1f}  "
        f"{scores_synth[i]:>6.3f}  {flag:>5}  {injected:>8}"
    )


# ── STEP 8 — Coupled SHAP explanations on 3 detected anomalies ──────────────
print()
print(SEP)
print("STEP 8 — Coupled SHAP explanations (score + SHAP) on detected anomalies")
print(SEP)

detected_anomaly_idx = [i for i in range(N_TEST) if flags_synth[i] and synth_labels[i]]
example_idx = detected_anomaly_idx[:3]   # up to 3 examples

for rank, i in enumerate(example_idx, 1):
    anom_type = (
        "Type 1 — spike"       if i == idx_spike else
        "Type 2 — burst"       if i in burst_idx  else
        "Type 3 — dip"         if i == idx_dip    else
        "other"
    )
    result = explain_anomaly(
        model     = model,
        detector  = detector,
        X_row     = X_test.iloc[[i]],
        y_actual  = float(y_synth[i]),
        timestamp = test_dates[i],
        top_k     = 3,
    )
    print(f"  Example {rank} — {anom_type}")
    print(f"    Timestamp      : {result.timestamp}")
    print(f"    Actual EC      : {result.actual_value:.1f} µS/cm  "
          f"(original: {y_test.iloc[i]:.1f})")
    print(f"    Predicted EC   : {result.predicted_value:.1f} µS/cm")
    print(f"    Residual       : {result.residual:+.1f} µS/cm")
    print(f"    Anomaly score  : {result.anomaly_score:.4f}  "
          f"({'ANOMALY' if result.is_anomaly else 'normal'}, threshold={detector.threshold})")
    print(f"    SHAP top-3     :")
    for feat in result.shap_top_features:
        bar = "▲" if feat["direction"] == "positive" else "▼"
        feat_val = float(X_test.iloc[i][feat["feature"]])
        print(f"      {bar} {feat['feature']:<24} SHAP={feat['shap_value']:>+8.2f}  "
              f"feat={feat_val:.2f}")
    print()


# ── STEP 9 — Summary ─────────────────────────────────────────────────────────
print(SEP)
print("SUMMARY")
print(SEP)
print(f"  Detector            : Isolation Forest on residuals (contamination={detector.contamination})")
print(f"  Reference period    : train+val ({len(X_tv)} rows), residual mean={res_tv.mean():.2f} std={res_tv.std():.2f}")
print(f"  Threshold           : {detector.threshold} (A-12 PADSV convention)")
print()
print(f"  Clean test FP rate  : {fp_rate*100:.1f}%  ({flags_clean.sum()}/{N_TEST} flags on unmodified data)")
print(f"  Synthetic precision : {precision:.2f}")
print(f"  Synthetic recall    : {recall:.2f}  (on {synth_labels.sum()} injected anomaly points)")
print(f"  F1 score            : {f1:.2f}")
print()
if recall >= 0.8 and fp_rate <= 0.15:
    print("  ✓ Detector behaves as expected: high recall on synthetic anomalies,")
    print("    contained false-positive rate on clean data.")
elif recall < 0.5:
    print("  ⚠ Recall is low — consider lowering the threshold or increasing contamination.")
else:
    print(f"  ~ Recall={recall:.2f} / FP={fp_rate*100:.1f}% — review threshold if FP rate is too high.")
print(SEP)
