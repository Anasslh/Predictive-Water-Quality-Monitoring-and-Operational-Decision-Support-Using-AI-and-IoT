"""
validate_ec_anomaly_intensity.py — Detection-limit characterization for ECAnomalyDetector.

Sweeps 6 injection intensities (0.5σ – 3.0σ, where σ = residual std from
train+val) across 3 anomaly types (spike, burst, dip) with 15 repetitions per
cell, to identify the intensity at which the detector achieves reliable recall.

Injection protocol (A-12 extended):
  - Only y_actual is modified; X features (lag-based history) are unchanged.
    This matches a real sensor fault: the model predicts normally from past
    history while the current reading is corrupted.
  - Spike : single-point positive injection.
  - Burst : 4 consecutive points raised by the same magnitude.
  - Dip   : single-point negative injection.
  - 15 uniformly spaced injection positions across the test set per cell.

Metrics aggregated across all 15 reps per cell (point-level):
  precision, recall, F1, mean anomaly score on injected points.
  Burst also reports event-level recall (≥ 1 of 4 consecutive points flagged).

σ reference: residual_std from train+val (the Isolation Forest's own training
distribution) — more principled than EC_STD for a residual-based detector.

Outputs:
  - Console: FP confirmation, per-type summary tables, tipping points.
  - reports/ec_anomaly_intensity_sweep.png (recall vs intensity, 3 lines).

Run from repo root:
    python src/anomaly/validate_ec_anomaly_intensity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.models.ec.xgboost_model import ECModelXGBoost
from src.anomaly.residual import compute_residuals
from src.anomaly.detector import ECAnomalyDetector

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT   = Path(__file__).resolve().parents[2]
PROC   = ROOT / "data" / "processed"
REPORT = ROOT / "reports"
REPORT.mkdir(exist_ok=True)

# ── Sweep configuration ───────────────────────────────────────────────────────
INTENSITIES    = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # multiples of residual σ
N_REPS         = 15                                  # repetitions per cell
BURST_LEN      = 4                                   # consecutive points per burst
RECALL_FLOOR   = 0.80                                # tipping-point threshold

# Reference palette: slots 1/3/5 (blue / amber / violet), CVD-safe in light mode.
COLORS = {
    "spike": "#2a78d6",   # slot 1 — blue
    "burst": "#eda100",   # slot 3 — amber
    "dip":   "#4a3aa7",   # slot 5 — violet
}

SEP = "=" * 72

# ── Load splits ───────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()
X_test  = pd.read_csv(PROC / "ec_X_test.csv")
y_test  = pd.read_csv(PROC / "ec_y_test.csv").squeeze()

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

N_TEST = len(X_test)

# ── Fit model + detector (frozen — no retuning) ───────────────────────────────
print(SEP)
print("SETUP — Fit model + detector on train+val (frozen config)")
print(SEP)

model = ECModelXGBoost()
model.fit(X_tv, y_tv)

res_tv   = compute_residuals(model, X_tv, y_tv)
SIGMA    = float(res_tv.std())   # residual std = injection σ reference

detector = ECAnomalyDetector(contamination=0.05, threshold=0.5, random_state=42)
detector.fit(res_tv)

# FP baseline: score the unmodified test set once
res_clean   = compute_residuals(model, X_test, y_test)
flags_clean = detector.predict(res_clean)
fp_clean    = int(flags_clean.sum())
fp_rate     = fp_clean / N_TEST

print(f"  Model        : {model.model_version}  (depth=3, n=100, lr=0.01)")
print(f"  Residual σ   : {SIGMA:.2f} µS/cm  ← injection reference")
print(f"  Test N       : {N_TEST}")
print(f"  Threshold    : {detector.threshold}")
print(f"  FP on clean  : {fp_clean}/{N_TEST}  ({fp_rate*100:.1f}%)  ← false-positive baseline")
print()

# ── Injection positions (uniformly spaced, edge-safe) ─────────────────────────
# Spike/Dip: avoid the first 2 and last 2 indices (edge lag effects).
spike_positions = np.unique(np.linspace(2, N_TEST - 3, N_REPS).astype(int))

# Burst: start index must leave room for BURST_LEN consecutive points.
burst_starts = np.unique(np.linspace(1, N_TEST - BURST_LEN - 1, N_REPS).astype(int))


# ── Intensity sweep ───────────────────────────────────────────────────────────
# results[anom_type][sigma_mult] = metrics dict
results: dict[str, dict[float, dict]] = {t: {} for t in ["spike", "burst", "dip"]}

for sigma_mult in INTENSITIES:
    magnitude = sigma_mult * SIGMA

    for anom_type in ["spike", "burst", "dip"]:
        total_tp = total_fp = total_fn = 0
        inj_scores: list[float] = []
        events_detected = 0   # burst: ≥1 point of 4 flagged

        positions = burst_starts if anom_type == "burst" else spike_positions

        for pos in positions:
            # Fresh copy of y_test per repetition
            y_mod = y_test.values.astype(float).copy()

            if anom_type == "spike":
                y_mod[pos] += magnitude
                inj_idx = [pos]
            elif anom_type == "burst":
                for k in range(BURST_LEN):
                    y_mod[pos + k] += magnitude
                inj_idx = list(range(pos, pos + BURST_LEN))
            else:  # dip
                y_mod[pos] -= magnitude
                inj_idx = [pos]

            res_mod   = compute_residuals(model, X_test, pd.Series(y_mod))
            scores    = detector.score(res_mod)
            flags_mod = detector.predict(res_mod)

            inj_mask = np.zeros(N_TEST, dtype=bool)
            for i in inj_idx:
                inj_mask[i] = True

            total_tp += int((flags_mod &  inj_mask).sum())
            total_fp += int((flags_mod & ~inj_mask).sum())
            total_fn += int((~flags_mod & inj_mask).sum())
            inj_scores.extend(float(scores[i]) for i in inj_idx)

            if anom_type == "burst" and any(flags_mod[i] for i in inj_idx):
                events_detected += 1

        n_reps_used = len(positions)
        denom_p = total_tp + total_fp
        denom_r = total_tp + total_fn

        precision  = total_tp / denom_p if denom_p > 0 else float("nan")
        recall     = total_tp / denom_r if denom_r > 0 else 0.0
        f1         = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        mean_score = float(np.mean(inj_scores)) if inj_scores else 0.0

        cell: dict = {
            "precision":  round(precision,  3),
            "recall":     round(recall,     3),
            "f1":         round(f1,         3),
            "mean_score": round(mean_score, 3),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        }
        if anom_type == "burst":
            cell["event_recall"] = round(events_detected / n_reps_used, 3)

        results[anom_type][sigma_mult] = cell


# ── Summary tables ────────────────────────────────────────────────────────────
print(SEP)
print("RESULTS — Intensity sweep  (σ = residual std, N_REPS=15 per cell)")
print(SEP)
print(f"  FP on clean test : {fp_clean}/{N_TEST}  ({fp_rate*100:.1f}%)\n")

for anom_type in ["spike", "burst", "dip"]:
    extra = "  EventRecall" if anom_type == "burst" else ""
    hdr = f"  {'Intensity':>9}  {'Mag (µS)':>8}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'MeanScore':>9}{extra}"
    print(f"  {anom_type.upper()}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for sm in INTENSITIES:
        c   = results[anom_type][sm]
        mag = sm * SIGMA
        prec_str = f"{c['precision']:6.3f}" if not (isinstance(c['precision'], float) and np.isnan(c['precision'])) else "   n/a"
        row = (f"  {sm:>8.1f}σ  {mag:>8.1f}  "
               f"{prec_str}  {c['recall']:6.3f}  "
               f"{c['f1']:6.3f}  {c['mean_score']:9.3f}")
        if anom_type == "burst":
            row += f"  {c['event_recall']:11.3f}"
        print(row)
    print()


# ── Tipping points ────────────────────────────────────────────────────────────
print(SEP)
print(f"TIPPING POINTS — first intensity where point-level recall ≥ {RECALL_FLOOR:.0%}")
print(SEP)
for anom_type in ["spike", "burst", "dip"]:
    tip = next(
        (sm for sm in INTENSITIES if results[anom_type][sm]["recall"] >= RECALL_FLOOR),
        None,
    )
    if tip is not None:
        c = results[anom_type][tip]
        print(f"  {anom_type:<6}:  {tip}σ = {tip*SIGMA:.1f} µS/cm  "
              f"→ recall={c['recall']:.3f}  F1={c['f1']:.3f}")
    else:
        max_r = max(results[anom_type][s]["recall"] for s in INTENSITIES)
        print(f"  {anom_type:<6}:  not reached at 3.0σ  (max recall = {max_r:.3f})")
print()


# ── Figure — recall vs intensity ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x_vals = INTENSITIES
for anom_type, label in [("spike", "Spike"), ("burst", "Burst"), ("dip", "Dip")]:
    y_vals = [results[anom_type][s]["recall"] for s in x_vals]
    ax.plot(
        x_vals, y_vals,
        color=COLORS[anom_type],
        linewidth=2,
        marker="o",
        markersize=8,
        label=label,
        zorder=3,
    )

# Recall target reference line
ax.axhline(
    RECALL_FLOOR,
    color="#898781", linewidth=1, linestyle="--",
    label=f"Target recall = {RECALL_FLOOR:.0%}",
    zorder=2,
)

# Recessive grid
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax.grid(axis="y", color="#e1e0d9", linewidth=0.6, zorder=1)
ax.grid(axis="y", which="minor", color="#e1e0d9", linewidth=0.3, zorder=1)
ax.set_axisbelow(True)

ax.set_xlim(0.25, 3.25)
ax.set_ylim(-0.05, 1.08)
ax.set_xticks(x_vals)
ax.set_xticklabels([f"{s}σ" for s in x_vals], color="#52514e", fontsize=9)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([f"{v:.0%}" for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]],
                   color="#52514e", fontsize=9)
ax.tick_params(axis="both", which="both", length=0)

for spine in ax.spines.values():
    spine.set_color("#c3c2b7")
    spine.set_linewidth(0.8)

ax.set_xlabel(
    f"Injection intensity  (σ = residual std = {SIGMA:.1f} µS/cm)",
    color="#52514e", fontsize=9,
)
ax.set_ylabel("Point-level recall", color="#52514e", fontsize=9)
ax.set_title(
    "EC anomaly detector — recall vs injection intensity",
    color="#0b0b0b", fontsize=11, fontweight="bold", pad=10,
)

legend = ax.legend(
    loc="lower right",
    frameon=True, framealpha=0.95,
    facecolor="#fcfcfb", edgecolor="#c3c2b7",
    fontsize=9,
)
for text in legend.get_texts():
    text.set_color("#52514e")

fig.tight_layout(pad=1.2)

out_path = REPORT / "ec_anomaly_intensity_sweep.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Figure saved → {out_path.relative_to(ROOT)}")
print(SEP)
