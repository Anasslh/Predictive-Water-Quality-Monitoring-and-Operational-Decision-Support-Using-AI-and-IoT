"""
generate_figures_anomaly.py — Anomaly detection figures for the EC pipeline.

Generates 2 figures (PNG 300 dpi + PDF) in reports/figures/:

  fig_09_anomaly_detection_example.png/pdf
      Visual example of a detected burst anomaly (35-day test window,
      injection at 2.5σ_resid, style after A-12/Guo et al.).

  fig_10_score_ceiling_mechanism.png/pdf
      Score vs residual magnitude (-8σ to +8σ): documents the asymmetric
      ceiling discovered during the dip-vs-spike investigation.

Style: identical to generate_figures_ec.py (same palette, 300 dpi, rcParams).
No model retraining: XGBoost loaded from models_store/ec_xgboost_v1_final.json;
ECAnomalyDetector re-fitted on train+val residuals (deterministic, <1 s).

Run from repo root:
    python reports/generate_figures_anomaly.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from src.models.ec.xgboost_model import ECModelXGBoost
from src.anomaly.residual import compute_residuals
from src.anomaly.detector import ECAnomalyDetector

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette (identical to generate_figures_ec.py) ─────────────────────────────
BG      = "#fcfcfb"
INK     = "#0b0b0b"
SEC     = "#52514e"
MUT     = "#898781"
GRD     = "#e1e0d9"
AX_C    = "#c3c2b7"

C_BLUE   = "#2a78d6"   # slot 1 — prediction / positive residuals
C_AQUA   = "#1baf7a"   # slot 2
C_AMBER  = "#eda100"   # slot 3
C_VIOLET = "#4a3aa7"   # slot 5 — anomaly score
C_RED    = "#e34948"   # slot 6 — negative residuals / degraded
C_ORANGE = "#eb6834"   # slot 8

DPI = 300

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.titlesize":     10.5,
    "axes.titleweight":   "bold",
    "axes.titlepad":      8,
    "axes.labelsize":     9,
    "axes.labelcolor":    SEC,
    "axes.edgecolor":     AX_C,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     BG,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         GRD,
    "grid.linewidth":     0.5,
    "xtick.color":        SEC,
    "ytick.color":        SEC,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   AX_C,
    "legend.facecolor":   BG,
    "figure.facecolor":   BG,
    "text.color":         INK,
    "savefig.facecolor":  BG,
})


def ticks_off(ax):
    ax.tick_params(which="both", length=0)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {stem}.png / .pdf")


# ── Load data & fit model + detector ─────────────────────────────────────────
X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()
X_test  = pd.read_csv(PROC / "ec_X_test.csv")
y_test  = pd.read_csv(PROC / "ec_y_test.csv").squeeze()

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

df_raw     = (pd.read_csv(PROC / "c1_clean.csv", parse_dates=["Date"])
              .sort_values("Date").reset_index(drop=True))
test_dates = df_raw["Date"].iloc[310:365].reset_index(drop=True)

model = ECModelXGBoost()
model.model.load_model(str(ROOT / "models_store" / "ec_xgboost_v1_final.json"))
model._is_fitted = True

res_tv   = compute_residuals(model, X_tv, y_tv)
SIGMA    = float(res_tv.std())          # 89.35 µS/cm — residual reference σ
detector = ECAnomalyDetector(contamination=0.05, threshold=0.5, random_state=42)
detector.fit(res_tv)

y_pred_test   = model.predict(X_test)
res_clean     = compute_residuals(model, X_test, y_test)
scores_clean  = detector.score(res_clean)

print("Generating anomaly detection figures →", FIG_DIR)
print(f"  σ_resid = {SIGMA:.2f} µS/cm\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 09 — Visual example of a detected burst anomaly
#
# Window : test rows 10–44 (35 days, 2025-12-18 → 2026-01-21)
# Burst  : rows 22–25 (+4 consecutive), magnitude = 2.5 σ_resid = 223 µS/cm
#          → dates 2025-12-30 → 2026-01-02
# All 4 burst points detected (scores: 0.780 / 0.522 / 0.948 / 0.723)
#
# Form: change-over-time, 2 panels (shared x-axis).
#   Top   : actual EC (injected, black) vs predicted (blue dashed).
#   Bottom: anomaly score (violet), threshold 0.5 (gray dashed).
#   Both  : gray band over the injection window.
# ─────────────────────────────────────────────────────────────────────────────
WIN_START   = 10
WIN_END     = 45          # exclusive → 35-day window
BURST_START = 22
BURST_LEN   = 4
MAGNITUDE   = 2.5 * SIGMA

# Inject burst into y_actual copy only (X features unchanged)
y_injected = y_test.values.astype(float).copy()
for k in range(BURST_LEN):
    y_injected[BURST_START + k] += MAGNITUDE

res_injected   = compute_residuals(model, X_test, pd.Series(y_injected))
scores_injected = detector.score(res_injected)

# Window slices
w_dates   = test_dates.iloc[WIN_START:WIN_END]
w_actual  = y_injected[WIN_START:WIN_END]
w_pred    = y_pred_test[WIN_START:WIN_END]
w_scores  = scores_injected[WIN_START:WIN_END]

burst_t0  = test_dates.iloc[BURST_START] - pd.Timedelta(hours=12)
burst_t1  = test_dates.iloc[BURST_START + BURST_LEN - 1] + pd.Timedelta(hours=12)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6.5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.5]},
)
fig.suptitle(
    "EC anomaly detection — burst injection example (2.5 σ_resid = 223 µS/cm, 4 consecutive days)\n"
    "Residual-based Isolation Forest  |  All 4 burst points detected  |  0 false positives on clean rows",
    fontsize=10.5, fontweight="bold", color=INK,
)

# Injection window shading (both panels)
for ax in (ax1, ax2):
    ax.axvspan(burst_t0, burst_t1, color=MUT, alpha=0.15, zorder=0,
               label="Injected anomaly (2025-12-30 → 2026-01-02)")

# Panel 1: EC series
ax1.plot(w_dates, y_test.values[WIN_START:WIN_END],
         color=AX_C, linewidth=1.2, linestyle="--", zorder=2, label="EC actual (clean baseline)")
ax1.plot(w_dates, w_actual,
         color=INK, linewidth=1.6, zorder=3, label="EC actual (with injected burst)")
ax1.plot(w_dates, w_pred,
         color=C_BLUE, linewidth=1.6, linestyle="--", zorder=4, alpha=0.85,
         label="EC predicted (XGBoost — unchanged by injection)")
ax1.set_ylabel("EC (µS/cm)")

# Annotate the burst peak
burst_peak_k = np.argmax(w_actual[BURST_START - WIN_START: BURST_START - WIN_START + BURST_LEN])
burst_peak_idx = BURST_START - WIN_START + burst_peak_k
ax1.annotate(
    f"+{MAGNITUDE:.0f} µS/cm\n(+2.5 σ_resid)",
    xy=(w_dates.iloc[burst_peak_idx], w_actual[burst_peak_idx]),
    xytext=(w_dates.iloc[burst_peak_idx] + pd.Timedelta(days=4),
            w_actual[burst_peak_idx] + 15),
    fontsize=7.5, color=SEC,
    arrowprops=dict(arrowstyle="->", color=MUT, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, edgecolor=AX_C, alpha=0.9),
)
ax1.legend(loc="upper left", fontsize=7.5)
ticks_off(ax1)

# Panel 2: anomaly score
ax2.plot(w_dates, w_scores,
         color=C_VIOLET, linewidth=1.8, zorder=3, label="Anomaly score")
ax2.scatter(w_dates, w_scores, color=C_VIOLET, s=18, zorder=4)  # daily markers
ax2.axhline(detector.threshold, color=MUT, linewidth=1.2, linestyle="--", zorder=5,
            label=f"Detection threshold = {detector.threshold}")

# Mark detected points
burst_local = slice(BURST_START - WIN_START, BURST_START - WIN_START + BURST_LEN)
det_dates   = w_dates.iloc[burst_local][scores_injected[BURST_START:BURST_START + BURST_LEN] >= detector.threshold]
det_scores  = scores_injected[BURST_START:BURST_START + BURST_LEN][
              scores_injected[BURST_START:BURST_START + BURST_LEN] >= detector.threshold]
ax2.scatter(det_dates, det_scores, color=C_RED, s=60, zorder=6,
            label=f"Detected points ({len(det_scores)}/4)", marker="D", linewidths=0)

ax2.set_ylim(-0.05, 1.12)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_ylabel("Anomaly score")
ax2.legend(loc="upper left", fontsize=7.5)
ticks_off(ax2)

# Clean score annotation
clean_max = scores_clean[WIN_START:WIN_END].max()
ax2.text(0.995, 0.10,
         f"Max score on clean rows (window) = {clean_max:.3f}",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUT)

ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax2.set_xlabel("Date (test set, 35-day window)")

fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.4)
save(fig, "fig_09_anomaly_detection_example")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Asymmetric score ceiling: score vs residual magnitude
#
# Shows the full score response curve from -8σ to +8σ_resid.
# Key findings:
#   - Positive side: score reaches 1.000 at +3.35σ (= +299 µS/cm, training max)
#   - Negative side: score plateaus at 0.878 from -2.50σ onwards — permanent ceiling
#     caused by a single training point (most anomalous training residual was
#     positive, raw IF score 0.8075, which calibrated the MinMaxScaler maximum).
#     Verified by fine sweep (0.01σ steps): -2.49σ→0.8596, -2.50σ→0.8782 (onset).
#   - Detection threshold 0.5 crossed at: ~+2.35σ (positive), ~-1.78σ (negative)
#     → negative side (dip) reaches detection at LOWER magnitude than positive (spike)
#
# Form: score vs ordered magnitude → line chart, diverging color (blue/red).
#   Positive curve = slot 1 (blue), negative curve = slot 6 (red) — diverging pair.
# ─────────────────────────────────────────────────────────────────────────────

# Compute score curve over fine grid (vectorised — one detector.score() call)
mults        = np.linspace(-8.0, 8.0, 800)
resid_grid   = mults * SIGMA
scores_grid  = detector.score(resid_grid)   # shape (800,)

pos_mask = mults >= 0
neg_mask = mults <= 0

# Find exact threshold crossings (interpolation)
def threshold_crossing(mults_side, scores_side, threshold=0.5):
    """Return the σ multiple where the score first crosses `threshold`."""
    above = scores_side >= threshold
    if not above.any():
        return None
    idx = np.argmax(above)
    if idx == 0:
        return float(mults_side[0])
    # Linear interpolation
    m0, m1 = mults_side[idx - 1], mults_side[idx]
    s0, s1 = scores_side[idx - 1], scores_side[idx]
    return float(m0 + (threshold - s0) / (s1 - s0) * (m1 - m0))

pos_cross = threshold_crossing(mults[pos_mask],  scores_grid[pos_mask])
neg_cross = threshold_crossing(-mults[neg_mask][::-1], scores_grid[neg_mask][::-1])

CEILING_NEG       = 0.8782   # empirical plateau value, confirmed to 4 decimal places
CEILING_POS       = 1.0000
TRAIN_MAX_SIGMA   = 3.35     # +299 µS/cm = training max residual → normalization anchor
PLATEAU_NEG_SIGMA = 2.50     # onset of negative plateau: verified by 0.01σ sweep
                             # (-2.49σ→0.8596, -2.50σ→0.8782 first plateau point)

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.suptitle(
    "EC anomaly detector — score response vs residual magnitude\n"
    "Asymmetric ceiling: positive residuals reach score = 1.000, "
    "negative residuals plateau at 0.878",
    fontsize=10.5, fontweight="bold", color=INK,
)

# Positive side (blue)
ax.plot(mults[pos_mask], scores_grid[pos_mask],
        color=C_BLUE, linewidth=2.2, zorder=4, label="Positive residual (spike / burst)")

# Negative side (red) — plot against positive x for readability, mirror via |mult|
ax.plot(-mults[neg_mask], scores_grid[neg_mask],
        color=C_RED, linewidth=2.2, zorder=4, label="Negative residual (dip)")

# Detection threshold
ax.axhline(detector.threshold, color=MUT, linewidth=1.2, linestyle="--", zorder=5,
           label=f"Detection threshold = {detector.threshold}")

# Ceiling lines
ax.axhline(CEILING_NEG, color=C_RED, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)
ax.axhline(CEILING_POS, color=C_BLUE, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)

# Saturation / ceiling annotations
ax.annotate(
    f"Positive ceiling = 1.000\n"
    f"(MinMaxScaler upper bound,\n"
    f"set by training max residual\n"
    f"at +{TRAIN_MAX_SIGMA:.2f}σ = +{TRAIN_MAX_SIGMA*SIGMA:.0f} µS/cm\n"
    f"raw IF score = 0.8075)",
    xy=(TRAIN_MAX_SIGMA, 1.0),
    xytext=(4.8, 0.91),
    fontsize=7.2, color=C_BLUE,
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=C_BLUE, alpha=0.85, linewidth=0.8),
)

ax.annotate(
    f"Negative ceiling = {CEILING_NEG}\n"
    f"(permanent plateau from −{PLATEAU_NEG_SIGMA:.2f}σ;\n"
    f"any larger dip stays at 0.878\n"
    f"→ score alone cannot rank\n"
    f"  dip severity above this point)",
    xy=(PLATEAU_NEG_SIGMA, CEILING_NEG),
    xytext=(4.2, 0.70),
    fontsize=7.2, color=C_RED,
    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=C_RED, alpha=0.85, linewidth=0.8),
)

# Threshold crossing markers
if pos_cross:
    ax.axvline(pos_cross, color=C_BLUE, linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
    ax.text(pos_cross + 0.1, 0.03, f"+{pos_cross:.2f}σ", fontsize=7.2, color=C_BLUE)

if neg_cross:
    ax.axvline(neg_cross, color=C_RED, linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
    ax.text(neg_cross + 0.1, 0.03, f"−{neg_cross:.2f}σ", fontsize=7.2, color=C_RED)

ax.text(0.01, 0.97,
        f"Dip crosses threshold at lower magnitude ({neg_cross:.2f}σ) than spike ({pos_cross:.2f}σ)\n"
        f"→ explains why dip recall > spike recall at all intensities in the sweep",
        transform=ax.transAxes, va="top", fontsize=7.5, color=SEC,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=AX_C, alpha=0.9))

ax.set_xlabel(f"|Residual magnitude|  (σ_resid = {SIGMA:.1f} µS/cm per unit)")
ax.set_ylabel("Normalised anomaly score  [0, 1]")
ax.set_xlim(-0.1, 8.2)
ax.set_ylim(-0.04, 1.15)
ax.set_xticks([0, 1, 1.5, 2, 2.5, 3, 3.35, 4, 5, 6, 7, 8])
ax.set_xticklabels(["0", "1σ", "1.5σ", "2σ", "2.5σ", "3σ", "3.35σ", "4σ", "5σ", "6σ", "7σ", "8σ"],
                   fontsize=7.5)
ax.set_yticks([0, 0.25, 0.5, 0.878, 1.0])
ax.set_yticklabels(["0", "0.25", "0.50", "0.878", "1.00"], fontsize=7.5)

ax.legend(loc="center right", fontsize=8)
ax.grid(axis="y"); ax.grid(axis="x", visible=False)
ticks_off(ax)

fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_10_score_ceiling_mechanism")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("2 figures saved in:", FIG_DIR)
print("  9. reports/figures/fig_09_anomaly_detection_example.png  +  .pdf")
print(" 10. reports/figures/fig_10_score_ceiling_mechanism.png    +  .pdf")
