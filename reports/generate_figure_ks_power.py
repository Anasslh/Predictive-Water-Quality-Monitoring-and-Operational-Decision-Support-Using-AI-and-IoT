"""
generate_figure_ks_power.py — Statistical power: KS test vs RMSE ratio criterion.

Generates 1 figure (PNG 300 dpi + PDF) in reports/figures/:

  fig_12_ks_power_analysis.png/pdf
      Detection rate (%) vs scale magnitude for two drift criteria:
        - KS two-sample test  (p < 0.05)
        - RMSE ratio          (recent/ref > 1.10)
      1 000 bootstrap repetitions per scale point.
      Demonstrates why RMSE ratio was chosen as the primary drift gate:
      KS power stays flat at ~5–10% even at ×1.20 degradation, while
      RMSE ratio reaches 100% at ×1.15.

Style: identical to generate_figures_ec.py and generate_figures_anomaly.py.

Run from repo root:
    python reports/generate_figure_ks_power.py
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
from scipy.stats import ks_2samp

from src.models.ec.xgboost_model import ECModelXGBoost
from src.data.feature_engineering import build_features
from src.data.split import chronological_split

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette (identical to other figure scripts) ───────────────────────────────
BG      = "#fcfcfb"
INK     = "#0b0b0b"
SEC     = "#52514e"
MUT     = "#898781"
GRD     = "#e1e0d9"
AX_C    = "#c3c2b7"

C_BLUE   = "#2a78d6"   # RMSE ratio criterion
C_RED    = "#e34948"   # KS test criterion

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


# ── Simulation parameters ─────────────────────────────────────────────────────
N_REPS           = 1_000   # bootstrap repetitions per scale point
N_RECENT         = 55      # matches the actual test set size (C-1, 15% split)
RMSE_THRESHOLD   = 1.10    # RMSE ratio gate used in drift_detector.py
KS_ALPHA         = 0.05    # significance level used in drift_detector.py
SCALES           = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]

# ── Load EC model and compute reference residuals (train + val) ───────────────
print("Loading EC model…")
X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()

model = ECModelXGBoost()
model.model.load_model(str(ROOT / "models_store" / "ec_xgboost_v1_final.json"))
model._is_fitted = True

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

ref_preds     = model.predict(X_tv)
ref_residuals = y_tv.values - ref_preds           # shape (n_ref,)
ref_rmse      = float(np.sqrt(np.mean(ref_residuals ** 2)))
n_ref         = len(ref_residuals)
sigma_ref     = float(np.std(ref_residuals))

print(f"  Reference residuals: n={n_ref}, "
      f"RMSE={ref_rmse:.2f} µS/cm, σ={sigma_ref:.2f} µS/cm")

# ── Bootstrap power analysis ──────────────────────────────────────────────────
print(f"\nPower analysis: {N_REPS} repetitions × {len(SCALES)} scales × "
      f"n_recent={N_RECENT}…")

rng         = np.random.default_rng(42)
all_indices = rng.integers(0, n_ref, size=(N_REPS, N_RECENT))
# all_indices[i, :] = row indices into ref_residuals for the i-th repetition

ks_rates   = []
rmse_rates = []

for scale in SCALES:
    # Draw scaled bootstrap samples: shape (N_REPS, N_RECENT)
    samples = ref_residuals[all_indices] * scale

    # RMSE ratio — fully vectorized
    recent_rmses = np.sqrt(np.mean(samples ** 2, axis=1))   # (N_REPS,)
    rmse_fire    = recent_rmses / ref_rmse > RMSE_THRESHOLD
    rmse_rates.append(100.0 * np.mean(rmse_fire))

    # KS test — scalar per repetition, unavoidable loop
    ks_fire = np.array([
        ks_2samp(ref_residuals, samples[i])[1] < KS_ALPHA
        for i in range(N_REPS)
    ])
    ks_rates.append(100.0 * np.mean(ks_fire))

    print(f"  ×{scale:.2f}  →  KS={ks_rates[-1]:5.1f}%   RMSE ratio={rmse_rates[-1]:5.1f}%")

ks_rates   = np.array(ks_rates)
rmse_rates = np.array(rmse_rates)

# ── Figure ────────────────────────────────────────────────────────────────────
print("\nGenerating fig_12_ks_power_analysis…")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
fig.patch.set_facecolor(BG)

xs = np.array(SCALES)

# --- Reference lines (drawn first, behind data) ---
ax.axhline(5,  color=MUT, linewidth=1.0, linestyle="--", zorder=1)
ax.axhline(80, color=MUT, linewidth=1.0, linestyle="--", zorder=1)
ax.axvline(RMSE_THRESHOLD, color=AX_C, linewidth=1.0, linestyle=":", zorder=1)

# --- Data curves ---
ax.plot(xs, ks_rates,
        color=C_RED,  linewidth=2.0, marker="o", markersize=6,
        zorder=3, label="KS two-sample test  (p < 0.05)")
ax.plot(xs, rmse_rates,
        color=C_BLUE, linewidth=2.0, marker="s", markersize=6,
        zorder=3, label="RMSE ratio  (recent / ref > 1.10)")

# --- Annotations for reference lines ---
ax.text(1.505, 5,
        "α = 5%  (chance level)", va="center", ha="left",
        fontsize=7.5, color=MUT)
ax.text(1.505, 80,
        "80% power  (conventional minimum)", va="center", ha="left",
        fontsize=7.5, color=MUT)

# --- RMSE threshold vertical label ---
ax.text(RMSE_THRESHOLD - 0.005, 104,
        "RMSE\nthreshold\n×1.10",
        va="top", ha="right",
        fontsize=7, color=AX_C, linespacing=1.35)

# --- Annotate KS flat zone ---
ks_at_120  = ks_rates[SCALES.index(1.20)]
ks_at_150  = ks_rates[SCALES.index(1.50)]
ax.annotate(
    f"KS: {ks_at_120:.0f}% at ×1.20\n(near-random)",
    xy=(1.20, ks_at_120),
    xytext=(1.215, 22),
    fontsize=7.5, color=C_RED,
    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.9),
    ha="left",
)

# --- Annotate RMSE first crossing 80% power ---
rmse_at_120 = rmse_rates[SCALES.index(1.20)]
ax.annotate(
    f"RMSE: {rmse_at_120:.0f}% at ×1.20\n(clears 80% power)",
    xy=(1.20, rmse_at_120),
    xytext=(1.215, 68),
    fontsize=7.5, color=C_BLUE,
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.9),
    ha="left",
)

# --- Axes ---
ax.set_xlim(1.025, 1.545)
ax.set_ylim(-5, 112)
ax.set_xticks(xs)
ax.set_xticklabels([f"×{s:.2f}" for s in xs], rotation=0)
ax.set_xlabel("Scale factor applied to residuals (simulated degradation magnitude)", color=SEC)
ax.set_ylabel("Detection rate  (%)", color=SEC)
ax.set_yticks([0, 20, 40, 60, 80, 100])

ticks_off(ax)

# --- Title + subtitle ---
ax.set_title(
    "KS Test vs RMSE Ratio: Detection Power for Scale-Based Degradation",
    loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10,
)
ax.text(
    0.0, 1.01,
    f"n_recent = {N_RECENT}  ·  n_reference = {n_ref}  ·  {N_REPS:,} bootstrap repetitions per point  "
    f"·  ref σ = {sigma_ref:.1f} µS/cm",
    transform=ax.transAxes,
    fontsize=7.5, color=MUT, va="bottom",
)

# --- Legend ---
ax.legend(loc="upper left", framealpha=0.92)

fig.tight_layout()
save(fig, "fig_12_ks_power_analysis")

# ── Summary print ─────────────────────────────────────────────────────────────
print()
print("=" * 62)
print(f"{'Scale':>7}  {'KS detection':>14}  {'RMSE detection':>15}")
print("-" * 62)
for scale, ks, rmse in zip(SCALES, ks_rates, rmse_rates):
    ks_mark   = " ✓" if ks   >= 80 else ("  " if ks   < 10 else "  ")
    rmse_mark = " ✓" if rmse >= 80 else ""
    print(f"  ×{scale:.2f}   {ks:>8.1f}%{ks_mark}      {rmse:>8.1f}%{rmse_mark}")
print("=" * 62)
print(f"  Threshold for RMSE ratio: >{RMSE_THRESHOLD}   (10% degradation)")
print(f"  Threshold for KS         : p < {KS_ALPHA}")
print(f"  Dashed lines: 5% (α level) and 80% (power minimum)")
print()
print("1 figure saved.")
