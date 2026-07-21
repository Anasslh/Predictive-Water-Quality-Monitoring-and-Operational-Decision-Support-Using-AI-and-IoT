"""
generate_figures_ec.py — Publication-quality figures for the EC pipeline.

Generates 9 figures (PNG 300 dpi + PDF) in reports/figures/.
No model retraining: XGBoost loaded from models_store/ec_xgboost_v1_final.json.

Run from repo root:
    python reports/generate_figures_ec.py

Figures produced:
    fig_01_eda_timeseries.png/pdf
    fig_02_ec_noise_diagnostic.png/pdf
    fig_03_ec_split_distribution.png/pdf
    fig_04_model_comparison_oob.png/pdf
    fig_05_model_comparison_tuned.png/pdf
    fig_06_feature_exploration.png/pdf
    fig_07_prediction_vs_actual.png/pdf
    fig_08_shap_importance.png/pdf
    fig_11_relative_error_timeline.png/pdf

Color palette: slots 1,2,3,5,6,8 from the reference palette (Brown et al. 1972
CVD-validated, ΔE ≥ 24.2 on light surface). Each model keeps its hue across all
figures (color follows entity, not rank).
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
from scipy.stats import gaussian_kde
import shap

from src.models.ec.xgboost_model import ECModelXGBoost

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference palette (slots 1‑3, 5‑6, 8 — pre-validated, ΔE ≥ 24.2) ────────
# Color follows entity, assigned in fixed slot order, never cycled.
BG      = "#fcfcfb"   # chart surface
INK     = "#0b0b0b"   # primary text
SEC     = "#52514e"   # secondary text / axis labels
MUT     = "#898781"   # muted: axes, gridlines, baseline bars
GRD     = "#e1e0d9"   # gridline hairline
AX_C    = "#c3c2b7"   # spine / baseline

C_BLUE   = "#2a78d6"  # slot 1 — XGBoost | Train  | Pred
C_AQUA   = "#1baf7a"  # slot 2 — RF      | Test   | Better
C_AMBER  = "#eda100"  # slot 3 — Val     | Residuals (positive)
C_VIOLET = "#4a3aa7"  # slot 5 — SVR
C_RED    = "#e34948"  # slot 6 — Worse   | Residuals (negative)
C_ORANGE = "#eb6834"  # slot 8 — Ensemble (marginal/special)

DPI = 300

# ── Global rcParams ───────────────────────────────────────────────────────────
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
    "lines.linewidth":    1.8,
    "savefig.facecolor":  BG,
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def ticks_off(ax):
    ax.tick_params(which="both", length=0)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {stem}.png / .pdf")


def hbar_labels(ax, bars, values, fmt="{:.1f}", pad=0.8):
    """Inline value labels on horizontal bars."""
    for bar, val in zip(bars, values):
        ax.text(
            val + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center", fontsize=7.5, color=SEC,
        )


# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = (
    pd.read_csv(PROC / "c1_clean.csv", parse_dates=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)

X_train = pd.read_csv(PROC / "ec_X_train.csv")
y_train = pd.read_csv(PROC / "ec_y_train.csv").squeeze()
X_val   = pd.read_csv(PROC / "ec_X_val.csv")
y_val   = pd.read_csv(PROC / "ec_y_val.csv").squeeze()
X_test  = pd.read_csv(PROC / "ec_X_test.csv")
y_test  = pd.read_csv(PROC / "ec_y_test.csv").squeeze()

# Test period dates: rows 310–364 in df_raw (after 7-row rolling trim + 250 train + 53 val)
test_dates = df_raw["Date"].iloc[310:365].reset_index(drop=True)

# Split boundary timestamps (for vertical markers)
VAL_START  = pd.Timestamp("2025-10-16")
TEST_START = pd.Timestamp("2025-12-08")
SERIES_END = pd.Timestamp("2026-01-31")

# ── Load pre-trained XGBoost (no retraining) ──────────────────────────────────
model = ECModelXGBoost()
model.model.load_model(str(ROOT / "models_store" / "ec_xgboost_v1_final.json"))
model._is_fitted = True

y_pred   = model.predict(X_test)
residuals = y_test.values - y_pred

print("Generating EC pipeline figures →", FIG_DIR)
print()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 01 — EDA: time series pH, EC, Turbidity (365 days)
# Form: change-over-time → line chart, 3 panels, shared x-axis.
# Color: one hue per parameter (categorical, slots 1-3).
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle(
    "Water quality time series — Ramgarh mining station (C-1 dataset)\n"
    "365 daily observations, 2025-02-01 → 2026-01-31",
    fontsize=11, fontweight="bold", color=INK,
)

panel_cfg = [
    ("pH",        "pH",                    "—",      C_BLUE),
    ("EC",        "Electrical Conductivity","µS/cm",  C_AMBER),
    ("Turbidity", "Turbidity",              "NTU",    C_AQUA),
]

for ax, (col, name, unit, color) in zip(axes, panel_cfg):
    ax.plot(df_raw["Date"], df_raw[col], color=color, linewidth=1.2, zorder=3)
    ax.set_ylabel(f"{name}\n({unit})", fontsize=8.5)
    # Shade val and test periods
    ax.axvspan(VAL_START,  TEST_START, alpha=0.07, color=C_AMBER, zorder=0)
    ax.axvspan(TEST_START, SERIES_END, alpha=0.07, color=C_AQUA,  zorder=0)
    # Split boundary lines
    ax.axvline(VAL_START,  color=AX_C, linewidth=0.8, linestyle="--", zorder=2)
    ax.axvline(TEST_START, color=AX_C, linewidth=0.8, linestyle="--", zorder=2)
    ticks_off(ax)

# Split legend on top panel
axes[0].legend(
    handles=[
        Patch(facecolor=C_BLUE,  alpha=0.6, label="Train  2025-02-08 → 10-15  (n = 250)"),
        Patch(facecolor=C_AMBER, alpha=0.5, label="Val    2025-10-16 → 12-07  (n = 53)"),
        Patch(facecolor=C_AQUA,  alpha=0.5, label="Test   2025-12-08 → 01-31  (n = 55)"),
    ],
    loc="upper right", fontsize=7.5,
)

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ticks_off(axes[-1])

fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.6)
save(fig, "fig_01_eda_timeseries")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 02 — EC noise diagnostic: raw series + rolling 7-day std
# Form: change-over-time, 2 panels. No dual axis (guideline).
# Shows that rolling σ during the high-variability train period reaches
# ~150 µS/cm → denoising would destroy signal, not just noise.
# ─────────────────────────────────────────────────────────────────────────────
roll_mean = df_raw["EC"].rolling(7, min_periods=1).mean()
roll_std  = df_raw["EC"].rolling(7, min_periods=1).std().fillna(0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.suptitle(
    "EC signal noise analysis — motivation for retaining the raw signal\n"
    "Rolling 7-day σ reaches ~150 µS/cm in the train period: denoising would destroy signal",
    fontsize=11, fontweight="bold", color=INK,
)

ax1.plot(df_raw["Date"], df_raw["EC"],  color=C_AMBER, linewidth=1.2, label="EC raw",           zorder=3)
ax1.plot(df_raw["Date"], roll_mean,     color=INK,     linewidth=1.0, label="Rolling 7d mean",  zorder=4,
         linestyle="--", alpha=0.7)
ax1.set_ylabel("EC (µS/cm)")
ax1.legend(loc="upper right")
ax1.axvspan(VAL_START, TEST_START, alpha=0.06, color=C_AMBER, zorder=0)
ax1.axvspan(TEST_START, SERIES_END, alpha=0.06, color=C_AQUA,  zorder=0)
ticks_off(ax1)

ax2.fill_between(df_raw["Date"], 0, roll_std, color=C_AMBER, alpha=0.35, zorder=2)
ax2.plot(df_raw["Date"], roll_std, color=C_AMBER, linewidth=1.0, zorder=3)
ax2.axvspan(VAL_START,  TEST_START, alpha=0.06, color=C_AMBER, zorder=0)
ax2.axvspan(TEST_START, SERIES_END, alpha=0.06, color=C_AQUA,  zorder=0)
ax2.set_ylabel("Rolling 7d σ (µS/cm)")
ax2.set_xlabel("Date")

# Annotate the high-variability train window
peak_idx = roll_std.idxmax()
ax2.annotate(
    f"Peak σ = {roll_std.max():.0f} µS/cm\n(train period)",
    xy=(df_raw["Date"].iloc[peak_idx], roll_std.max()),
    xytext=(df_raw["Date"].iloc[peak_idx] - pd.Timedelta(days=60), roll_std.max() * 0.85),
    fontsize=7.5, color=SEC,
    arrowprops=dict(arrowstyle="->", color=MUT, lw=0.8),
)
ax2.axhline(roll_std[df_raw["Date"] >= TEST_START].mean(), color=AX_C,
            linewidth=0.8, linestyle=":", zorder=2)
ax2.text(TEST_START + pd.Timedelta(days=2),
         roll_std[df_raw["Date"] >= TEST_START].mean() + 2,
         f"test σ̄ = {roll_std[df_raw['Date'] >= TEST_START].mean():.0f}", fontsize=7, color=MUT)
ticks_off(ax2)

ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.6)
save(fig, "fig_02_ec_noise_diagnostic")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 03 — EC distribution by split (level shift diagnostic)
# Form: distribution comparison → KDE curves + fill, one panel.
# Color: Train=blue (slot 1), Val=amber (slot 3), Test=aqua (slot 2).
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle(
    "EC distribution by split — structural level shift diagnostic\n"
    "Train covers flood/drought season (σ = 212 µS/cm); Val and Test are calmer (σ ≈ 50)",
    fontsize=11, fontweight="bold", color=INK,
)

split_cfg = [
    (y_train.values, C_BLUE,  f"Train  n=250  mean={y_train.mean():.0f}  σ={y_train.std():.0f} µS/cm"),
    (y_val.values,   C_AMBER, f"Val    n=53   mean={y_val.mean():.0f}  σ={y_val.std():.0f} µS/cm"),
    (y_test.values,  C_AQUA,  f"Test   n=55   mean={y_test.mean():.0f}  σ={y_test.std():.0f} µS/cm"),
]

x_range = np.linspace(300, 1400, 500)
for vals, color, label in split_cfg:
    kde = gaussian_kde(vals, bw_method="scott")
    ax.fill_between(x_range, kde(x_range), alpha=0.18, color=color, zorder=2)
    ax.plot(x_range, kde(x_range), color=color, linewidth=2.0, label=label, zorder=3)
    ax.axvline(np.mean(vals), color=color, linewidth=1.0, linestyle="--", alpha=0.8, zorder=4)

ax.set_xlabel("EC (µS/cm)")
ax.set_ylabel("Density")
ax.text(0.01, 0.97, "Dashed lines = split means", transform=ax.transAxes,
        color=MUT, fontsize=7.5, va="top")
ax.legend(loc="upper right")
ax.grid(axis="y")
ax.grid(axis="x", visible=False)
ticks_off(ax)

fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_03_ec_split_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 04 — Model comparison: out-of-the-box (validation RMSE)
# Form: ranking comparison → horizontal bar chart (RMSE, lower = better).
# Color: model entity (XGBoost=blue, RF=aqua, SVR=violet, baselines=gray).
# ─────────────────────────────────────────────────────────────────────────────
PERSISTENCE_RMSE = 76.02

oob_models = [
    # (label, rmse_val, color)
    ("Train mean baseline", 98.10, MUT),
    ("Persistence (EC_lag1)", PERSISTENCE_RMSE, "#c3c2b7"),
    ("XGBoost (default, lr=0.1)", 77.32, C_BLUE),
    ("Random Forest (default)", 70.11, C_AQUA),
    ("SVR (default + y-scaling)", 58.15, C_VIOLET),
]
# Sort worst→best so best bar is at bottom (ascending RMSE from bottom)
oob_models.sort(key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    "EC model comparison — out-of-the-box performance (validation RMSE)\n"
    "SVR leads before tuning; XGBoost underperforms at default lr = 0.1",
    fontsize=11, fontweight="bold", color=INK,
)

labels  = [m[0] for m in oob_models]
rmses   = [m[1] for m in oob_models]
colors  = [m[2] for m in oob_models]

bars = ax.barh(labels, rmses, color=colors, height=0.55, zorder=3, linewidth=0)
ax.axvline(PERSISTENCE_RMSE, color=MUT, linewidth=1.2, linestyle="--", zorder=5,
           label=f"Persistence = {PERSISTENCE_RMSE:.1f} µS/cm (naive reference)")
hbar_labels(ax, bars, rmses, pad=0.6)

ax.set_xlabel("RMSE on validation set (µS/cm)")
ax.set_xlim(0, max(rmses) * 1.16)
ax.invert_yaxis()
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ticks_off(ax)

ax.legend(
    handles=[
        Patch(facecolor=MUT,      label="Naive baselines"),
        Patch(facecolor=C_BLUE,   label="XGBoost"),
        Patch(facecolor=C_AQUA,   label="Random Forest"),
        Patch(facecolor=C_VIOLET, label="SVR"),
        plt.Line2D([0], [0], color=MUT, linestyle="--", label=f"Persistence ({PERSISTENCE_RMSE:.1f})"),
    ],
    loc="lower right", fontsize=7.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_04_model_comparison_oob")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 05 — Model comparison: after tuning (validation RMSE)
# Same form and color mapping as Fig 04 — visual continuity.
# ─────────────────────────────────────────────────────────────────────────────
tuned_models = [
    ("Persistence (reference)",  76.02, MUT),
    ("Random Forest (tuned)",    67.30, C_AQUA),
    ("SVR (tuned, C=1 ε=0.5)",  64.80, C_VIOLET),
    ("XGBoost (tuned, lr=0.01)", 60.90, C_BLUE),
]
tuned_models.sort(key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.suptitle(
    "EC model comparison — after hyperparameter tuning (validation RMSE)\n"
    "Tuning reverses the ranking: XGBoost becomes best (lr 0.1 → 0.01, −16.4 RMSE)",
    fontsize=11, fontweight="bold", color=INK,
)

labels_t = [m[0] for m in tuned_models]
rmses_t  = [m[1] for m in tuned_models]
colors_t = [m[2] for m in tuned_models]

bars_t = ax.barh(labels_t, rmses_t, color=colors_t, height=0.55, zorder=3, linewidth=0)
ax.axvline(PERSISTENCE_RMSE, color=MUT, linewidth=1.2, linestyle="--", zorder=5)
hbar_labels(ax, bars_t, rmses_t, pad=0.4)

# Delta annotation vs persistence
best_rmse = min(rmses_t)
ax.text(best_rmse + 0.5, len(tuned_models) - 0.5,
        f"Δ vs persistence: −{PERSISTENCE_RMSE - best_rmse:.1f} µS/cm",
        fontsize=7.5, color=C_BLUE, va="top")

ax.set_xlabel("RMSE on validation set (µS/cm)")
ax.set_xlim(0, max(rmses_t) * 1.18)
ax.invert_yaxis()
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ticks_off(ax)
ax.legend(
    handles=[
        Patch(facecolor=MUT,      label="Baseline (persistence)"),
        Patch(facecolor=C_BLUE,   label="XGBoost  ← best"),
        Patch(facecolor=C_AQUA,   label="Random Forest"),
        Patch(facecolor=C_VIOLET, label="SVR"),
    ],
    loc="lower right", fontsize=7.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_05_model_comparison_tuned")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 06 — Feature exploration: RMSE vs baseline
# Form: deviation from reference → horizontal bar chart with reference line.
# Color encodes result category (not model identity): better/neutral/worse/special.
# ─────────────────────────────────────────────────────────────────────────────
BASELINE = 60.94   # XGBoost tuned, 9-feature set

NEUTRAL_TOL = 1.0   # ± 1 µS/cm = within noise

feat_rows = [
    # (label, rmse_val, category)
    ("Baseline — XGBoost tuned\n9-feature set (reference)",  60.94, "reference"),
    ("Differencing (ΔEC)",                                    78.53, "worse"),
    ("EWMA(3)",                                               65.40, "worse"),
    ("EWMA(3,5) combined",                                    65.87, "worse"),
    ("Long lags (lag7 + lag14)",                              64.22, "worse"),
    ("EWMA(5)",                                               60.95, "neutral"),
    ("Rolling slope (7d)",                                    60.91, "neutral"),
    ("Bayesian tuning — Optuna\n40 trials (depth=2, lr≈0.005)", 60.74, "neutral"),
    ("Ensemble XGB+SVR+RF\n(simple average — discarded)",     59.65, "discarded"),
]
feat_rows.sort(key=lambda x: x[1], reverse=True)   # worst → best (top → bottom)

FEAT_COLORS = {
    "reference":  C_BLUE,
    "worse":      C_RED,
    "neutral":    MUT,
    "discarded":  C_ORANGE,
}

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    "EC feature exploration — validation RMSE vs baseline (9-feature XGBoost)\n"
    "No tested approach yields a significant or generalizable gain",
    fontsize=11, fontweight="bold", color=INK,
)

feat_labels = [r[0] for r in feat_rows]
feat_rmses  = [r[1] for r in feat_rows]
feat_colors = [FEAT_COLORS[r[2]] for r in feat_rows]

bars_f = ax.barh(feat_labels, feat_rmses, color=feat_colors, height=0.6, zorder=3, linewidth=0)
ax.axvline(BASELINE, color=C_BLUE, linewidth=1.5, linestyle="--", zorder=5,
           label=f"Baseline = {BASELINE:.2f} µS/cm")
hbar_labels(ax, bars_f, feat_rmses, pad=0.3)

ax.set_xlabel("RMSE on validation set (µS/cm)")
ax.set_xlim(0, max(feat_rmses) * 1.14)
ax.invert_yaxis()
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ticks_off(ax)

ax.legend(
    handles=[
        Patch(facecolor=C_BLUE,   label=f"Reference baseline ({BASELINE:.2f} µS/cm)"),
        Patch(facecolor=MUT,      label=f"Neutral  (Δ RMSE < {NEUTRAL_TOL:.0f} µS/cm)"),
        Patch(facecolor=C_RED,    label="Worse than baseline"),
        Patch(facecolor=C_ORANGE, label="Marginal gain but discarded\n(XAI complexity, +3 models)"),
    ],
    loc="lower right", fontsize=7.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_06_feature_exploration")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 07 — Prediction vs actual (test set) + residuals
# Form: change-over-time (2 panels, shared x-axis).
# Top: actual (black) vs predicted (blue). Bottom: residuals, diverging blue/red.
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.2]},
)
fig.suptitle(
    "EC prediction vs actual — test set (2025-12-08 → 2026-01-31, n = 55)\n"
    "XGBoost  depth=3, n=100, lr=0.01  |  RMSE = 53.48 µS/cm  |  gain vs persistence = +14.24 µS/cm",
    fontsize=11, fontweight="bold", color=INK,
)

# Panel 1: actual vs predicted
ax1.plot(test_dates, y_test.values, color=INK,    linewidth=1.6, label="Actual EC",             zorder=4)
ax1.plot(test_dates, y_pred,        color=C_BLUE,  linewidth=1.6, label="Predicted EC (XGBoost)", zorder=3,
         linestyle="--", alpha=0.88)
ax1.fill_between(test_dates, y_test.values, y_pred, alpha=0.12, color=C_AMBER, zorder=2,
                 label="Prediction error band")
ax1.set_ylabel("EC (µS/cm)")
ax1.legend(loc="upper left")
ticks_off(ax1)

metrics_txt = (
    "RMSE = 53.48 µS/cm    MAE = 42.42 µS/cm\n"
    "R² = −0.23  (structural artifact — test σ = 48.6 µS/cm ≈ RMSE)\n"
    "Relative error ≈ 6.8% of test range"
)
ax1.text(0.995, 0.04, metrics_txt, transform=ax1.transAxes,
         fontsize=7.2, color=SEC, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=AX_C, alpha=0.92))

# Panel 2: residuals (diverging blue/red)
ax2.fill_between(test_dates, 0, residuals,
                 where=residuals >= 0, interpolate=True, color=C_BLUE, alpha=0.70, zorder=3,
                 label="Positive residual (actual > predicted)")
ax2.fill_between(test_dates, 0, residuals,
                 where=residuals < 0,  interpolate=True, color=C_RED,  alpha=0.70, zorder=3,
                 label="Negative residual (actual < predicted)")
ax2.axhline(0, color=AX_C, linewidth=0.9, zorder=2)
ax2.axhline(np.mean(residuals), color=MUT, linewidth=0.8, linestyle=":", zorder=4)
ax2.set_ylabel("Residual (µS/cm)")
ax2.set_xlabel("Date (test set)")
ax2.legend(loc="upper left", fontsize=7.5)
ax2.text(0.995, 0.96,
         f"bias = {np.mean(residuals):+.1f} µS/cm  |  σ_resid = {np.std(residuals):.1f} µS/cm",
         transform=ax2.transAxes, ha="right", va="top", fontsize=7.5, color=SEC)
ticks_off(ax2)

ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # every Monday
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.4)
save(fig, "fig_07_prediction_vs_actual")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 08 — SHAP feature importance (mean |SHAP|, 5 test rows)
# Form: magnitude ranking → horizontal bar chart.
# Color: sequential blue ramp (one hue, light→dark by importance magnitude).
# ─────────────────────────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model.model)
features  = X_test.columns.tolist()

# Full test set (55 rows) — reference figure
shap_vals_all  = np.array(explainer.shap_values(X_test))        # (55, 9)
mean_abs_all   = np.abs(shap_vals_all).mean(axis=0)

# First 5 rows kept separately for qualitative per-row commentary in the text
shap_vals_5    = shap_vals_all[:5]                               # (5, 9)
mean_abs_5     = np.abs(shap_vals_5).mean(axis=0)

order       = np.argsort(mean_abs_all)   # ascending → most important at top after invert
feat_sorted = [features[i] for i in order]
vals_sorted = mean_abs_all[order]

# Sequential blue ramp: step 250 (#86b6ef) → step 450 (#2a78d6), 9 steps
blue_ramp = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
]
n = len(feat_sorted)
bar_colors_shap = [blue_ramp[int(i * (len(blue_ramp) - 1) / (n - 1))] for i in range(n)]

fig, ax = plt.subplots(figsize=(8.5, 5))
fig.suptitle(
    "EC prediction — SHAP feature importance  (mean |SHAP value|)\n"
    "Averaged over full test set (n = 55)  |  XGBoost xgb_ec_v1_final  |  explainer_type = tree",
    fontsize=11, fontweight="bold", color=INK,
)

bars_s = ax.barh(feat_sorted, vals_sorted, color=bar_colors_shap, height=0.6, zorder=3, linewidth=0)
hbar_labels(ax, bars_s, vals_sorted, fmt="{:.1f}", pad=0.25)

ax.set_xlabel("Mean |SHAP value| (µS/cm)")
ax.set_xlim(0, vals_sorted.max() * 1.18)
ax.invert_yaxis()
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ticks_off(ax)

top_feat_pct = mean_abs_all.max() / mean_abs_all.sum() * 100
ax.text(
    0.995, 0.04,
    f"EC_roll3_mean dominant ({top_feat_pct:.0f}% of total |SHAP|)\n"
    "XGBoost prefers 3-day smoothed lag over raw lag1\n"
    "→ low-cost sensors in a mining area introduce noise",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=SEC,
    bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=AX_C, alpha=0.92),
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_08_shap_importance")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Relative prediction error timeline (test set, 55 days)
#
# Form: change-over-time → line + fill, single panel.
# Motivation: RMSE = 53.48 µS/cm is hard to interpret for non-technical readers.
#   Relative error (%) is scale-invariant and immediately actionable.
#
# Reference lines at 10% and 20% — generic visual guides only, no sourced standard.
#   These values are common round-number breakpoints used informally in applied ML
#   reporting (e.g. "single-digit %", "below 10%", "below 20%"), but no specific
#   normative threshold was verified from a primary source for this application.
#   They are presented purely as orientation markers for the non-technical reader.
# ─────────────────────────────────────────────────────────────────────────────
rel_err = np.abs(y_test.values - y_pred) / y_test.values * 100   # % per day

mean_rel     = float(np.mean(rel_err))
median_rel   = float(np.median(rel_err))
n_above20    = int(np.sum(rel_err > 20.0))
pct_above20  = n_above20 / len(rel_err) * 100

print("  FIG 11 — Relative prediction error (test set, n = 55):")
print(f"    Mean        = {mean_rel:.2f}%")
print(f"    Median      = {median_rel:.2f}%")
print(f"    Days > 20%  = {n_above20}/55  ({pct_above20:.1f}%)")
print()

# Top 3 worst days
top3_idx = np.argsort(rel_err)[-3:][::-1]   # descending by relative error
y_max    = rel_err.max() * 1.14              # headroom for annotations

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle(
    f"EC model — relative prediction error per day (test set, n = 55)\n"
    f"Mean = {mean_rel:.1f}%  ·  Median = {median_rel:.1f}%  ·  "
    f"{n_above20} day{'s' if n_above20 != 1 else ''} above 20%  ·  "
    f"XGBoost xgb_ec_v1_final  (RMSE = 53.48 µS/cm)",
    fontsize=10.5, fontweight="bold", color=INK,
)

# Fill + line
ax.fill_between(test_dates, 0, rel_err, alpha=0.15, color=C_BLUE, zorder=2)
ax.plot(test_dates, rel_err, color=C_BLUE, linewidth=1.6, zorder=3,
        label=r"|actual − predicted| / actual × 100")
ax.scatter(test_dates, rel_err, color=C_BLUE, s=20, zorder=4)

# Mean line
ax.axhline(mean_rel, color=MUT, linewidth=1.4, linestyle="--", zorder=5,
           label=f"Mean = {mean_rel:.1f}%")

# Reference threshold lines (recessive — visual guides only)
for lvl in (10.0, 20.0):
    ax.axhline(lvl, color=AX_C, linewidth=0.9, linestyle=":", zorder=3, alpha=0.85)
    ax.text(test_dates.iloc[-1], lvl + 0.4, f" {lvl:.0f}%",
            fontsize=7, color=MUT, va="bottom")

# Subtle red shading above 20% to highlight the "flag for review" zone
ax.fill_between(
    [test_dates.iloc[0], test_dates.iloc[-1]],
    20.0, y_max,
    alpha=0.04, color=C_RED, zorder=1,
)

# Annotate the top 3 worst days
for rank, idx in enumerate(top3_idx):
    # Push the label right for early dates, left for late dates
    dx = pd.Timedelta(days=7 if idx < 28 else -11)
    dy = 3.5 + rank * 1.5   # stagger labels if peaks are close
    ax.annotate(
        f"{test_dates.iloc[idx].strftime('%d %b')}\n{rel_err[idx]:.1f}%",
        xy=(test_dates.iloc[idx], rel_err[idx]),
        xytext=(test_dates.iloc[idx] + dx, rel_err[idx] + dy),
        fontsize=7.2, color=SEC,
        arrowprops=dict(arrowstyle="->", color=MUT, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=AX_C,
                  linewidth=0.7, alpha=0.92),
    )

ax.set_ylim(0, y_max)
ax.set_ylabel("Relative prediction error (%)")
ax.set_xlabel("Date (test set,  2025-12-08 → 2026-01-31)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(axis="y"); ax.grid(axis="x", visible=False)
ticks_off(ax)

ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # every Monday
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

ax.text(0.004, 0.975,
        "Lines at 10% and 20% are generic visual reference markers — no specific standard verified",
        transform=ax.transAxes, va="top", fontsize=6.8, color=MUT, style="italic")

fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_11_relative_error_timeline")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("9 figures saved in:", FIG_DIR)
for i, stem in enumerate([
    "fig_01_eda_timeseries",
    "fig_02_ec_noise_diagnostic",
    "fig_03_ec_split_distribution",
    "fig_04_model_comparison_oob",
    "fig_05_model_comparison_tuned",
    "fig_06_feature_exploration",
    "fig_07_prediction_vs_actual",
    "fig_08_shap_importance",
    "fig_11_relative_error_timeline",
], 1):
    print(f"  {i}. reports/figures/{stem}.png  +  .pdf")
