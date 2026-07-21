"""
generate_figures_forecast.py — Multi-step recursive forecast figures.

Generates 2 figures (PNG 300 dpi + PDF) in reports/figures/:

  fig_13_recursive_forecast_example.png/pdf
      One recursive forecast trajectory (3 steps, 72 h horizon) overlaid on
      14 days of historical EC context. Shows widening uncertainty bands and
      comparison with true values.

  fig_14_error_growth_by_step.png/pdf
      RMSE (µS/cm) at each forecast step (step 1 = 24 h, step 3 = 72 h),
      quantifying how quickly recursive error accumulates. Reference line =
      one-step static model RMSE from the main EC pipeline evaluation.

Style: identical to previous figure scripts (same palette, 300 dpi, rcParams).

Run from repo root:
    python reports/generate_figures_forecast.py
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
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ── Run validation (reuses all computation) ───────────────────────────────────
# Import the validation module, which executes on import and exposes its
# results as module-level variables.
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import src.forecasting.validate_ec_forecast as val

RMSE_BY_STEP   = val.RMSE_BY_STEP          # [rmse_step1, rmse_step2, rmse_step3]
STD_BY_STEP    = val.STD_BY_STEP
FORECASTER     = val.FORECASTER
DF_RAW         = val.DF_RAW
N_STEPS        = val.N_STEPS_RESULT        # 3
RAW_TEST_START = val.RAW_TEST_START        # 310
Y_TEST         = val.Y_TEST
FREQ           = val.FREQ

print(f"  RMSE by step: {[f'{r:.2f}' for r in RMSE_BY_STEP]} µS/cm")

# ── Paths & palette ───────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG      = "#fcfcfb"
INK     = "#0b0b0b"
SEC     = "#52514e"
MUT     = "#898781"
GRD     = "#e1e0d9"
AX_C    = "#c3c2b7"

C_BLUE   = "#2a78d6"
C_AQUA   = "#1baf7a"
C_AMBER  = "#eda100"
C_RED    = "#e34948"
C_ORANGE = "#eb6834"

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


# ─────────────────────────────────────────────────────────────────────────────
# Fig 13 — Recursive forecast trajectory example
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating fig_13_recursive_forecast_example…")

EXAMPLE_I   = 10           # test-set starting index (date: 2025-12-17)
HIST_DAYS   = 14           # days of historical context to show
raw_end     = RAW_TEST_START + EXAMPLE_I

# Historical context window for display
hist_slice  = DF_RAW.iloc[raw_end - HIST_DAYS: raw_end].copy()
hist_dates  = pd.to_datetime(hist_slice["Date"])
hist_ec     = hist_slice["EC"].values

# Run forecast
forecast_result = FORECASTER.forecast(DF_RAW.iloc[:raw_end].copy(), N_STEPS)
preds  = np.array(forecast_result.predictions)
stds   = np.array([s if s is not None else 0.0 for s in forecast_result.uncertainty_std])

# True values at forecast steps
true_vals = Y_TEST.values[EXAMPLE_I: EXAMPLE_I + N_STEPS]
freq_h    = FREQ.total_seconds() / 3600.0

# Forecast dates (step 1 = day after last history day)
last_hist_date  = pd.to_datetime(DF_RAW["Date"].iloc[raw_end - 1])
forecast_dates  = [last_hist_date + FREQ * (k + 1) for k in range(N_STEPS)]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
fig.patch.set_facecolor(BG)

# Historical line
ax.plot(hist_dates, hist_ec, color=MUT, linewidth=1.5, zorder=2, label="Historical EC")
ax.plot([hist_dates.iloc[-1], forecast_dates[0]],
        [hist_ec[-1], preds[0]],
        color=MUT, linewidth=1.5, zorder=2, linestyle="--", alpha=0.5)

# Vertical separator
ax.axvline(last_hist_date + pd.Timedelta(hours=12),
           color=AX_C, linewidth=0.9, linestyle=":", zorder=1)
ax.text(last_hist_date + pd.Timedelta(hours=14), ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1200,
        "forecast →", fontsize=7, color=MUT, va="top")

# Uncertainty bands at each forecast step
for k, (fd, pred, std_k) in enumerate(zip(forecast_dates, preds, stds)):
    # ±1σ band (lighter) and ±2σ band (very light)
    ax.vlines(fd, pred - 2 * std_k, pred + 2 * std_k,
              color=C_BLUE, linewidth=8, alpha=0.12, zorder=2)
    ax.vlines(fd, pred - std_k, pred + std_k,
              color=C_BLUE, linewidth=8, alpha=0.22, zorder=2)

# Forecast line
ax.plot(forecast_dates, preds,
        color=C_BLUE, linewidth=2.0, marker="o", markersize=7,
        zorder=4, label=f"Recursive forecast (±1σ, ±2σ)")

# True values
ax.plot(forecast_dates, true_vals,
        color=C_AMBER, linewidth=0, marker="D", markersize=7,
        zorder=5, label="True values (test set)")

# Annotate each step
step_labels = ["+24 h", "+48 h", "+72 h"]
for k, (fd, pred, true_v) in enumerate(zip(forecast_dates, preds, true_vals)):
    err = true_v - pred
    sign = "+" if err >= 0 else ""
    ax.annotate(
        f"{step_labels[k]}\nΔ={sign}{err:.0f}",
        xy=(fd, pred), xytext=(2, 12 if k % 2 == 0 else -28),
        textcoords="offset points", fontsize=7, color=C_BLUE,
        ha="center",
    )

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
fig.autofmt_xdate(rotation=0, ha="center")

ax.set_xlabel("Date", color=SEC)
ax.set_ylabel("EC (µS/cm)", color=SEC)
ax.set_title(
    "Recursive Forecast Example — EC at C-1 Station",
    loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=8,
)
ax.text(
    0.0, 1.01,
    f"Starting point: {last_hist_date.date()}  ·  "
    f"72 h horizon ({N_STEPS} steps, daily data)  ·  "
    f"Uncertainty bands = empirical σ of recursive residuals",
    transform=ax.transAxes, fontsize=7.5, color=MUT, va="bottom",
)

ticks_off(ax)
ax.legend(loc="upper left", framealpha=0.92)
fig.tight_layout()

# Annotate vertical separator after layout (reliable y-range)
y_top = ax.get_ylim()[1]
ax.text(last_hist_date + pd.Timedelta(hours=14), y_top * 0.99,
        "forecast →", fontsize=7, color=MUT, va="top")

save(fig, "fig_13_recursive_forecast_example")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 14 — Error growth by forecast step
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating fig_14_error_growth_by_step…")

STATIC_RMSE_1STEP = 53.48   # one-step RMSE of the frozen EC model (full test set)

steps       = list(range(1, N_STEPS + 1))
step_labels = [f"Step {k}\n(+{int(k * freq_h):.0f} h)" for k in steps]
n_starting  = len(val.FORECAST_RESIDUALS_BY_STEP[0])

# Bootstrap 95% CI on RMSE via percentile bootstrap (200 draws)
rng = np.random.default_rng(42)
rmse_ci_lo, rmse_ci_hi = [], []
for k in range(N_STEPS):
    residuals = np.array(val.FORECAST_RESIDUALS_BY_STEP[k])
    boot_rmses = []
    for _ in range(500):
        sample = rng.choice(residuals, size=len(residuals), replace=True)
        boot_rmses.append(np.sqrt(np.mean(sample ** 2)))
    rmse_ci_lo.append(np.percentile(boot_rmses, 2.5))
    rmse_ci_hi.append(np.percentile(boot_rmses, 97.5))

fig, ax = plt.subplots(figsize=(5.5, 4.0))
fig.patch.set_facecolor(BG)

xs = np.array(steps)
bar_w = 0.55

# Bar colors: same blue, slightly darker with each step to hint at accumulation
bar_colors = ["#2a78d6", "#2065b8", "#17529b"]

bars = ax.bar(xs, RMSE_BY_STEP, width=bar_w,
              color=bar_colors[:N_STEPS], zorder=3, linewidth=0)

# 95% CI error bars
ci_lo_err = [r - lo for r, lo in zip(RMSE_BY_STEP, rmse_ci_lo)]
ci_hi_err = [hi - r for r, hi in zip(RMSE_BY_STEP, rmse_ci_hi)]
ax.errorbar(xs, RMSE_BY_STEP,
            yerr=[ci_lo_err, ci_hi_err],
            fmt="none", color=INK, linewidth=1.2, capsize=4, zorder=4)

# Reference: 1-step static model RMSE
ax.axhline(STATIC_RMSE_1STEP, color=C_AMBER, linewidth=1.2,
           linestyle="--", zorder=2, label=f"Static 1-step model RMSE ({STATIC_RMSE_1STEP:.1f} µS/cm)")

# Value labels on bars
for bar, rmse in zip(bars, RMSE_BY_STEP):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{rmse:.1f}", ha="center", va="bottom",
            fontsize=8.5, color=INK, fontweight="bold")

# Growth annotation
rmse_growth = (RMSE_BY_STEP[-1] - RMSE_BY_STEP[0]) / RMSE_BY_STEP[0] * 100
ax.text(N_STEPS, RMSE_BY_STEP[-1] + 4.5,
        f"+{rmse_growth:.1f}% step 1→{N_STEPS}",
        ha="center", fontsize=7.5, color=SEC)

ax.set_xticks(xs)
ax.set_xticklabels(step_labels)
ax.set_ylabel("RMSE (µS/cm)", color=SEC)
ax.set_ylim(0, max(RMSE_BY_STEP) * 1.22)
ticks_off(ax)

ax.set_title(
    "Recursive Forecast Error by Step — EC Pipeline",
    loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=8,
)
ax.text(
    0.0, 1.01,
    f"n = {n_starting} starting points (test set)  ·  "
    f"error bars = 95% bootstrap CI  ·  "
    f"daily data, {N_STEPS}-step horizon ({int(N_STEPS * freq_h)} h)",
    transform=ax.transAxes, fontsize=7.5, color=MUT, va="bottom",
)

ax.legend(loc="lower right", framealpha=0.92)
fig.tight_layout()
save(fig, "fig_14_error_growth_by_step")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("2 figures saved.")
print(f"  RMSE step 1 (24h) : {RMSE_BY_STEP[0]:.2f} µS/cm")
print(f"  RMSE step 2 (48h) : {RMSE_BY_STEP[1]:.2f} µS/cm")
print(f"  RMSE step 3 (72h) : {RMSE_BY_STEP[2]:.2f} µS/cm")
print(f"  Growth step1→3    : +{rmse_growth:.1f}%  — gentle / near-flat")
