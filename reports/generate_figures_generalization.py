"""
generate_figures_generalization.py — Cross-parameter generalization figures.

Generates 5 publication-quality figures (PNG 300 dpi + PDF) in reports/figures/.
No model retraining: results loaded from benchmark CSVs and hardcoded test metrics.

Run from repo root:
    python reports/generate_figures_generalization.py

Figures produced:
    fig_15_skill_vs_persistence.png/pdf
    fig_16_cv_vs_rel_rmse.png/pdf
    fig_17_turbidity_split_distribution.png/pdf
    fig_18_ph_benchmark_ranking.png/pdf
    fig_19_turbidity_raw_vs_log.png/pdf

Test-set metrics (one-shot evaluation, chronological 70/15/15 split, n=55 each):
    EC        — XGBoost  RMSE=53.48 µS/cm   persistence RMSE=67.73   skill=+21.0%
    pH        — XGBoost  RMSE=0.854 pH       persistence RMSE=1.105   skill=+22.7%
    Turbidity — XGBoost  RMSE=36.633 NTU     persistence RMSE=47.219  skill=+22.4%

Signal CV (full dataset, same as used in onboarding for log-transform decision):
    EC=23.4%   pH=11.9%   Turbidity=86.6%

Benchmark cache CSVs (written by one-off benchmark runs, loaded here):
    data/processed/ph_benchmark_results.csv
    data/processed/turbidity_raw_benchmark_results.csv

Color palette: same slots as generate_figures_ec.py (CVD-validated, ΔE ≥ 24.2).
Color follows entity, never rank:
    EC  = C_AMBER (slot 3)   pH  = C_BLUE (slot 1)   Turbidity = C_AQUA (slot 2)
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
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference palette (identical to generate_figures_ec.py) ──────────────────
BG      = "#fcfcfb"
INK     = "#0b0b0b"
SEC     = "#52514e"
MUT     = "#898781"
GRD     = "#e1e0d9"
AX_C    = "#c3c2b7"

C_BLUE   = "#2a78d6"   # pH
C_AQUA   = "#1baf7a"   # Turbidity
C_AMBER  = "#eda100"   # EC
C_VIOLET = "#4a3aa7"   # SVR / log variant
C_RED    = "#e34948"
C_ORANGE = "#eb6834"

DPI = 300

# ── Global rcParams (identical to generate_figures_ec.py) ────────────────────
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


def hbar_labels(ax, bars, values, fmt="{:.1f}", pad=0.5):
    for bar, val in zip(bars, values):
        ax.text(
            val + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center", fontsize=7.5, color=SEC,
        )


# ── Hardcoded test-set metrics (one-shot, never re-run) ──────────────────────
# All values from the final test-set evaluation (n=55, 2025-12-08 → 2026-01-31).
# Chronological 70/15/15 split — no data leakage.
EC_PERSIST_RMSE   = 67.73     # µS/cm
EC_XGB_RMSE       = 53.48     # µS/cm
EC_XGB_REL_RMSE   = 6.79      # % of test mean (787.62 µS/cm)
EC_SKILL          = (EC_PERSIST_RMSE - EC_XGB_RMSE) / EC_PERSIST_RMSE * 100  # 21.0%
EC_CV             = 23.4      # % coefficient of variation on full dataset

PH_PERSIST_RMSE   = 1.105     # pH units
PH_XGB_RMSE       = 0.854     # pH units
PH_XGB_REL_RMSE   = 11.83     # % of test mean (7.221 pH)
PH_SKILL          = (PH_PERSIST_RMSE - PH_XGB_RMSE) / PH_PERSIST_RMSE * 100   # 22.7%
PH_CV             = 11.9      # %

TURB_PERSIST_RMSE = 47.219    # NTU
TURB_XGB_RMSE     = 36.633    # NTU
TURB_XGB_REL_RMSE = 99.89     # % of test mean (36.672 NTU) — seasonal artifact
TURB_SKILL        = (TURB_PERSIST_RMSE - TURB_XGB_RMSE) / TURB_PERSIST_RMSE * 100  # 22.4%
TURB_CV           = 86.6      # %

# Turbidity raw vs log (validation set, benchmark selection phase)
TURB_RAW_BEST_RMSE = 36.23    # NTU  — XGBoost rank 1 (chosen)
TURB_LOG_BEST_RMSE = 39.92    # NTU  — SVR rank 1 in log space, back-transformed to NTU
TURB_LOG_DEGRADATION = (TURB_LOG_BEST_RMSE - TURB_RAW_BEST_RMSE) / TURB_RAW_BEST_RMSE * 100  # +10.2%

print("Generating cross-parameter generalization figures →", FIG_DIR)
print()

# ── Load data for Fig 17 ─────────────────────────────────────────────────────
df_raw = (
    pd.read_csv(PROC / "c1_with_wqi.csv", parse_dates=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)

n = len(df_raw)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
# Rows 0..n_train-1 → train, n_train..n_train+n_val-1 → val, rest → test
# The feature engineering drops 7 rows (rolling window warmup), so actual splits shift
# slightly, but the raw series split is close enough for the distribution diagnostic.
turb_train = df_raw["Turbidity"].iloc[:n_train].values
turb_val   = df_raw["Turbidity"].iloc[n_train : n_train + n_val].values
turb_test  = df_raw["Turbidity"].iloc[n_train + n_val :].values

# ── Load benchmark CSVs for Figs 18–19 ───────────────────────────────────────
ph_bench_path   = PROC / "ph_benchmark_results.csv"
turb_bench_path = PROC / "turbidity_raw_benchmark_results.csv"

if not ph_bench_path.exists():
    raise FileNotFoundError(
        f"{ph_bench_path} not found.\n"
        "Run the pH benchmark once to generate it:\n"
        "  python -c \"import sys; sys.path.insert(0,'.')\n"
        "  from src.pipeline.orchestrator import *; ...\"\n"
        "Or re-run reports/generate_figures_ec.py which caches the data."
    )

ph_bench   = pd.read_csv(ph_bench_path)
turb_bench = pd.read_csv(turb_bench_path)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 15 — Skill score vs persistence — 3-parameter comparison
# Form: magnitude comparison → horizontal bar chart (positive = beats persistence).
# Color: one fixed hue per parameter (entity-consistent with EC figures).
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = [
    # (label, skill_pct, rmse_model, rmse_persist, unit, color)
    ("EC",        EC_SKILL,   EC_XGB_RMSE,   EC_PERSIST_RMSE,   "µS/cm", C_AMBER),
    ("pH",        PH_SKILL,   PH_XGB_RMSE,   PH_PERSIST_RMSE,   "pH",    C_BLUE),
    ("Turbidity", TURB_SKILL, TURB_XGB_RMSE, TURB_PERSIST_RMSE, "NTU",   C_AQUA),
]

fig, ax = plt.subplots(figsize=(9, 4))
fig.suptitle(
    "XGBoost vs persistence (lag-1) — skill score on test set  (n = 55 per parameter)\n"
    "Chronological 70/15/15 split  |  skill = (RMSE_persist − RMSE_model) / RMSE_persist × 100",
    fontsize=10.5, fontweight="bold", color=INK,
)

param_labels = [p[0] for p in PARAMS]
skills       = [p[1] for p in PARAMS]
colors_p     = [p[5] for p in PARAMS]

# Sort ascending so best bar is at bottom
order    = sorted(range(len(skills)), key=lambda i: skills[i])
y_labels = [param_labels[i] for i in order]
y_skills = [skills[i]       for i in order]
y_colors = [colors_p[i]     for i in order]

bars = ax.barh(y_labels, y_skills, color=y_colors, height=0.5, zorder=3, linewidth=0)

# Value labels
for bar, val in zip(bars, y_skills):
    ax.text(
        val + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"+{val:.1f}%",
        va="center", fontsize=8, color=SEC,
    )

# Reference markers for each parameter (RMSE values in annotation box)
for i, idx in enumerate(order):
    p = PARAMS[idx]
    ax.text(
        0.99, (i + 0.5) / len(PARAMS),
        f"RMSE {p[3]:.2f} → {p[2]:.2f} {p[4]}",
        transform=ax.transAxes, ha="right", va="center",
        fontsize=7.2, color=MUT,
    )

ax.axvline(0, color=AX_C, linewidth=0.9, zorder=2)
ax.set_xlabel("Skill score vs persistence (%)")
ax.set_xlim(-2, max(y_skills) * 1.35)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ticks_off(ax)

ax.text(
    0.01, 0.04,
    "Positive skill = model beats the naive lag-1 forecast on ALL three parameters.\n"
    "Skill scores 21–23% despite very different signal dynamics (CV: 11.9%→86.6%).",
    transform=ax.transAxes, va="bottom", fontsize=7.2, color=MUT, style="italic",
)

ax.legend(
    handles=[
        Patch(facecolor=C_AMBER, label=f"EC        | CV={EC_CV}%"),
        Patch(facecolor=C_BLUE,  label=f"pH        | CV={PH_CV}%"),
        Patch(facecolor=C_AQUA,  label=f"Turbidity | CV={TURB_CV}%"),
    ],
    loc="lower right", fontsize=7.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig_15_skill_vs_persistence")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 16 — Signal CV vs relative RMSE (test set)
# Form: scatter, 3 labeled points — reveals the structural relationship between
#   signal volatility and prediction difficulty.
# Color: one hue per parameter (entity-consistent).
# ─────────────────────────────────────────────────────────────────────────────
CV_VALUES      = [EC_CV,           PH_CV,          TURB_CV]
REL_RMSE_VALUES = [EC_XGB_REL_RMSE, PH_XGB_REL_RMSE, TURB_XGB_REL_RMSE]
POINT_COLORS   = [C_AMBER,         C_BLUE,          C_AQUA]
POINT_LABELS   = ["EC\n(µS/cm)",    "pH",            "Turbidity\n(NTU)"]

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(
    "Signal coefficient of variation vs relative RMSE on test set\n"
    "EC C-1 dataset  |  XGBoost (frozen)  |  n = 55 per parameter",
    fontsize=10.5, fontweight="bold", color=INK,
)

for cv, rel, col, label in zip(CV_VALUES, REL_RMSE_VALUES, POINT_COLORS, POINT_LABELS):
    ax.scatter(cv, rel, color=col, s=160, zorder=4, edgecolors=BG, linewidths=1.5)

# Parameter labels with offset to avoid overlap
offsets = {
    "EC\n(µS/cm)":     (-6,   4),
    "pH":              (-4, -10),
    "Turbidity\n(NTU)": (2,   3),
}
for cv, rel, col, label in zip(CV_VALUES, REL_RMSE_VALUES, POINT_COLORS, POINT_LABELS):
    dx, dy = offsets.get(label, (2, 2))
    ax.annotate(
        label,
        xy=(cv, rel),
        xytext=(cv + dx, rel + dy),
        fontsize=8, color=col, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.7),
    )

# Annotate exact values
for cv, rel, col in zip(CV_VALUES, REL_RMSE_VALUES, POINT_COLORS):
    ax.text(cv, rel - 4.5, f"CV={cv}%\nrel RMSE={rel:.1f}%",
            ha="center", va="top", fontsize=7, color=SEC)

# Trend line (visual guide only — 3 points, not a statistical fit)
z = np.polyfit(CV_VALUES, REL_RMSE_VALUES, 1)
x_line = np.linspace(0, 100, 200)
y_line = np.polyval(z, x_line)
ax.plot(x_line, y_line, color=MUT, linewidth=1.0, linestyle="--", zorder=2, alpha=0.6,
        label="Linear trend (3-point — visual guide only)")

ax.set_xlabel("Signal CV (%) — coefficient of variation on full dataset")
ax.set_ylabel("Relative RMSE on test set (%) — RMSE / test mean × 100")
ax.set_xlim(-5, 100)
ax.set_ylim(-5, 115)
ax.legend(loc="upper left", fontsize=7.5)
ticks_off(ax)

ax.text(
    0.98, 0.04,
    "Turbidity rel RMSE ≈ 100%: seasonal artifact — test set (mean 36.7 NTU)\n"
    "vs train (mean 46.0 NTU). Model skill vs persistence = +22.4% → real signal.",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=MUT,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=AX_C, alpha=0.9),
)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig_16_cv_vs_rel_rmse")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 17 — Turbidity distribution by split (level-shift diagnostic)
# Form: distribution comparison → KDE curves + fill, single panel.
# Color: Train=blue, Val=amber, Test=aqua (same as EC split figure).
# Explains why rel RMSE ≈ 100% on test is a seasonal artifact, not failure.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.suptitle(
    "Turbidity distribution by split — seasonal level-shift diagnostic\n"
    "Test period (Jan 2026) is systematically calmer than train/val → rel RMSE artifact",
    fontsize=10.5, fontweight="bold", color=INK,
)

split_cfg = [
    (turb_train, C_BLUE,
     f"Train  n={len(turb_train)}  mean={turb_train.mean():.1f}  "
     f"median={np.median(turb_train):.1f}  σ={turb_train.std():.1f} NTU"),
    (turb_val, C_AMBER,
     f"Val    n={len(turb_val)}    mean={turb_val.mean():.1f}  "
     f"median={np.median(turb_val):.1f}   σ={turb_val.std():.1f} NTU"),
    (turb_test, C_AQUA,
     f"Test   n={len(turb_test)}   mean={turb_test.mean():.1f}  "
     f"median={np.median(turb_test):.1f}  σ={turb_test.std():.1f} NTU"),
]

x_range = np.linspace(0, 250, 600)
for vals, color, label in split_cfg:
    kde = gaussian_kde(vals, bw_method="scott")
    ax.fill_between(x_range, kde(x_range), alpha=0.18, color=color, zorder=2)
    ax.plot(x_range, kde(x_range), color=color, linewidth=2.0, label=label, zorder=3)
    ax.axvline(np.mean(vals), color=color, linewidth=1.2, linestyle="--", alpha=0.85, zorder=4)
    ax.axvline(np.median(vals), color=color, linewidth=0.8, linestyle=":", alpha=0.7, zorder=4)

ax.set_xlabel("Turbidity (NTU)")
ax.set_ylabel("Density")
ax.set_xlim(0, 220)
ax.legend(loc="upper right")
ax.grid(axis="y"); ax.grid(axis="x", visible=False)
ticks_off(ax)

ax.text(0.01, 0.98,
        "Dashed = split mean  ·  Dotted = split median",
        transform=ax.transAxes, color=MUT, fontsize=7.5, va="top")

# Annotate the shift with a bracket-style annotation
ax.annotate(
    "",
    xy=(turb_train.mean(), 0.0095),
    xytext=(turb_test.mean(), 0.0095),
    arrowprops=dict(arrowstyle="<->", color=MUT, lw=1.2),
)
ax.text(
    (turb_train.mean() + turb_test.mean()) / 2, 0.011,
    f"Mean shift\n{turb_train.mean():.1f} → {turb_test.mean():.1f} NTU\n"
    f"(−{turb_train.mean() - turb_test.mean():.1f} NTU, −{(turb_train.mean()-turb_test.mean())/turb_train.mean()*100:.0f}%)",
    ha="center", va="bottom", fontsize=7.5, color=SEC,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=AX_C, alpha=0.9),
)

# Highlight the >100 NTU zone
ax.axvspan(100, 220, alpha=0.05, color=C_RED, zorder=0)
ax.text(130, ax.get_ylim()[1] * 0.85, ">100 NTU\ntrain: 15%\nval:    9%\ntest:   5%",
        fontsize=7.2, color=MUT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=AX_C, alpha=0.85))

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig_17_turbidity_split_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 18 — pH benchmark ranking: top-10 configs by validation RMSE
# Form: magnitude ranking → horizontal bar chart, chosen config highlighted.
# Color: chosen rank highlighted in blue; all others in muted gray.
# ─────────────────────────────────────────────────────────────────────────────
CHOSEN_RANK = 3   # human-confirmed choice

ph_top10 = ph_bench.head(10).copy()

# Parse hyperparams for compact labels
def _compact_label(row):
    alg = row["algorithm"]
    hp  = eval(row["hyperparams"])   # safe: known format from our own code
    if alg == "XGBoost":
        return (f"XGBoost  d={hp.get('max_depth','?')}  "
                f"n={hp.get('n_estimators','?')}  "
                f"lr={hp.get('learning_rate','?')}")
    if alg == "RF":
        return (f"RF  n={hp.get('n_estimators','?')}  "
                f"d={hp.get('max_depth','?')}")
    if alg == "SVR":
        return f"SVR  C={hp.get('C','?')}  ε={hp.get('epsilon','?')}"
    return alg

ph_top10["label"] = ph_top10.apply(_compact_label, axis=1)
ph_top10["color"] = ph_top10["rank"].apply(lambda r: C_BLUE if r == CHOSEN_RANK else MUT)

# Sort worst→best (ascending RMSE from bottom)
ph_top10 = ph_top10.sort_values("val_rmse", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    f"pH benchmark — top-10 configurations by validation RMSE  (n=33 configs total)\n"
    f"Chosen: rank {CHOSEN_RANK}  (depth=2, n=50, lr=0.01)  — conservative depth for generalization",
    fontsize=10.5, fontweight="bold", color=INK,
)

bars = ax.barh(
    ph_top10["label"],
    ph_top10["val_rmse"],
    color=ph_top10["color"],
    height=0.6, zorder=3, linewidth=0,
)

# Value labels
for bar, val in zip(bars, ph_top10["val_rmse"]):
    ax.text(
        val + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", fontsize=7.5, color=SEC,
    )

# Annotate chosen bar
chosen_row = ph_top10[ph_top10["rank"] == CHOSEN_RANK].iloc[0]
chosen_y   = ph_top10[ph_top10["rank"] == CHOSEN_RANK].index[0]
ax.annotate(
    f"Chosen (rank {CHOSEN_RANK})\nConservative depth=2\nbetter generalization",
    xy=(chosen_row["val_rmse"], chosen_y),
    xytext=(chosen_row["val_rmse"] + 0.004, chosen_y + 2.2),
    fontsize=7.5, color=C_BLUE,
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=AX_C, alpha=0.92),
)

ax.set_xlabel("Validation RMSE (pH units)")
ax.set_xlim(0, ph_top10["val_rmse"].max() * 1.16)
ax.invert_yaxis()
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ticks_off(ax)

# Show RMSE range of top-10
best_rmse  = ph_top10["val_rmse"].min()
worst_rmse = ph_top10["val_rmse"].max()
ax.text(0.99, 0.01,
        f"Top-10 RMSE range: {best_rmse:.4f} – {worst_rmse:.4f} pH units\n"
        f"Spread = {(worst_rmse - best_rmse)*1000:.1f} mpH — ranking is tight, all XGBoost top-7",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=MUT)

ax.legend(
    handles=[
        Patch(facecolor=C_BLUE, label=f"Chosen config (rank {CHOSEN_RANK})"),
        Patch(facecolor=MUT,    label="Other top-10 configs"),
    ],
    loc="lower right", fontsize=7.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig_18_ph_benchmark_ranking")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 19 — Turbidity: raw vs log-transform (validation RMSE, best per variant)
# Form: two-bar comparison → vertical bar chart.
# Color: raw (aqua, entity color) vs log (violet, contrast/alternative).
# ─────────────────────────────────────────────────────────────────────────────
RAW_LABEL = f"Raw target\nXGBoost d=2 n=50 lr=0.01\n(rank 1)"
LOG_LABEL = f"Log-transformed\nSVR  C=1  ε=0.1\n(best in log space)"

VARIANT_RMSES  = [TURB_RAW_BEST_RMSE, TURB_LOG_BEST_RMSE]
VARIANT_LABELS = [RAW_LABEL, LOG_LABEL]
VARIANT_COLORS = [C_AQUA, C_VIOLET]

fig, ax = plt.subplots(figsize=(7, 5.5))
fig.suptitle(
    "Turbidity: raw target vs log-transform  —  best validation RMSE per variant\n"
    f"Log degradation: +{TURB_LOG_DEGRADATION:.1f}% RMSE  |  raw (NTU) wins  |  same XGBoost architecture",
    fontsize=10.5, fontweight="bold", color=INK,
)

bars = ax.bar(
    VARIANT_LABELS, VARIANT_RMSES,
    color=VARIANT_COLORS, width=0.45, zorder=3, linewidth=0,
)

# Value labels above bars
for bar, val in zip(bars, VARIANT_RMSES):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.4,
        f"{val:.2f} NTU",
        ha="center", va="bottom", fontsize=9, color=SEC, fontweight="bold",
    )

# Delta annotation between bars
mid_x    = 0.5
y_annot  = (TURB_RAW_BEST_RMSE + TURB_LOG_BEST_RMSE) / 2
ax.annotate(
    "",
    xy=(0.58, TURB_LOG_BEST_RMSE),
    xytext=(0.58, TURB_RAW_BEST_RMSE),
    arrowprops=dict(arrowstyle="<->", color=MUT, lw=1.2),
)
ax.text(
    0.62, y_annot,
    f"+{TURB_LOG_DEGRADATION:.1f}%\n(+{TURB_LOG_BEST_RMSE - TURB_RAW_BEST_RMSE:.2f} NTU)",
    ha="left", va="center", fontsize=8, color=SEC,
)

ax.set_ylabel("Validation RMSE (NTU)")
ax.set_ylim(0, max(VARIANT_RMSES) * 1.25)
ax.grid(axis="y"); ax.grid(axis="x", visible=False)
ticks_off(ax)

ax.text(
    0.98, 0.04,
    f"Log transform evaluated because CV={TURB_CV}% > 60% threshold.\n"
    "Log variant fits on ln(Turbidity), evaluates RMSE in original NTU scale.\n"
    "Raw wins → signal structure (not skewness) drives predictability.",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=MUT,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=AX_C, alpha=0.9),
)

ax.legend(
    handles=[
        Patch(facecolor=C_AQUA,   label="Raw target  ← chosen"),
        Patch(facecolor=C_VIOLET, label="Log-transformed (back-projected to NTU)"),
    ],
    loc="upper left", fontsize=7.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig_19_turbidity_raw_vs_log")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("5 figures saved in:", FIG_DIR)
stems = [
    "fig_15_skill_vs_persistence",
    "fig_16_cv_vs_rel_rmse",
    "fig_17_turbidity_split_distribution",
    "fig_18_ph_benchmark_ranking",
    "fig_19_turbidity_raw_vs_log",
]
for i, stem in enumerate(stems, 15):
    print(f"  {i}. reports/figures/{stem}.png  +  .pdf")
