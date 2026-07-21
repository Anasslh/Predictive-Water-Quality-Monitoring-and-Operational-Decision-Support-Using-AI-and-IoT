"""
compute_wqi.py — Weighted Arithmetic Water Quality Index (WAWQI) for C-1 dataset.

CONTEXT
-------
The WQI column in the original C-1 dataset (Ramgarh station, Jharkhand mining zone,
India) is unreliable: values range 10–1417.7 despite a claimed 0–100 bound, the
computation method is undocumented, and the Class labels (Safe/Moderate/Poor/
Hazardous) do not correspond to any recognised standard. This script replaces it
with a transparent, citable WAWQI computed from first principles.

METHOD — Weighted Arithmetic WQI (WAWQI)
-----------------------------------------
Source: Brown, R.M., McClelland, N.I., Deininger, R.A., & Tozer, R.G. (1972).
        A water quality index — do we dare? Water & Sewage Works, 119(10), 339–343.
        (Extended formulation: Tiwari, T.N. & Misra, M. (1985). A preliminary
        assignment of water quality index to major Indian rivers. Indian Journal of
        Environmental Protection, 5(4), 276–279.)

Formula:
  (1) Unit weight     wi  = K / Si
                      K   = 1 / Σ_j(1/S_j)  [ensures Σ wi = 1]

  (2) Sub-index       Qi  = 100 × |Vi − Vi_ideal| / |Si − Vi_ideal|
      pH              : Vi_ideal = 7.0  (neutral, per WHO/WAWQI convention)
                        Si = 8.5
                        → Qi_pH = 100 × |pH − 7| / |8.5 − 7| = 100 × |pH − 7| / 1.5
      EC              : Vi_ideal = 0   (no conductivity = purest water)
                        Si = 1500 µS/cm
                        → Qi_EC = 100 × EC / 1500
      Turbidity       : Vi_ideal = 0   (no turbidity = purest water)
                        Si = 5 NTU
                        → Qi_Turb = 100 × Turbidity / 5

  (3) WQI = Σ_i (wi × Qi)   [= Σ wi × Qi since Σ wi = 1]

STANDARDS (Si) AND SOURCES
----------------------------
  pH = 8.5
    WHO: World Health Organization (2022). Guidelines for Drinking-Water Quality,
         4th ed., 2nd addendum. Geneva: WHO Press. ISBN 978-92-4-004506-4.
         Acceptable range 6.5–8.5 (no health-based guideline exists; 8.5 is the
         upper aesthetic limit). Ideal value 7.0 (neutral) per WAWQI convention.
    BIS: Bureau of Indian Standards (2012). IS 10500:2012 — Drinking Water.
         New Delhi: BIS. Desirable limit pH 6.5–8.5.

  EC (Electrical Conductivity) = 1500 µS/cm
    BIS: IS 10500:2012, Table 1 — EC permissible limit 1500 µS/cm
         (desirable: 750 µS/cm; permissible in absence of alternate source: 1500).
    WHO: No health-based guideline exists (WHO 2022 §12.1). WHO notes aesthetic
         palatability issues beyond ~2500 µS/cm.
    LIMITATION: At Ramgarh station (mining zone), EC normally ranges 416–1200 µS/cm.
         All measured values remain below the 1500 µS/cm standard, so Qi_EC < 100
         for all 365 rows. Combined with the 1/Si weighting rule, EC receives a
         weight of only 0.21% in this WAWQI — a known limitation of the method when
         one parameter has a large numerical standard. Interpretation must therefore
         rely on the EC sub-index (Qi_EC) and on model predictions directly, not
         only on WQI. This is consistent with how other mining-zone WQI studies
         handle EC (see Kumar et al., 2021; Yadav et al., 2020).

  Turbidity = 5 NTU
    BIS: IS 10500:2012, Table 1 — permissible limit 5 NTU
         (desirable: 1 NTU; permissible in absence of alternate source: 5 NTU).
    WHO: WHO (2022) guideline < 1 NTU for disinfection efficacy; 5 NTU is the
         BIS permissible limit and is the most commonly used Si value in Indian
         WQI studies (Ramakrishnaiah et al., 2009; Yadav et al., 2020).
    NOTE: At Ramgarh station, 276/365 rows exceed 5 NTU (mean = 44 NTU). This
         makes turbidity the primary WQI driver at this site — a physically
         coherent result for a mining-zone water body with suspended mine tailings.

CLASSIFICATION (Brown et al., 1972; widely adopted in Indian WQI literature):
  WQI ≤ 25              : Excellent
  25 < WQI ≤ 50         : Good
  50 < WQI ≤ 75         : Poor
  75 < WQI ≤ 100        : Very Poor
  WQI > 100             : Unsuitable for drinking

OUTPUT
------
  data/processed/c1_with_wqi.csv — original columns + WQI_new + Class_new +
                                   audit columns (Qi_pH, Qi_EC, Qi_Turb, W_pH, W_EC, W_Turb)
  reports/wqi_comparison.png     — time-series comparison old vs new WQI

Run from repo root:
    python src/data/compute_wqi.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[2]
PROC   = ROOT / "data" / "processed"
REPORT = ROOT / "reports"
REPORT.mkdir(exist_ok=True)

SEP = "=" * 72


# ── Standards (Si) and ideal values ──────────────────────────────────────────
# All values and sources documented in module docstring above.
STANDARDS = {
    "pH":        {"Si": 8.5,    "Vi_ideal": 7.0,  "unit": "—"},
    "EC":        {"Si": 1500.0, "Vi_ideal": 0.0,  "unit": "µS/cm"},
    "Turbidity": {"Si": 5.0,    "Vi_ideal": 0.0,  "unit": "NTU"},
}

PARAMS = ["pH", "EC", "Turbidity"]   # ordered for iteration

# ── Classification thresholds (Brown et al., 1972) ───────────────────────────
WQI_CLASSES = [
    (25,   "Excellent"),
    (50,   "Good"),
    (75,   "Poor"),
    (100,  "Very Poor"),
    (float("inf"), "Unsuitable"),
]


# ── Helper functions ──────────────────────────────────────────────────────────
def unit_weights(standards: dict) -> dict[str, float]:
    """
    Compute normalised unit weights wi = K/Si where K = 1/Σ(1/Si).
    Guarantees Σ wi = 1.
    Source: Brown et al. (1972), Tiwari & Misra (1985).
    """
    inv_sum = sum(1.0 / s["Si"] for s in standards.values())
    K = 1.0 / inv_sum
    return {p: K / standards[p]["Si"] for p in standards}


def sub_index(param: str, value: float, standards: dict) -> float:
    """
    Qi = 100 × |Vi − Vi_ideal| / |Si − Vi_ideal|.
    Qi can exceed 100 when the measured value exceeds the standard.
    """
    Si      = standards[param]["Si"]
    Vi_ideal = standards[param]["Vi_ideal"]
    return 100.0 * abs(value - Vi_ideal) / abs(Si - Vi_ideal)


def classify(wqi: float) -> str:
    for threshold, label in WQI_CLASSES:
        if wqi <= threshold:
            return label
    return "Unsuitable"


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(PROC / "c1_clean.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(SEP)
print("WAWQI COMPUTATION — C-1 dataset  (Brown et al., 1972)")
print(SEP)
print(f"  Rows : {len(df)}  |  Period : {df.Date.iloc[0].date()} → {df.Date.iloc[-1].date()}")
print()


# ── Compute unit weights ──────────────────────────────────────────────────────
W = unit_weights(STANDARDS)

print("WEIGHTS (wi = K / Si, K = 1 / Σ(1/Si))")
print(f"  {'Parameter':<12}  {'Si':>8}  {'1/Si':>10}  {'wi':>10}  {'wi%':>8}")
print("  " + "─" * 53)
inv_sum = sum(1.0 / STANDARDS[p]["Si"] for p in PARAMS)
for p in PARAMS:
    si = STANDARDS[p]["Si"]
    print(f"  {p:<12}  {si:>8.1f}  {1/si:>10.6f}  {W[p]:>10.6f}  {W[p]*100:>7.3f}%")
print(f"  {'TOTAL':<12}  {'—':>8}  {inv_sum:>10.6f}  {sum(W.values()):>10.6f}  {sum(W.values())*100:>7.2f}%")
print()

print("  ⚠  EC weight = 0.21%: the 1/Si weighting penalises parameters with")
print("     large numerical standards. EC is the primary concern at this mining")
print("     site but contributes minimally to WQI. Report Qi_EC separately.")
print()


# ── Compute sub-indices and WQI ───────────────────────────────────────────────
for p in PARAMS:
    df[f"Qi_{p}"] = df[p].apply(lambda v: sub_index(p, v, STANDARDS))

df["WQI_new"]   = sum(W[p] * df[f"Qi_{p}"] for p in PARAMS)
df["Class_new"] = df["WQI_new"].apply(classify)


# ── Print descriptive statistics ──────────────────────────────────────────────
print("SUB-INDICES  Qi (min / mean / max)")
print(f"  {'Parameter':<12}  {'Si':>6}  {'Qi min':>8}  {'Qi mean':>8}  {'Qi max':>8}  {'n(Qi>100)':>10}")
print("  " + "─" * 58)
for p in PARAMS:
    col = f"Qi_{p}"
    n_exceed = int((df[col] > 100).sum())
    print(f"  {p:<12}  {STANDARDS[p]['Si']:>6.1f}  "
          f"{df[col].min():>8.2f}  {df[col].mean():>8.2f}  "
          f"{df[col].max():>8.2f}  {n_exceed:>10}")
print()

print("NEW WQI — descriptive statistics:")
print(f"  min    : {df.WQI_new.min():.2f}")
print(f"  mean   : {df.WQI_new.mean():.2f}")
print(f"  median : {df.WQI_new.median():.2f}")
print(f"  max    : {df.WQI_new.max():.2f}")
print()

print("CLASS DISTRIBUTION (new WQI — Brown et al., 1972):")
class_order = ["Excellent", "Good", "Poor", "Very Poor", "Unsuitable"]
class_counts = df["Class_new"].value_counts()
for cls in class_order:
    n = class_counts.get(cls, 0)
    print(f"  {cls:<12}  {n:>4} / 365  ({n/365*100:>5.1f}%)")
print()

print("ORIGINAL WQI (dataset, undocumented method):")
print(f"  min    : {df.WQI.min():.2f}")
print(f"  mean   : {df.WQI.mean():.2f}")
print(f"  max    : {df.WQI.max():.2f}")
print(f"  range  : {df.WQI.max()-df.WQI.min():.2f}  (non-bounded, contradicts claimed 0–100)")
print()

# Correlation between old and new
corr = df["WQI_new"].corr(df["WQI"])
print(f"  Pearson correlation new vs old : r = {corr:.4f}")
print()


# ── Save output CSV ───────────────────────────────────────────────────────────
# Columns: Date, pH, EC, Turbidity, WQI (original), Class (original),
#          WQI_new, Class_new, Qi_pH, Qi_EC, Qi_Turbidity, W_pH, W_EC, W_Turbidity
for p in PARAMS:
    df[f"W_{p}"] = W[p]   # constant columns for audit trail

out_cols = (
    ["Date", "pH", "EC", "Turbidity", "WQI", "Class"]
    + ["WQI_new", "Class_new"]
    + [f"Qi_{p}" for p in PARAMS]
    + [f"W_{p}"  for p in PARAMS]
)
out_path = PROC / "c1_with_wqi.csv"
df[out_cols].round(4).to_csv(out_path, index=False)
print(f"Output saved → {out_path.relative_to(ROOT)}")
print(f"  Columns: {out_cols}")
print()


# ── Comparison figure ─────────────────────────────────────────────────────────
# Three-panel figure:
#   Top    : time series new WQI, colour-coded by class band
#   Middle : time series old WQI (original dataset, undocumented)
#   Bottom : EC and Turbidity sub-indices (Qi) over time — the audit trail

CLASS_COLORS = {
    "Excellent":  "#1baf7a",   # slot 2 — aqua/green
    "Good":       "#2a78d6",   # slot 1 — blue
    "Poor":       "#eda100",   # slot 3 — amber
    "Very Poor":  "#eb6834",   # slot 8 — orange
    "Unsuitable": "#e34948",   # slot 6 — red
}
BG  = "#fcfcfb"
AX  = "#e1e0d9"
INK = "#0b0b0b"
SEC = "#52514e"

dates = df["Date"].values

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig.patch.set_facecolor(BG)

# ── Panel 1 : New WQI time series, coloured segments by class ────────────────
ax1 = axes[0]
ax1.set_facecolor(BG)

# Background class bands
class_thresholds = [(0, 25, "Excellent"), (25, 50, "Good"),
                    (50, 75, "Poor"), (75, 100, "Very Poor"), (100, 4500, "Unsuitable")]
for lo, hi, cls in class_thresholds:
    ax1.axhspan(lo, hi, color=CLASS_COLORS[cls], alpha=0.08, zorder=0)

ax1.plot(dates, df["WQI_new"], color="#0b0b0b", linewidth=0.9, zorder=3, alpha=0.85)

# Horizontal class boundary lines
for boundary in [25, 50, 75, 100]:
    ax1.axhline(boundary, color="#c3c2b7", linewidth=0.7, linestyle="--", zorder=2)

ax1.set_ylabel("WQI new  (WAWQI)", color=SEC, fontsize=9)
ax1.set_title("New WQI  (WAWQI — Brown et al., 1972 / BIS IS 10500:2012 standards)",
              color=INK, fontsize=10, fontweight="bold", loc="left", pad=6)

# Legend: class bands
legend_patches = [Patch(facecolor=CLASS_COLORS[c], alpha=0.5, label=f"{c}")
                  for c in class_order if c in CLASS_COLORS]
ax1.legend(handles=legend_patches, loc="upper right", fontsize=7.5, ncol=3,
           framealpha=0.9, facecolor=BG, edgecolor="#c3c2b7")

ax1_max = df["WQI_new"].max() * 1.05
ax1.set_ylim(0, ax1_max)


# ── Panel 2 : Original WQI ───────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.plot(dates, df["WQI"], color="#e34948", linewidth=0.9, zorder=3, alpha=0.85,
         label="WQI original (undocumented method)")
ax2.set_ylabel("WQI original  (dataset)", color=SEC, fontsize=9)
ax2.set_title("Original WQI  (undocumented — range 10–1417.7, claimed bound 0–100 violated)",
              color="#e34948", fontsize=10, fontweight="bold", loc="left", pad=6)
ax2.set_ylim(0, df["WQI"].max() * 1.05)

# ── Panel 3 : Sub-indices Qi_Turbidity and Qi_EC (the two non-trivial drivers)
ax3 = axes[2]
ax3.set_facecolor(BG)
ax3.plot(dates, df["Qi_Turbidity"], color="#eda100", linewidth=1.0, zorder=3,
         label=f"Qi_Turbidity  (Si = 5 NTU,  W = {W['Turbidity']*100:.1f}%)")
ax3.plot(dates, df["Qi_pH"],        color="#2a78d6", linewidth=1.0, zorder=3, alpha=0.7,
         label=f"Qi_pH          (Si = 8.5,      W = {W['pH']*100:.1f}%)")
ax3.plot(dates, df["Qi_EC"],        color="#4a3aa7", linewidth=1.0, zorder=3, alpha=0.7,
         label=f"Qi_EC          (Si = 1500 µS, W = {W['EC']*100:.2f}%)")
ax3.axhline(100, color="#c3c2b7", linewidth=0.8, linestyle="--", zorder=2,
            label="Qi = 100 (at standard limit)")
ax3.set_ylabel("Sub-index Qi", color=SEC, fontsize=9)
ax3.set_title("Sub-indices  Qi  (audit trail — per-parameter quality index)",
              color=INK, fontsize=10, fontweight="bold", loc="left", pad=6)
ax3.legend(loc="upper right", fontsize=7.5, framealpha=0.9,
           facecolor=BG, edgecolor="#c3c2b7")
ax3.set_ylim(bottom=0)

# ── Shared styling ────────────────────────────────────────────────────────────
for ax in axes:
    ax.grid(axis="y", color=AX, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7"); sp.set_linewidth(0.8)
    ax.tick_params(color="#c3c2b7", labelcolor=SEC, labelsize=8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

axes[2].set_xlabel("Date", color=SEC, fontsize=9)
import matplotlib.dates as mdates
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))

fig.suptitle(
    "Ramgarh Mining Station (C-1) — Water Quality Index comparison",
    color=INK, fontsize=11, fontweight="bold", y=0.995,
)
fig.tight_layout(pad=1.5, h_pad=1.2)

fig_path = REPORT / "wqi_comparison.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Figure saved → {fig_path.relative_to(ROOT)}")


# ── Methodology report ────────────────────────────────────────────────────────
report_lines = [
    "WAWQI METHODOLOGY REPORT — C-1 Ramgarh Mining Station",
    "=" * 60,
    "",
    "Method: Weighted Arithmetic Water Quality Index (WAWQI)",
    "Reference: Brown et al. (1972); Tiwari & Misra (1985)",
    "",
    "FORMULA",
    "  wi  = K / Si     (K = 1 / Σ(1/Sj), ensures Σwi = 1)",
    "  Qi  = 100 × |Vi − Vi_ideal| / |Si − Vi_ideal|",
    "  WQI = Σ(wi × Qi)",
    "",
    "PARAMETERS, STANDARDS, WEIGHTS",
    f"  {'Param':<12} {'Si':>8} {'Vi_ideal':>9} {'Wi':>10} {'Wi%':>8}  Source",
]
for p in PARAMS:
    si    = STANDARDS[p]["Si"]
    vi0   = STANDARDS[p]["Vi_ideal"]
    src   = {"pH": "WHO 2022; BIS IS 10500:2012",
             "EC": "BIS IS 10500:2012 (permissible)",
             "Turbidity": "BIS IS 10500:2012 (permissible)"}[p]
    report_lines.append(
        f"  {p:<12} {si:>8.1f} {vi0:>9.1f} {W[p]:>10.6f} {W[p]*100:>7.3f}%  {src}"
    )

report_lines += [
    "",
    "SUB-INDEX STATISTICS",
    f"  {'Param':<12} {'min':>8} {'mean':>8} {'max':>8} {'n(>100)':>9}",
]
for p in PARAMS:
    col = f"Qi_{p}"
    n_ex = int((df[col] > 100).sum())
    report_lines.append(
        f"  {p:<12} {df[col].min():>8.2f} {df[col].mean():>8.2f} {df[col].max():>8.2f} {n_ex:>9}"
    )

report_lines += [
    "",
    "WQI STATISTICS (new vs original)",
    f"  New  : min={df.WQI_new.min():.2f}  mean={df.WQI_new.mean():.2f}"
    f"  median={df.WQI_new.median():.2f}  max={df.WQI_new.max():.2f}",
    f"  Orig : min={df.WQI.min():.2f}  mean={df.WQI.mean():.2f}"
    f"  max={df.WQI.max():.2f}  (undocumented method)",
    f"  Pearson r(new, orig) = {corr:.4f}",
    "",
    "CLASSIFICATION (Brown et al., 1972)",
    "  0–25: Excellent | 25–50: Good | 50–75: Poor | 75–100: Very Poor | >100: Unsuitable",
]
for cls in class_order:
    n = class_counts.get(cls, 0)
    report_lines.append(f"  {cls:<12} : {n:>4} / 365  ({n/365*100:.1f}%)")

report_lines += [
    "",
    "KEY LIMITATIONS",
    "  1. EC weight = 0.21%: the 1/Si rule systematically under-weights parameters",
    "     with large numerical standards. EC sub-index (Qi_EC) must be reported",
    "     separately for the mining-context interpretation.",
    "  2. Turbidity dominates WQI (W=62.8%): physically coherent for a mining-zone",
    "     water body with suspended tailings (mean=44 NTU, max=170 NTU).",
    "  3. pH shows acidic episodes (min=5.14): both low and high pH contribute",
    "     symmetrically to Qi_pH via the |pH−7| formulation.",
    "",
    "Generated by src/data/compute_wqi.py",
]

rpt_path = REPORT / "wqi_methodology.txt"
rpt_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Methodology report → {rpt_path.relative_to(ROOT)}")
print(SEP)
