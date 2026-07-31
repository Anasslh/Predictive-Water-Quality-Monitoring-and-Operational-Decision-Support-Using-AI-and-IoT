"""
theme.py — Centralised design tokens and chart semantics.

Single source of truth for colour, spacing, typography and chart styling so the
whole dashboard reads as one intentionally-designed system. Python code (chart
builders) imports these tokens directly; the CSS layer (assets/theme.css) mirrors
the same values as CSS custom properties.

Design direction: a restrained, dense engineering / operations console.
Neutral canvas, deep-slate structure, muted teal for water emphasis. Colour is
used sparingly and only carries meaning (amber = warning, red = confirmed
critical, green = confirmed normal). No gradients, no neon, no glassmorphism.
"""

from __future__ import annotations

from typing import Final

# ── Palette ─────────────────────────────────────────────────────────────────
# Structure / neutrals
COLOR: Final[dict[str, str]] = {
    "bg":            "#eef0f3",   # app canvas (neutral gray)
    "surface":       "#ffffff",   # card / panel background
    "surface_alt":   "#f7f8fa",   # subtle alternate surface (table stripes)
    "structure":     "#152230",   # deep slate — header / sidebar
    "structure_2":   "#1e2f40",   # slightly lighter slate
    "ink":           "#1b2430",   # primary text
    "ink_2":         "#5b6472",   # secondary text
    "ink_3":         "#8b93a0",   # tertiary / muted text
    "border":        "#dce0e6",   # hairline dividers / card borders
    "border_strong": "#c3c9d2",
    # Water emphasis
    "accent":        "#0e7c86",   # muted teal — primary accent (water)
    "accent_soft":   "#e0f0f1",   # teal tint background
    "accent_2":      "#2f6f9f",   # muted blue — secondary accent
    # Status semantics (used ONLY when the state is confirmed by data)
    "ok":            "#2f7d5d",   # green — confirmed normal
    "ok_soft":       "#e3f1ea",
    "warn":          "#b9770e",   # amber — warning / attention
    "warn_soft":     "#fbf0da",
    "critical":      "#b23b3b",   # red — confirmed critical
    "critical_soft": "#f7e3e3",
    "neutral_soft":  "#eceef1",   # unknown / no-data (calm gray, never alarming)
}

# ── Chart semantics ─────────────────────────────────────────────────────────
# Fixed visual grammar shared by every chart so the same concept always looks
# the same across the whole dashboard.
CHART: Final[dict[str, object]] = {
    "actual_color":     COLOR["ink"],        # actual measurements — solid slate
    "actual_marker":    6,
    "predicted_color":  COLOR["accent"],     # model prediction — teal
    "predicted_dash":   "dash",
    "forecast_color":   COLOR["accent_2"],   # forecast — blue
    "forecast_dash":    "dot",
    "band_color":       "rgba(47,111,159,0.14)",  # forecast uncertainty band
    "anomaly_color":    COLOR["critical"],   # anomaly markers — red diamonds
    "anomaly_symbol":   "diamond",
    "anomaly_size":     11,
    "limit_color":      COLOR["warn"],       # reference limit lines — amber, dashed
    "grid_color":       "#e6e9ee",
    "axis_color":       COLOR["ink_3"],
    "font_family":      "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    "font_size":        12,
    "height_detail":    360,
    "height_compact":   200,
}

# ── Spacing / radius / typography scale (mirrors theme.css) ──────────────────
SPACE: Final[dict[str, str]] = {
    "xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px",
}
RADIUS: Final[str] = "6px"          # restrained rounding, not pill-shaped
FONT_STACK: Final[str] = CHART["font_family"]  # type: ignore[assignment]


# ── Status → colour mapping ─────────────────────────────────────────────────
def status_colors(status: str) -> tuple[str, str]:
    """
    Map a semantic status token to (foreground, soft-background) colours.

    status ∈ {"ok", "warn", "critical", "neutral"}.
    "neutral" is used for unknown / not-yet-available states and is deliberately
    calm (gray), never alarming.
    """
    table = {
        "ok":       (COLOR["ok"], COLOR["ok_soft"]),
        "warn":     (COLOR["warn"], COLOR["warn_soft"]),
        "critical": (COLOR["critical"], COLOR["critical_soft"]),
        "neutral":  (COLOR["ink_2"], COLOR["neutral_soft"]),
    }
    return table.get(status, table["neutral"])


def plotly_layout(height: int, *, rtl: bool = False) -> dict:
    """Return a shared Plotly layout dict enforcing consistent chart styling."""
    return {
        "height": height,
        "margin": {"l": 56, "r": 20, "t": 28, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": CHART["font_family"], "size": CHART["font_size"], "color": COLOR["ink_2"]},
        "xaxis": {
            "gridcolor": CHART["grid_color"],
            "linecolor": COLOR["border_strong"],
            "zeroline": False,
            "ticks": "outside",
            "tickcolor": COLOR["border_strong"],
        },
        "yaxis": {
            "gridcolor": CHART["grid_color"],
            "linecolor": COLOR["border_strong"],
            "zeroline": False,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom", "y": 1.02,
            "xanchor": "right" if not rtl else "left",
            "x": 1 if not rtl else 0,
            "font": {"size": 11},
        },
        "hovermode": "x unified",
    }
