"""
charts.py — Plotly chart builders with a fixed visual grammar.

The same concept always looks the same across the dashboard:

  • Measured values       → solid slate line + markers
  • Model prediction      → teal dashed line
  • Forecast              → blue dotted line
  • Forecast interval     → subtle blue uncertainty band
  • Anomalies             → red diamond markers
  • Missing values        → gaps (connectgaps=False), never interpolated

These functions return plotly Figures and import no Streamlit, so they can be
tested headless.
"""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.components.format import DASH, fmt_number
from dashboard.config.theme import CHART, COLOR, plotly_layout
from dashboard.i18n.translator import Translator
from dashboard.models.schemas import ParameterData
from dashboard.services.validation import resolve_duplicate_timestamps, sort_by_time
from dashboard.services.wqi import WqiDiagnostic

# Class colours mirror src/data/compute_wqi.py so the diagnostic reads the same
# as the pipeline's own figures.
_WQI_CLASS_COLORS = {
    "Excellent": "#1baf7a",
    "Good": "#2a78d6",
    "Poor": "#eda100",
    "Very Poor": "#eb6834",
    "Unsuitable": "#e34948",
}


def _dated(pdata: ParameterData):
    records = resolve_duplicate_timestamps(pdata.records)
    return [r for r in sort_by_time(records) if r.timestamp is not None]


def actual_vs_predicted(
    pdata: ParameterData,
    tr: Translator,
) -> go.Figure:
    """Measured vs predicted time series with explicit gaps and anomaly markers."""
    fig = go.Figure()
    dated = _dated(pdata)
    unit = pdata.display_unit

    if dated:
        x = [r.timestamp for r in dated]
        customdata = [
            [
                _tooltip_value(r.actual_value, unit),
                _tooltip_value(r.predicted_value, unit),
                _anomaly_label(r.is_anomaly, tr),
            ]
            for r in dated
        ]
        hover = (
            f'<b>{tr.t("timestamp")}</b>: %{{x|%Y-%m-%d %H:%M:%S}}<br>'
            f'<b>{tr.t("measured_series")}</b>: %{{customdata[0]}}<br>'
            f'<b>{tr.t("predicted_series")}</b>: %{{customdata[1]}}<br>'
            f'<b>{tr.t("anomaly_state")}</b>: %{{customdata[2]}}<extra></extra>'
        )

        # Prediction (drawn first, sits under the measured line).
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[r.predicted_value for r in dated],
                customdata=customdata,
                name=tr.t("predicted_series"),
                mode="lines",
                line=dict(color=CHART["predicted_color"], width=2, dash=CHART["predicted_dash"]),
                connectgaps=False,
                hovertemplate=hover,
            )
        )
        # Measured values. Nulls remain in y so Plotly preserves visible gaps.
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[r.actual_value for r in dated],
                customdata=customdata,
                name=tr.t("measured_series"),
                mode="lines+markers",
                line=dict(color=CHART["actual_color"], width=1.8),
                marker=dict(size=5, color=CHART["actual_color"]),
                connectgaps=False,
                hovertemplate=hover,
            )
        )
        # Anomaly markers on the measured value.
        anomalies = [r for r in dated if r.is_anomaly is True and r.actual_value is not None]
        if anomalies:
            fig.add_trace(
                go.Scatter(
                    x=[r.timestamp for r in anomalies],
                    y=[r.actual_value for r in anomalies],
                    customdata=[
                        [
                            _tooltip_value(r.actual_value, unit),
                            _tooltip_value(r.predicted_value, unit),
                            tr.t("anomaly"),
                        ]
                        for r in anomalies
                    ],
                    name=tr.t("anomaly"),
                    mode="markers",
                    marker=dict(
                        symbol=CHART["anomaly_symbol"],
                        size=CHART["anomaly_size"],
                        color=CHART["anomaly_color"],
                        line=dict(width=1, color="#ffffff"),
                    ),
                    hovertemplate=hover,
                )
            )

    layout = plotly_layout(CHART["height_detail"], rtl=tr.is_rtl)
    layout["yaxis"]["title"] = {"text": unit, "font": {"size": 11}}
    fig.update_layout(**layout)
    return fig


def _tooltip_value(value: float | None, unit: str) -> str:
    """Format one tooltip value without ever borrowing from another series."""
    if value is None:
        return DASH
    return f"{fmt_number(value)} {unit}".strip()


def _anomaly_label(value: bool | None, tr: Translator) -> str:
    if value is True:
        return tr.t("anomaly")
    if value is False:
        return tr.t("normal")
    return tr.t("unknown")


def sparkline(pdata: ParameterData) -> go.Figure:
    """Tiny actual-value sparkline for compact cards (no axes, no legend)."""
    dated = [r for r in _dated(pdata) if r.actual_value is not None]
    fig = go.Figure()
    if dated:
        fig.add_trace(
            go.Scatter(
                x=[r.timestamp for r in dated],
                y=[r.actual_value for r in dated],
                mode="lines",
                line=dict(color=CHART["accent"], width=1.6),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=48,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def forecast_chart(
    records,
    tr: Translator,
    unit: str = "",
) -> go.Figure | None:
    """
    Forecast chart — only rendered when a record carries a forecast block.

    Returns None when no forecast data exists (the view then shows a calm empty
    state instead). Never fabricates a trajectory.
    """
    latest_fc = None
    anchor_ts = None
    for rec in reversed(records):
        if rec.forecast is not None and rec.forecast.predictions:
            latest_fc = rec.forecast
            anchor_ts = rec.timestamp
            break
    if latest_fc is None:
        return None

    steps = list(range(1, len(latest_fc.predictions) + 1))
    fig = go.Figure()
    # Uncertainty band (if provided).
    if latest_fc.lower and latest_fc.upper and len(latest_fc.lower) == len(latest_fc.upper):
        fig.add_trace(
            go.Scatter(
                x=steps + steps[::-1],
                y=list(latest_fc.upper) + list(latest_fc.lower)[::-1],
                fill="toself",
                fillcolor=CHART["band_color"],
                line=dict(width=0),
                name=tr.t("forecast_band"),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=latest_fc.predictions,
            mode="lines+markers",
            line=dict(color=CHART["forecast_color"], width=2, dash=CHART["forecast_dash"]),
            marker=dict(size=6, color=CHART["forecast_color"]),
            name=tr.t("forecast_value"),
        )
    )
    layout = plotly_layout(CHART["height_detail"], rtl=tr.is_rtl)
    layout["xaxis"]["title"] = {"text": tr.t("forecast_step"), "font": {"size": 11}}
    layout["yaxis"]["title"] = {"text": unit, "font": {"size": 11}}
    layout["hovermode"] = "x"
    fig.update_layout(**layout)
    return fig


def wqi_class_distribution(diag: WqiDiagnostic, tr: Translator) -> go.Figure:
    """Horizontal bar of the historical WQI class distribution."""
    rows = diag.class_share()
    labels = [c for c, _, _ in rows]
    counts = [n for _, n, _ in rows]
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker=dict(color=[_WQI_CLASS_COLORS.get(c, COLOR["ink_3"]) for c in labels]),
            hovertemplate="%{x} " + tr.t("count") + "<extra>%{y}</extra>",
        )
    )
    layout = plotly_layout(220, rtl=tr.is_rtl)
    layout["xaxis"]["title"] = {"text": tr.t("count"), "font": {"size": 11}}
    layout["showlegend"] = False
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    return fig
