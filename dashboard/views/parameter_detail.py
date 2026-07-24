"""
parameter_detail.py — Deep view for a single parameter.

Current value, latest prediction and error, the actual-vs-predicted history with
anomaly markers and documented reference line, a date-range filter, data-quality
indicators, the SHAP explanation, and the model-status summary.
"""

from __future__ import annotations

from datetime import datetime, time

import streamlit as st

from dashboard.components import charts, layout
from dashboard.components.format import DASH, fmt_timestamp, fmt_value
from dashboard.components.shap_panel import render_shap
from dashboard.context import AppContext
from dashboard.models.schemas import ParameterData


def render(ctx: AppContext) -> None:
    tr = ctx.tr
    if not ctx.params:
        return

    names = ctx.parameter_names
    # Parameter selector (remembered across reruns).
    selected = st.selectbox(
        tr.t("parameter"), names,
        index=names.index(st.session_state.get("detail_param", names[0]))
        if st.session_state.get("detail_param") in names else 0,
        key="detail_param",
    )
    pdata = ctx.params[selected]
    unit = pdata.display_unit

    layout.section(f"{selected}", _subtitle(pdata, tr))

    # ── Headline tiles ──────────────────────────────────────────────────────
    latest_actual = pdata.latest_actual_record
    latest = pdata.latest
    actual_val = latest_actual.actual_value if latest_actual else None
    pred_val = latest.predicted_value if latest else None
    error = latest_actual.residual if latest_actual else None
    measured_at = latest_actual.timestamp if latest_actual else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        layout.kpi_tile(tr.t("current_value"), fmt_value(actual_val, unit))
    with c2:
        layout.kpi_tile(tr.t("predicted_value"), fmt_value(pred_val, unit))
    with c3:
        layout.kpi_tile(tr.t("prediction_error"), fmt_value(error, unit))
    with c4:
        layout.kpi_tile(tr.t("measured_at"), fmt_timestamp(measured_at))

    # ── History chart with date-range filter ────────────────────────────────
    layout.section(tr.t("history"))
    filtered = _date_range_filter(pdata, tr)
    fig = charts.actual_vs_predicted(
        filtered, tr, water_use_profile=ctx.settings.water_use_profile
    )
    st.plotly_chart(fig, width="stretch", config=layout.plotly_config())

    # ── Data quality + explanation, side by side ────────────────────────────
    left, right = st.columns([1, 1.4])
    with left:
        layout.section(tr.t("data_quality"))
        _data_quality(filtered, tr)
    with right:
        layout.section(tr.t("explanation"))
        render_shap(latest, tr)


def _subtitle(pdata: ParameterData, tr) -> str:
    if pdata.status and pdata.status.model_version:
        return f'{tr.t("model_version")}: {pdata.status.model_version}'
    return ""


def _date_range_filter(pdata: ParameterData, tr) -> ParameterData:
    """Slice records to a user-selected date range. No-op when < 2 dated records."""
    dated = [r for r in pdata.records if r.timestamp is not None]
    if len(dated) < 2:
        return pdata

    lo = dated[0].timestamp.date()
    hi = dated[-1].timestamp.date()
    picked = st.date_input(
        tr.t("date_range"), value=(lo, hi), min_value=lo, max_value=hi,
        key=f"range_{pdata.name}",
    )
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        return pdata
    start, end = picked
    start_dt = datetime.combine(start, time.min).replace(tzinfo=dated[0].timestamp.tzinfo)
    end_dt = datetime.combine(end, time.max).replace(tzinfo=dated[0].timestamp.tzinfo)
    kept = [
        r for r in pdata.records
        if r.timestamp is None or (start_dt <= r.timestamp <= end_dt)
    ]
    return ParameterData(
        name=pdata.name, unit=pdata.unit, records=kept,
        status=pdata.status, load_errors=pdata.load_errors,
    )


def _data_quality(pdata: ParameterData, tr) -> None:
    total = len(pdata.records)
    missing_actual = sum(1 for r in pdata.records if r.actual_value is None)
    rows = [
        (tr.t("records_in_window"), str(total)),
        (tr.t("missing_actuals"), str(missing_actual)),
    ]
    if pdata.load_errors:
        # Reported quietly as a data-quality figure, not an alarm banner.
        rows.append(("Skipped malformed lines", str(pdata.load_errors)))
    layout.kv_table(rows)
