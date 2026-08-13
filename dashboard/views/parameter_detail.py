"""
parameter_detail.py — Deep view for a single parameter.

Current value, latest prediction and error, measured-vs-predicted history with
anomaly markers, a date-range filter, data-quality indicators and SHAP context.
Drinking-water WQI references are intentionally confined to Methodology.
"""

from __future__ import annotations

from datetime import datetime, time
import pandas as pd
import streamlit as st

from dashboard.components import charts, layout
from dashboard.components.format import DASH, fmt_timestamp, fmt_value
from dashboard.components.shap_panel import render_shap
from dashboard.context import AppContext
from dashboard.models.schemas import ParameterData, MeasurementRecord
from dashboard.services.db_service import fetch_historical_parameter_data


def render(ctx: AppContext) -> None:
    tr = ctx.tr
    if not ctx.params:
        return

    names = ctx.parameter_names
    selected = st.selectbox(
        tr.t("parameter"), names,
        index=names.index(st.session_state.get("detail_param", names[0]))
        if st.session_state.get("detail_param") in names else 0,
        key="detail_param",
    )
    
    # ── HOT / WARM TIER ROUTING LOGIC ───────────────────────────────────────
    use_warm_tier = st.toggle("Load 12-Month Historical Archive (Warm Tier Database)", value=False)
    
    hot_pdata = ctx.params[selected]
    
    if use_warm_tier:
        with st.spinner(f"Querying local SQL Server for {selected} history..."):
            historical_df = fetch_historical_parameter_data(selected)
            warm_records = []
            
            if historical_df is not None and not historical_df.empty:
                for _, row in historical_df.iterrows():
                    try:
                        obj = {
                            "timestamp": str(row["timestamp"]),
                            "parameter_name": selected,
                            "predicted_value": None if pd.isna(row.get("predicted_value")) else row["predicted_value"],
                            "actual_value": None if pd.isna(row.get("actual_value")) else row["actual_value"],
                            "shap_top_features": row["shap_top_features"],
                            "is_anomaly": None if pd.isna(row.get("is_anomaly")) else bool(row["is_anomaly"]),
                            "anomaly_score": None if pd.isna(row.get("anomaly_score")) else row["anomaly_score"],
                            "retrain_alert": None if pd.isna(row.get("retrain_alert")) else row["retrain_alert"]
                        }
                        warm_records.append(MeasurementRecord.from_dict(obj))
                    except Exception:
                        pass
            
            pdata = ParameterData(
                name=selected,
                unit=hot_pdata.unit,
                records=warm_records,
                status=hot_pdata.status,
                load_errors=0
            )
    else:
        pdata = hot_pdata
        
    unit = pdata.display_unit

    layout.section(f"{selected}", _subtitle(pdata, tr))

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

    layout.section(tr.t("history"))
    filtered = _date_range_filter(pdata, tr)
    fig = charts.actual_vs_predicted(filtered, tr)
    st.plotly_chart(fig, width="stretch", config=layout.plotly_config())

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
        rows.append(("Skipped malformed lines", str(pdata.load_errors)))
    layout.kv_table(rows)