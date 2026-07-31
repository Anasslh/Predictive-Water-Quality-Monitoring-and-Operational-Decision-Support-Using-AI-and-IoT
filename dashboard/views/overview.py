"""
overview.py — System overview: latest state per parameter at a glance.

Answers, for each monitored parameter: what is the current observation, what
changed vs the previous measured value, and is the latest evaluated record
abnormal. Model availability and source freshness are kept semantically separate.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components import layout
from dashboard.components.format import DASH, TREND_GLYPH, fmt_age, fmt_signed, fmt_value
from dashboard.context import AppContext
from dashboard.services import transforms as tx


def render(ctx: AppContext) -> None:
    tr = ctx.tr
    layout.section(tr.t("overview_title"), tr.t("overview_intro"))

    # ── Top-line system KPIs ────────────────────────────────────────────────
    n_params = len(ctx.params)
    n_anom = tx.count_active_anomalies(ctx.params)
    n_reviews = tx.count_pending_reviews(ctx.params)

    c1, c2, c3 = st.columns(3)
    with c1:
        layout.kpi_tile(tr.t("parameters_monitored"), str(n_params))
    with c2:
        layout.kpi_tile(
            tr.t("active_anomalies"), str(n_anom),
            status="critical" if n_anom else "ok",
        )
    with c3:
        layout.kpi_tile(
            tr.t("pending_reviews"), str(n_reviews),
            status="warn" if n_reviews else "neutral",
        )

    # ── Per-parameter cards ─────────────────────────────────────────────────
    layout.section(tr.t("parameters_monitored"))
    cols = st.columns(min(3, max(1, n_params)))
    for i, (name, pdata) in enumerate(ctx.params.items()):
        with cols[i % len(cols)]:
            _parameter_card(ctx, name, pdata)


def _parameter_card(ctx: AppContext, name: str, pdata) -> None:
    tr = ctx.tr
    unit = pdata.display_unit

    latest_actual_rec = pdata.latest_actual_record
    latest_rec = pdata.latest
    actual_val = latest_actual_rec.actual_value if latest_actual_rec else None
    pred_val = latest_rec.predicted_value if latest_rec else None

    direction, delta = tx.trend_direction(pdata)
    anom_state = tx.latest_anomaly_state(pdata)
    fresh = tx.freshness(
        pdata,
        now=ctx.now,
        multiplier=ctx.settings.fresh_multiplier,
        source_mode=ctx.settings.data_source_mode,
    )
    model = tx.model_status(pdata)

    anom_status = {"anomaly": "critical", "normal": "ok"}.get(anom_state, "neutral")
    anom_label = {"anomaly": tr.t("anomaly"), "normal": tr.t("normal")}.get(anom_state, tr.t("unknown"))

    trend_txt = f'{TREND_GLYPH.get(direction, DASH)} {fmt_signed(delta, unit)}' if delta is not None else DASH
    if fresh.state == "historical":
        source_text = tr.t("historical_dataset")
    else:
        fresh_label = {"fresh": tr.t("fresh"), "stale": tr.t("stale")}.get(
            fresh.state, tr.t("unknown")
        )
        source_text = f"{fmt_age(fresh.age_seconds, tr)} · {fresh_label}"
    model_label = (
        tr.t("model_active") if model == "active" else tr.t("model_unavailable")
    )

    # A single HTML fragment avoids the empty Streamlit blocks produced when an
    # opening and closing card tag are emitted by separate markdown calls.
    html = (
        '<div class="wq-tile" style="margin-bottom:0.8rem;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div style="font-weight:640;font-size:1.02rem;color:var(--wq-ink);">{name}</div>'
        f'{layout.status_pill(anom_label, anom_status)}</div>'
        f'<div class="wq-tile-value" dir="ltr" style="margin-top:0.3rem;">{fmt_value(actual_val, unit)}</div>'
        f'<div class="wq-tile-sub">{tr.t("trend")}: {trend_txt}</div>'
        f'<div class="wq-tile-sub">{tr.t("latest_predicted")}: {fmt_value(pred_val, unit)}</div>'
        f'<div class="wq-tile-sub">{tr.t("data_status")}: {source_text}</div>'
        f'<div class="wq-tile-sub">{tr.t("model_health")}: {model_label}</div>'
        '</div>'
    )
    st.markdown(
        html,
        unsafe_allow_html=True,
    )
