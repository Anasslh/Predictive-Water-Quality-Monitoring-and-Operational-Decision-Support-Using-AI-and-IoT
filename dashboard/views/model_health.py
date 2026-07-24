"""
model_health.py — Per-parameter model status and 30-day performance.

Surfaces exactly what the status export provides: model version, RMSE, MAE,
skill vs persistence, measurement count, pending approvals, consecutive
rejections, last model update. Fields that the pipeline does not export (R²,
explicit drift status) are shown as "—" without alarm rather than invented.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components import layout
from dashboard.components.format import DASH, fmt_number, fmt_pct, fmt_timestamp
from dashboard.context import AppContext
from dashboard.services import transforms as tx


def render(ctx: AppContext) -> None:
    tr = ctx.tr
    layout.section(tr.t("health_title"))

    for name, pdata in ctx.params.items():
        _parameter_health(ctx, name, pdata)


def _parameter_health(ctx: AppContext, name: str, pdata) -> None:
    tr = ctx.tr
    status = pdata.status
    health = tx.health_token(pdata)

    st.markdown(
        f'<div class="wq-section" style="margin-top:1rem;">'
        f'<div class="wq-section-title">{name} '
        f'{layout.status_pill(_health_label(health, tr), health)}</div></div>',
        unsafe_allow_html=True,
    )

    if status is None:
        layout.empty_state(tr.t("not_available"), "", tone="neutral")
        return

    perf = status.performance
    skill_val = None if perf.insufficient_data else perf.skill_vs_persistence_pct

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        layout.kpi_tile(tr.t("skill"), fmt_pct(skill_val),
                        status="ok" if (skill_val or 0) > 0 else ("warn" if skill_val is not None else "neutral"))
    with c2:
        layout.kpi_tile(tr.t("rmse"), fmt_number(perf.rmse))
    with c3:
        layout.kpi_tile(tr.t("mae"), fmt_number(perf.mae))
    with c4:
        layout.kpi_tile(tr.t("n_measurements"), str(perf.n_measurements))

    rows = [
        (tr.t("model_version"), status.model_version or DASH),
        (tr.t("last_promoted"), fmt_timestamp(status.last_promoted_at)),
        (tr.t("r2"), DASH),                    # not exported by the pipeline
        (tr.t("drift_status"), DASH),          # not exported by the pipeline
        (tr.t("pending_approvals"), str(status.pending_approvals)),
        (tr.t("consecutive_rejections"), str(status.consecutive_rejections)),
    ]
    layout.kv_table(rows)


def _health_label(token: str, tr) -> str:
    return {
        "ok": tr.t("health_ok"),
        "warn": tr.t("health_warn"),
        "critical": tr.t("health_critical"),
    }.get(token, tr.t("health_unknown"))
