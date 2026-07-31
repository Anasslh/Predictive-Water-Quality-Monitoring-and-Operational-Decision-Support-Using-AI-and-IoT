"""
model_health.py — Per-parameter model monitoring data and 30-day performance.

Surfaces exactly what the status export provides. Optional unavailable values
are omitted; no pass/fail health rule is inferred from raw performance metrics.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components import layout
from dashboard.components.format import fmt_number, fmt_pct, fmt_timestamp
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
    model = tx.model_status(pdata)
    model_label = (
        tr.t("model_active") if model == "active" else tr.t("model_unavailable")
    )

    st.markdown(
        f'<div class="wq-section" style="margin-top:1rem;">'
        f'<div class="wq-section-title">{name} '
        f'{layout.status_pill(model_label, "neutral")}</div></div>',
        unsafe_allow_html=True,
    )

    if status is None:
        layout.empty_state(tr.t("not_available"), "", tone="neutral")
        return

    perf = status.performance
    skill_val = None if perf.insufficient_data else perf.skill_vs_persistence_pct

    metrics: list[tuple[str, str]] = []
    if skill_val is not None:
        metrics.append((tr.t("skill"), fmt_pct(skill_val)))
    if perf.rmse is not None:
        metrics.append((tr.t("rmse"), fmt_number(perf.rmse)))
    if perf.mae is not None:
        metrics.append((tr.t("mae"), fmt_number(perf.mae)))
    if perf.r2 is not None:
        metrics.append((tr.t("r2"), fmt_number(perf.r2)))
    metrics.append((tr.t("n_measurements"), str(perf.n_measurements)))

    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        with column:
            layout.kpi_tile(label, value)

    rows = [
        (tr.t("model_version"), status.model_version or ""),
        (tr.t("last_promoted"), fmt_timestamp(status.last_promoted_at)),
        (tr.t("pending_approvals"), str(status.pending_approvals)),
        (tr.t("consecutive_rejections"), str(status.consecutive_rejections)),
    ]
    layout.kv_table(rows)
