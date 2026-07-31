"""
forecast.py — Short-term forecast view.

The monitoring pipeline computes a recursive multi-step forecast, but the
current export contract does NOT persist it (see DATA_CONTRACT.md). This view is
fully wired: if a record ever carries a ``forecast`` block it renders the
trajectory and its uncertainty band. Until then it shows a calm, non-alarming
empty state and never fabricates values.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components import charts, layout
from dashboard.context import AppContext


def render(ctx: AppContext) -> None:
    tr = ctx.tr
    layout.section(tr.t("forecast_title"))

    if not ctx.params:
        return

    names = ctx.parameter_names
    selected = st.selectbox(tr.t("parameter"), names, key="forecast_param")
    pdata = ctx.params[selected]

    fig = charts.forecast_chart(pdata.records, tr, unit=pdata.display_unit)
    if fig is None:
        layout.empty_state(
            tr.t("forecast_none_title"),
            tr.t("forecast_none_body"),
            tone="neutral",
        )
        return

    st.plotly_chart(fig, width="stretch", config=layout.plotly_config())
