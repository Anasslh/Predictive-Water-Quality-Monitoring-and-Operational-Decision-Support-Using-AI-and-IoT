"""
anomalies.py — Recent anomalies and retraining alerts across all parameters.

Aggregates every record flagged ``is_anomaly = true`` in the current window into
one chronological table: timestamp, parameter, measured value, predicted value,
anomaly score, state and any retraining alert. A calm message is shown when the
window contains no anomalies.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import layout
from dashboard.components.format import DASH, fmt_number, fmt_timestamp
from dashboard.context import AppContext
from dashboard.services import transforms as tx


def render(ctx: AppContext) -> None:
    tr = ctx.tr
    layout.section(tr.t("anomalies_title"))

    rows: list[dict] = []
    for name, pdata in ctx.params.items():
        unit = pdata.display_unit
        for rec in tx.anomaly_records(pdata):
            rows.append(
                {
                    tr.t("timestamp"): fmt_timestamp(rec.timestamp),
                    tr.t("parameter"): name,
                    tr.t("actual"): _v(rec.actual_value, unit),
                    tr.t("expected"): _v(rec.predicted_value, unit),
                    tr.t("score"): fmt_number(rec.anomaly_score, 3),
                    tr.t("state"): tr.t("anomaly"),
                    tr.t("retrain_alert"): rec.retrain_alert or DASH,
                    "_sort": rec.timestamp.isoformat() if rec.timestamp else "",
                }
            )

    if not rows:
        layout.empty_state(tr.t("anomalies_none"), "", tone="ok")
        return

    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    st.dataframe(df, width="stretch", hide_index=True)


def _v(value: float | None, unit: str) -> str:
    if value is None:
        return DASH
    return f"{fmt_number(value)} {unit}".strip()
