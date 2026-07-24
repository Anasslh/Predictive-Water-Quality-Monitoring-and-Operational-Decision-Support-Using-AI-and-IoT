"""
methodology.py — Data & methodology (restrained secondary reference page).

Documents the data source, units, model versions, known limitations, and the
historical WQI diagnostic with its full caveats. The WQI is presented strictly
as a documented historical diagnostic computed with drinking-water reference
standards — never as a live operational or safety verdict.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components import charts, layout
from dashboard.components.format import DASH, fmt_number
from dashboard.config import reference_limits as refs
from dashboard.context import AppContext
from dashboard.services.wqi import load_wqi_diagnostic

# Factual limitations about the data/exports (not invented science). Bilingual.
_LIMITATIONS: list[tuple[str, str]] = [
    (
        "Single station, one year: all data is from the C-1 station (Ramgarh, India), 365 daily rows.",
        "محطة واحدة، سنة واحدة: كل البيانات من محطة C-1 (رامغار، الهند)، 365 صفاً يومياً.",
    ),
    (
        "Forecasts are computed by the pipeline but not persisted in the current export contract.",
        "يحسب النظام التنبؤات لكنها غير محفوظة في عقد التصدير الحالي.",
    ),
    (
        "R² and an explicit drift status are not part of the status export.",
        "معامل التحديد R² وحالة الانزياح الصريحة غير مُضمَّنة في تصدير الحالة.",
    ),
    (
        "The WQI uses drinking-water standards and is a historical diagnostic only, not a live verdict.",
        "يستخدم مؤشر WQI معايير مياه الشرب وهو تشخيص تاريخي فقط وليس حكماً لحظياً.",
    ),
]


def render(ctx: AppContext) -> None:
    tr = ctx.tr
    s = ctx.settings
    ar = tr.is_rtl

    layout.section(tr.t("methodology_title"))

    # ── Data source & units ─────────────────────────────────────────────────
    rows = [
        (tr.t("data_source"), s.site_name),
        (tr.t("generated_from"), str(s.exports_dir)),
        (tr.t("profile"), tr.t(f"profile_{s.water_use_profile}")),
    ]
    for name, pdata in ctx.params.items():
        unit = pdata.display_unit or DASH
        ver = pdata.status.model_version if pdata.status and pdata.status.model_version else DASH
        rows.append((f"{name} · {tr.t('units')}", unit))
        rows.append((f"{name} · {tr.t('model_version')}", ver))
    layout.kv_table(rows)

    # ── Known limitations ───────────────────────────────────────────────────
    layout.section(tr.t("known_limitations"))
    bullets = "".join(f"<li>{(a if ar else e)}</li>" for e, a in _LIMITATIONS)
    st.markdown(
        f'<ul class="wq-limits" style="font-size:0.83rem;color:var(--wq-ink-2);line-height:1.6;">{bullets}</ul>',
        unsafe_allow_html=True,
    )

    # ── WQI diagnostic (only if the processed artifact exists) ──────────────
    diag = load_wqi_diagnostic(s.wqi_csv, s.wqi_methodology_txt)
    if diag is None:
        return  # No WQI artifact → omit the block entirely, no placeholder.

    layout.section(tr.t("wqi_title"), tr.t("wqi_context"))
    st.markdown(
        f'<div class="wq-shap-intro">{tr.t("wqi_intro")}</div>', unsafe_allow_html=True
    )

    meta = st.columns([1.2, 1])
    with meta[0]:
        # Standards table from documented reference values.
        std_rows: list[tuple[str, str]] = [
            (tr.t("wqi_method"), "WAWQI — Brown et al. (1972)"),
        ]
        for name in ("pH", "EC", "Turbidity"):
            for line in refs.reference_lines(name):
                std_rows.append((f"{name} · Si", f"{line.value:g}  ·  {line.source}"))
        layout.kv_table(std_rows)
    with meta[1]:
        wqi_rows = [
            ("WQI min", fmt_number(diag.wqi_min)),
            ("WQI mean", fmt_number(diag.wqi_mean)),
            ("WQI max", fmt_number(diag.wqi_max)),
        ]
        layout.kv_table(wqi_rows)

    layout.section(tr.t("wqi_class_dist"))
    st.plotly_chart(
        charts.wqi_class_distribution(diag, tr),
        width="stretch", config=layout.plotly_config(),
    )
