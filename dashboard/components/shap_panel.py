"""
shap_panel.py — Render SHAP feature attributions as operator-readable text.

Translates the numeric SHAP output into plain language while preserving the
original technical values. Uses influence language only ("influenced the
prediction", "pushed the prediction up/down") and never causal language about
the water itself — SHAP explains the model, not the physical process.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.format import DASH, fmt_number
from dashboard.config.theme import COLOR
from dashboard.i18n.translator import Translator
from dashboard.models.schemas import MeasurementRecord


def _direction_label(feature_direction: str | None, shap_value: float | None, tr: Translator) -> tuple[str, str]:
    """Return (text, color) describing whether the feature raised/lowered the prediction."""
    up = None
    if feature_direction in ("positive", "up", "+"):
        up = True
    elif feature_direction in ("negative", "down", "-"):
        up = False
    elif shap_value is not None:
        up = shap_value > 0
    if up is None:
        return DASH, COLOR["ink_3"]
    return (tr.t("increases"), COLOR["accent_2"]) if up else (tr.t("decreases"), COLOR["ink_2"])


def render_shap(record: MeasurementRecord | None, tr: Translator) -> None:
    """Render the explanation panel for one record's prediction."""
    st.markdown(f'<div class="wq-shap-intro">{tr.t("explanation_intro")}</div>', unsafe_allow_html=True)

    if record is None or not record.shap_top_features:
        st.markdown(
            f'<div class="wq-shap-empty">{tr.t("not_available")}</div>',
            unsafe_allow_html=True,
        )
        return

    # Magnitude bar is scaled to the largest |shap| in this record.
    magnitudes = [abs(f.shap_value) for f in record.shap_top_features if f.shap_value is not None]
    max_mag = max(magnitudes) if magnitudes else 1.0

    rows_html = []
    for f in record.shap_top_features:
        dir_text, dir_color = _direction_label(f.direction, f.shap_value, tr)
        mag = abs(f.shap_value) if f.shap_value is not None else 0.0
        width = int((mag / max_mag) * 100) if max_mag else 0
        rows_html.append(
            f"""
            <div class="wq-shap-row">
              <div class="wq-shap-feat" title="{f.feature}">{f.feature}</div>
              <div class="wq-shap-bar-wrap">
                <div class="wq-shap-bar" style="width:{width}%;background:{dir_color};"></div>
              </div>
              <div class="wq-shap-val">{fmt_number(f.shap_value, 3)}</div>
              <div class="wq-shap-dir" style="color:{dir_color};">{dir_text}</div>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="wq-shap">
          <div class="wq-shap-head">
            <div>{tr.t('feature')}</div><div></div>
            <div>{tr.t('influence')}</div><div>{tr.t('direction')}</div>
          </div>
          {''.join(rows_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )
