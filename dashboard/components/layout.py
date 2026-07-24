"""
layout.py — Streamlit layout primitives and the design-system chrome.

All shared UI furniture lives here: theme injection, the global header, section
headings, KPI tiles and status pills. Critical colours are applied inline (from
the theme tokens) so the components stay legible and intentional even if the
external stylesheet is disabled or only partially supported — the CSS file then
only adds refinement, never carries the whole look.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.components.format import DASH
from dashboard.config.settings import Settings
from dashboard.config.theme import COLOR, RADIUS, status_colors
from dashboard.i18n.translator import Translator

_CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "theme.css"


def plotly_config() -> dict:
    """Shared st.plotly_chart config: keep the toolbar minimal and unobtrusive."""
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
        ],
        "responsive": True,
    }


def inject_theme(tr: Translator) -> None:
    """Inject the stylesheet plus the per-language text direction."""
    css = _CSS_PATH.read_text(encoding="utf-8") if _CSS_PATH.exists() else ""
    direction = tr.dir
    align = tr.align
    st.markdown(
        f"<style>{css}\n"
        f":root {{ --wq-dir: {direction}; }}\n"
        f".stApp {{ direction: {direction}; }}\n"
        f".wq-align {{ text-align: {align}; }}\n"
        "</style>",
        unsafe_allow_html=True,
    )


def page_header(
    tr: Translator,
    settings: Settings,
    *,
    freshness_state: str,
    last_update: str,
) -> None:
    """Global header band: title, site, last update, freshness pill."""
    fg, bg = status_colors(
        {"fresh": "ok", "stale": "warn", "unknown": "neutral"}.get(freshness_state, "neutral")
    )
    fresh_label = {"fresh": tr.t("fresh"), "stale": tr.t("stale")}.get(
        freshness_state, tr.t("unknown")
    )
    st.markdown(
        f"""
        <div class="wq-header">
          <div class="wq-header-main">
            <div class="wq-header-title">{tr.t('app_title')}</div>
            <div class="wq-header-sub">{tr.t('app_subtitle')}</div>
          </div>
          <div class="wq-header-meta">
            <div class="wq-meta-item"><span class="wq-meta-k">{tr.t('site')}</span>
              <span class="wq-meta-v">{settings.site_name}</span></div>
            <div class="wq-meta-item"><span class="wq-meta-k">{tr.t('last_update')}</span>
              <span class="wq-meta-v">{last_update}</span></div>
            <div class="wq-meta-item">
              <span class="wq-pill" style="color:{fg};background:{bg};">&#9679; {fresh_label}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    """A section heading with a hairline rule."""
    sub = f'<div class="wq-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="wq-section"><div class="wq-section-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def status_pill(label: str, status: str) -> str:
    """Return an inline HTML pill (caller embeds it). status: ok|warn|critical|neutral."""
    fg, bg = status_colors(status)
    return (
        f'<span class="wq-pill" style="color:{fg};background:{bg};">'
        f'&#9679; {label}</span>'
    )


def kpi_tile(
    label: str,
    value: str,
    *,
    sub: str | None = None,
    status: str | None = None,
) -> None:
    """A single bordered KPI tile. Optional coloured left accent by status."""
    accent = ""
    if status:
        fg, _ = status_colors(status)
        accent = f"border-inline-start:3px solid {fg};"
    sub_html = f'<div class="wq-tile-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="wq-tile" style="{accent}">
          <div class="wq-tile-label">{label}</div>
          <div class="wq-tile-value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kv_table(rows: list[tuple[str, str]]) -> None:
    """Render a compact key/value definition table."""
    body = "".join(
        f'<div class="wq-kv-row"><div class="wq-kv-k">{k}</div>'
        f'<div class="wq-kv-v">{v if v else DASH}</div></div>'
        for k, v in rows
    )
    st.markdown(f'<div class="wq-kv">{body}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str, *, tone: str = "neutral") -> None:
    """
    A calm, non-alarming empty state (used for missing optional features).

    Deliberately understated — no bright colours, no warning iconography — so an
    unavailable optional feature never visually pollutes the interface.
    """
    _, bg = status_colors(tone)
    border = COLOR["border"]
    st.markdown(
        f"""
        <div class="wq-empty" style="background:{bg};border:1px solid {border};border-radius:{RADIUS};">
          <div class="wq-empty-title">{title}</div>
          <div class="wq-empty-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notice(text: str) -> None:
    """A quiet inline footnote (e.g. read-only reminder)."""
    st.markdown(f'<div class="wq-notice">{text}</div>', unsafe_allow_html=True)
