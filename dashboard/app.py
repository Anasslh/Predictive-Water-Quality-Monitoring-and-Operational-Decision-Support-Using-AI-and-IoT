"""
app.py — Water Quality Monitoring & Operational Decision-Support dashboard.

Read-only Streamlit application that consumes the pipeline's export files
(exports/<param>.jsonl + <param>_status.json) and never imports model pickles or
training code. Run from the repository root:

    streamlit run dashboard/app.py

Configuration is environment-driven (see dashboard/config/settings.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Make the repository root importable so `dashboard.*` resolves when Streamlit
# runs this file as a script (Streamlit only puts the script's own dir on path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from dashboard.components import layout  # noqa: E402
from dashboard.components.format import fmt_timestamp  # noqa: E402
from dashboard.config.settings import LANGUAGES, get_settings  # noqa: E402
from dashboard.context import AppContext  # noqa: E402
from dashboard.i18n.translator import Translator  # noqa: E402
from dashboard.models.schemas import ParameterData  # noqa: E402
from dashboard.services.discovery import discover_parameters  # noqa: E402
from dashboard.services.loaders import load_all_parameters  # noqa: E402
from dashboard.services import transforms as tx  # noqa: E402
from dashboard.views import (  # noqa: E402
    anomalies as v_anomalies,
    forecast as v_forecast,
    methodology as v_methodology,
    model_health as v_health,
    overview as v_overview,
    parameter_detail as v_parameter,
)

_LANG_LABELS = {"en": "English", "ar": "العربية"}
ViewEntry = tuple[str, Callable[[AppContext], None]]

# Base view registry. Forecast is inserted only when valid exported values exist.
_BASE_VIEWS: list[ViewEntry] = [
    ("nav_overview", v_overview.render),
    ("nav_parameter", v_parameter.render),
    ("nav_anomalies", v_anomalies.render),
    ("nav_health", v_health.render),
    ("nav_methodology", v_methodology.render),
]


def available_views(params: dict[str, ParameterData]) -> list[ViewEntry]:
    """Return navigation entries, gating Forecast on confirmed export data."""
    views = list(_BASE_VIEWS)
    if tx.has_forecast_data(params):
        views.insert(2, ("nav_forecast", v_forecast.render))
    return views


@st.cache_data(show_spinner=False)
def _load(exports_dir: str, retention_days: int, cache_key: float):
    """
    Load discovery + all parameter data.

    ``cache_key`` (the exports dir mtime) invalidates the cache when files change,
    so the dashboard reflects new measurements without a manual restart.
    """
    names = discover_parameters(exports_dir)
    params = load_all_parameters(names, exports_dir, retention_days)
    return names, params


def _exports_mtime(exports_dir: Path) -> float:
    """Latest mtime across the exports directory (cache-busting key)."""
    if not exports_dir.is_dir():
        return 0.0
    times = [p.stat().st_mtime for p in exports_dir.glob("*.json*") if p.is_file()]
    return max(times) if times else 0.0


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title="Water Quality Monitoring",
        page_icon="🌊",  # browser tab only; not shown inside the interface
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Language (session-persistent) ───────────────────────────────────────
    if "lang" not in st.session_state:
        st.session_state.lang = settings.default_lang if settings.default_lang in LANGUAGES else "en"
    tr = Translator(st.session_state.lang)

    layout.inject_theme(tr)

    # Load before building navigation so optional views can be data-gated.
    names, params = _load(
        str(settings.exports_dir), settings.retention_days, _exports_mtime(settings.exports_dir)
    )

    # ── Sidebar: identity, language, navigation ─────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div class="wq-side-title">{tr.t("app_title")}</div>'
            f'<div class="wq-side-sub">{tr.t("app_subtitle")}</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        lang = st.radio(
            tr.t("language"),
            options=list(LANGUAGES),
            format_func=lambda c: _LANG_LABELS.get(c, c),
            index=list(LANGUAGES).index(st.session_state.lang),
            horizontal=True,
            key="lang",
        )
        tr = Translator(lang)  # reflect an in-place change immediately
        st.divider()
        views = available_views(params)
        labels = [tr.t(key) for key, _ in views]
        if st.session_state.get("nav") not in labels:
            st.session_state.pop("nav", None)
        choice = st.radio(tr.t("navigation"), options=labels, key="nav")

    ctx = AppContext(settings=settings, tr=tr, params=params)

    # ── Header ──────────────────────────────────────────────────────────────
    last_update = fmt_timestamp(_latest_update(ctx))
    fresh_state = (
        tx.system_freshness(
            params,
            multiplier=settings.fresh_multiplier,
            source_mode=settings.data_source_mode,
        )
        if params
        else "unknown"
    )
    layout.page_header(tr, settings, freshness_state=fresh_state, last_update=last_update)

    # ── No data → calm empty state ──────────────────────────────────────────
    if not names:
        layout.empty_state(tr.t("no_parameters_title"), tr.t("no_parameters_body"), tone="neutral")
        return

    # ── Route to the selected view ──────────────────────────────────────────
    label_to_fn = {tr.t(key): fn for key, fn in views}
    render_fn = label_to_fn.get(choice, v_overview.render)
    render_fn(ctx)


def _latest_update(ctx: AppContext):
    """Most recent measurement timestamp across all parameters."""
    latest = None
    for pdata in ctx.params.values():
        rec = pdata.latest
        if rec and rec.timestamp:
            if latest is None or rec.timestamp > latest:
                latest = rec.timestamp
    return latest


if __name__ == "__main__":
    main()
