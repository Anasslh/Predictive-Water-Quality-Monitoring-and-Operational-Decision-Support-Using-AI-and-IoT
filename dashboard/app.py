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
from datetime import date, datetime, time, timedelta, timezone
import hashlib
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
from dashboard.services.db_service import (  # noqa: E402
    WarmQueryResult,
    WarmTierError,
    fetch_historical_parameters,
)
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


@st.cache_data(ttl=300, show_spinner=False)
def _load_warm(
    _connection_string: str,
    connection_fingerprint: str,
    site_id: str,
    start_iso: str,
    end_iso_exclusive: str,
    timeout_seconds: int,
) -> WarmQueryResult:
    """Load one bounded historical range; never cache the connection secret."""
    del connection_fingerprint  # non-secret cache partition; Streamlit hashes it
    return fetch_historical_parameters(
        _connection_string,
        site_id,
        datetime.fromisoformat(start_iso),
        datetime.fromisoformat(end_iso_exclusive),
        timeout_seconds=timeout_seconds,
    )


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

    # ── Sidebar: identity and language ─────────────────────────────────────
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

        source_options = ["current"]
        if settings.warm_configured:
            source_options.append("archive")
        source_labels = [
            tr.t("historical_archive" if value == "archive" else "current_monitoring")
            for value in source_options
        ]
        saved_source = st.session_state.get("_data_source_code", "current")
        if saved_source not in source_options:
            saved_source = "current"
        expected_label = source_labels[source_options.index(saved_source)]
        if st.session_state.get("data_source") not in source_labels:
            st.session_state.pop("data_source", None)
        source_label = st.radio(
            tr.t("data_view"),
            options=source_labels,
            index=source_labels.index(expected_label),
            key="data_source",
        )
        source = source_options[source_labels.index(source_label)]
        st.session_state._data_source_code = source
        if not settings.warm_configured:
            st.caption(tr.t("historical_unavailable"))

        warm_range: tuple[date, date] | None = None
        if source == "archive":
            today = datetime.now(timezone.utc).date()
            earliest = today - timedelta(days=settings.warm_lookback_days - 1)
            picked = st.date_input(
                tr.t("historical_range"),
                value=(earliest, today),
                min_value=earliest,
                max_value=today,
                key="warm_date_range",
            )
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                warm_range = (picked[0], picked[1])
            elif isinstance(picked, date):
                warm_range = (picked, picked)
        st.divider()

    source_error: str | None = None
    if source == "archive":
        start_date, end_date = warm_range or (
            datetime.now(timezone.utc).date(),
            datetime.now(timezone.utc).date(),
        )
        start_utc = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
        end_utc_exclusive = datetime.combine(
            end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        try:
            with st.spinner(tr.t("warm_loading")):
                warm_result = _load_warm(
                    settings.db_connection_string,
                    hashlib.sha256(
                        settings.db_connection_string.encode("utf-8")
                    ).hexdigest(),
                    settings.effective_warm_site_id,
                    start_utc.isoformat(),
                    end_utc_exclusive.isoformat(),
                    settings.db_timeout_seconds,
                )
            params = warm_result.params
            names = list(params)
        except WarmTierError as exc:
            source_error = exc.code
            names, params = [], {}
    else:
        names, params = _load(
            str(settings.exports_dir),
            settings.retention_days,
            _exports_mtime(settings.exports_dir),
        )

    # ── Sidebar navigation (gated by the selected source) ───────────────────
    with st.sidebar:
        views = available_views(params)
        labels = [tr.t(key) for key, _ in views]
        if st.session_state.get("nav") not in labels:
            st.session_state.pop("nav", None)
        choice = st.radio(tr.t("navigation"), options=labels, key="nav")

    ctx = AppContext(settings=settings, tr=tr, params=params, data_source=source)

    # ── Header ──────────────────────────────────────────────────────────────
    last_update = fmt_timestamp(_latest_update(ctx))
    effective_source_mode = "historical" if source == "archive" else settings.data_source_mode
    fresh_state = (
        tx.system_freshness(
            params,
            multiplier=settings.fresh_multiplier,
            source_mode=effective_source_mode,
        )
        if params
        else "unknown"
    )
    layout.page_header(
        tr,
        settings,
        freshness_state=fresh_state,
        last_update=last_update,
        source_note_override=(tr.t("historical_archive") if source == "archive" else None),
    )

    if source_error:
        key = f"warm_error_{source_error}"
        st.error(tr.t(key))

    # ── No data → calm empty state ──────────────────────────────────────────
    if not names:
        if source == "archive" and not source_error:
            layout.empty_state(
                tr.t("warm_no_records_title"), tr.t("warm_no_records_body"), tone="neutral"
            )
        elif not source_error:
            layout.empty_state(
                tr.t("no_parameters_title"), tr.t("no_parameters_body"), tone="neutral"
            )
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
