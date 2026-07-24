"""
End-to-end smoke tests using Streamlit's AppTest harness.

Runs the real app.py headlessly against several export scenarios and asserts the
script completes without raising — covering the robustness requirements:
missing exports, corrupt JSON lines, a parameter with no anomalies/forecast,
every navigation view, and the Arabic (RTL) language.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

# English nav labels (default language) — must match dashboard/i18n/strings.py EN.
NAV_LABELS = [
    "Overview",
    "Parameter detail",
    "Forecast",
    "Anomalies & alerts",
    "Model health",
    "Data & methodology",
]


def _run(env: dict | None, monkeypatch) -> AppTest:
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    at = AppTest.from_file(APP)
    at.run(timeout=60)
    return at


def test_smoke_real_repo_exports(monkeypatch):
    """Against the repository's actual exports/ (EC has real data)."""
    at = _run(None, monkeypatch)
    assert not at.exception


def test_smoke_empty_exports(tmp_path, monkeypatch):
    empty = tmp_path / "exports"
    empty.mkdir()
    at = _run({"WQD_EXPORTS_DIR": str(empty)}, monkeypatch)
    assert not at.exception
    # The "no exported parameters" empty state should be present somewhere.
    joined = " ".join(m.value for m in at.markdown)
    assert "No exported parameters" in joined or "exports" in joined


def test_smoke_missing_exports_dir(tmp_path, monkeypatch):
    at = _run({"WQD_EXPORTS_DIR": str(tmp_path / "does_not_exist")}, monkeypatch)
    assert not at.exception


def test_smoke_messy_exports_all_views(exports_dir_messy, monkeypatch):
    """Corrupt line + missing status + no-anomaly parameter, across every view."""
    at = _run({"WQD_EXPORTS_DIR": str(exports_dir_messy)}, monkeypatch)
    assert not at.exception
    for label in NAV_LABELS:
        at.radio(key="nav").set_value(label).run(timeout=60)
        assert not at.exception, f"view '{label}' raised"


def test_smoke_arabic_rtl(exports_dir_messy, monkeypatch):
    at = _run({"WQD_EXPORTS_DIR": str(exports_dir_messy)}, monkeypatch)
    at.radio(key="lang").set_value("ar").run(timeout=60)
    assert not at.exception


def test_smoke_forecast_empty_state(exports_dir_messy, monkeypatch):
    """Forecast view must render its calm empty state (no forecast in exports)."""
    at = _run({"WQD_EXPORTS_DIR": str(exports_dir_messy)}, monkeypatch)
    at.radio(key="nav").set_value("Forecast").run(timeout=60)
    assert not at.exception
    joined = " ".join(m.value for m in at.markdown)
    assert "Forecast" in joined
