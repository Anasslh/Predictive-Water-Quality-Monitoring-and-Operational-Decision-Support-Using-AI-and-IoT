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
    "Anomalies & alerts",
    "Model monitoring",
    "Data & methodology",
]

AR_NAV_LABELS = [
    "نظرة عامة",
    "تفاصيل المؤشر",
    "الحالات الشاذة والتنبيهات",
    "مراقبة النموذج",
    "البيانات والمنهجية",
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
    assert list(at.radio(key="nav").options) == AR_NAV_LABELS
    joined = " ".join(m.value for m in at.markdown)
    assert "direction: rtl" in joined

    for label in AR_NAV_LABELS:
        at.radio(key="nav").set_value(label).run(timeout=60)
        assert not at.exception, f"Arabic view '{label}' raised"

    at.radio(key="nav").set_value("الحالات الشاذة والتنبيهات").run(timeout=60)
    assert at.dataframe
    assert "الطابع الزمني" in list(at.dataframe[0].value.columns)


def test_forecast_navigation_hidden_without_exported_values(exports_dir_messy, monkeypatch):
    at = _run({"WQD_EXPORTS_DIR": str(exports_dir_messy)}, monkeypatch)
    assert not at.exception
    assert "Forecast" not in list(at.radio(key="nav").options)
    joined = " ".join(m.value for m in at.markdown)
    assert "Forecast not in current export contract" not in joined


def test_forecast_navigation_enabled_with_valid_export(tmp_path, monkeypatch):
    import json

    exports = tmp_path / "exports"
    exports.mkdir()
    record = {
        "timestamp": "2026-07-21T12:00:00+00:00",
        "parameter_name": "EC",
        "predicted_value": 100.0,
        "actual_value": 101.0,
        "is_anomaly": False,
        "forecast": {"predictions": [102.0, 103.0], "step_hours": 1},
    }
    (exports / "EC.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    at = _run({"WQD_EXPORTS_DIR": str(exports)}, monkeypatch)
    assert "Forecast" in list(at.radio(key="nav").options)
    at.radio(key="nav").set_value("Forecast").run(timeout=60)
    assert not at.exception
    joined = " ".join(m.value for m in at.markdown)
    assert "Short-term forecast" in joined
    assert "Forecast not in current export contract" not in joined


def test_wqi_only_appears_on_methodology(monkeypatch):
    at = _run(None, monkeypatch)
    overview = " ".join(m.value for m in at.markdown)
    assert "Water Quality Index" not in overview
    assert "WQI" not in overview

    at.radio(key="nav").set_value("Data & methodology").run(timeout=60)
    methodology = " ".join(m.value for m in at.markdown)
    assert "Water Quality Index (WQI)" in methodology
    assert "not live and not a water-safety verdict" in methodology


def test_historical_header_has_no_delayed_alert(monkeypatch):
    at = _run(None, monkeypatch)
    joined = " ".join(m.value for m in at.markdown)
    assert "Historical dataset" in joined
    assert "Delayed" not in joined


def test_null_optional_model_fields_are_hidden(monkeypatch):
    """The real status export has null version/update and no R² or drift field."""
    at = _run(None, monkeypatch)
    at.radio(key="nav").set_value("Model monitoring").run(timeout=60)
    joined = " ".join(m.value for m in at.markdown)
    assert "Model monitoring active" in joined
    assert "Model version" not in joined
    assert "Last model update" not in joined
    assert "R²" not in joined
    assert "Drift status" not in joined
