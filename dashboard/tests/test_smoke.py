"""
End-to-end smoke tests using Streamlit's AppTest harness.

Runs the real app.py headlessly against several export scenarios and asserts the
script completes without raising — covering the robustness requirements:
missing exports, corrupt JSON lines, a parameter with no anomalies/forecast,
every navigation view, and the Arabic (RTL) language.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

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


def test_skill_kpi_shows_localized_insufficient_data(tmp_path, monkeypatch):
    exports = tmp_path / "exports"
    exports.mkdir()
    record = {
        "timestamp": "2026-08-01T00:00:00+00:00",
        "parameter_name": "EC",
        "predicted_value": 100.0,
        "actual_value": 101.0,
        "is_anomaly": False,
    }
    status = {
        "parameter_name": "EC",
        "unit": "µS/cm",
        "performance_30d": {
            "skill_vs_persistence_pct": None,
            "rmse": None,
            "mae": None,
            "n_measurements": 3,
            "insufficient_data": True,
        },
        "pending_approvals": 0,
    }
    (exports / "EC.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (exports / "EC_status.json").write_text(json.dumps(status), encoding="utf-8")

    at = _run({"WQD_EXPORTS_DIR": str(exports)}, monkeypatch)
    at.radio(key="nav").set_value("Model monitoring").run(timeout=60)
    assert "Insufficient data" in " ".join(m.value for m in at.markdown)

    at.radio(key="lang").set_value("ar").run(timeout=60)
    at.radio(key="nav").set_value("مراقبة النموذج").run(timeout=60)
    assert "بيانات غير كافية" in " ".join(m.value for m in at.markdown)


def test_pending_approvals_zero_and_positive_are_read_only(exports_dir, monkeypatch):
    status_path = exports_dir / "EC_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["pending_approvals"] = 2
    status_path.write_text(json.dumps(status), encoding="utf-8")
    before = status_path.read_bytes()

    at = _run({"WQD_EXPORTS_DIR": str(exports_dir)}, monkeypatch)
    joined = " ".join(m.value for m in [*at.markdown, *at.caption])
    assert "Pending retraining reviews" in joined
    assert ">2<" in joined
    assert status_path.read_bytes() == before


class _WarmCursor:
    timeout = 0

    def __init__(self, rows):
        self.rows = rows
        self.description = [
            (name,)
            for name in (
                "MeasurementId",
                "MeasurementTimestamp",
                "ParameterName",
                "Unit",
                "PredictedValue",
                "ActualValue",
                "ShapTopFeatures",
                "IsAnomaly",
                "AnomalyScore",
                "RetrainAlert",
                "ForecastJson",
            )
        ]

    def execute(self, query, *params):
        return self

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class _WarmConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return _WarmCursor(self.rows)

    def close(self):
        pass


def test_historical_source_unconfigured_keeps_current_mode(monkeypatch):
    monkeypatch.delenv("DB_CONN_STR", raising=False)
    at = _run(None, monkeypatch)
    assert not at.exception
    assert list(at.radio(key="data_source").options) == ["Current monitoring"]
    joined = " ".join(m.value for m in [*at.markdown, *at.caption])
    assert "Historical archive is not configured" in joined


def test_historical_source_switch_is_global_generic_and_bilingual(monkeypatch):
    import dashboard.services.db_service as db_service

    ts = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = [
        (1, ts, "NewParameter", "mg/L", 10.0, 11.0, "[]", True, 0.8, None, None),
        (2, ts.replace(day=2), "NewParameter", "mg/L", 12.0, None, "[]", None, None, None, None),
    ]
    monkeypatch.setattr(
        db_service.pyodbc,
        "connect",
        lambda *args, **kwargs: _WarmConnection(rows),
    )
    at = _run(
        {
            "DB_CONN_STR": "warm-smoke-success",
            "WQD_WARM_SITE_ID": "C-1",
            "WQD_DB_TIMEOUT_SECONDS": "1",
        },
        monkeypatch,
    )
    at.radio(key="data_source").set_value("Historical archive").run(timeout=60)
    assert not at.exception
    joined = " ".join(m.value for m in at.markdown)
    assert "NewParameter" in joined
    assert "Historical archive" in joined
    assert "Current model status" in joined

    for label in NAV_LABELS:
        at.radio(key="nav").set_value(label).run(timeout=60)
        assert not at.exception, f"Warm English view '{label}' raised"

    at.radio(key="lang").set_value("ar").run(timeout=60)
    assert not at.exception
    joined_ar = " ".join(m.value for m in at.markdown)
    assert "الأرشيف التاريخي" in joined_ar
    assert "direction: rtl" in joined_ar
    for label in AR_NAV_LABELS:
        at.radio(key="nav").set_value(label).run(timeout=60)
        assert not at.exception, f"Warm Arabic view '{label}' raised"


def test_historical_connection_failure_is_safe_and_hot_remains_selectable(monkeypatch):
    import dashboard.services.db_service as db_service

    def fail(*args, **kwargs):
        raise RuntimeError("Login failed for user; Password=super-secret")

    monkeypatch.setattr(db_service.pyodbc, "connect", fail)
    at = _run({"DB_CONN_STR": "warm-smoke-failure"}, monkeypatch)
    at.radio(key="data_source").set_value("Historical archive").run(timeout=60)
    assert not at.exception
    errors = " ".join(item.value for item in at.error)
    assert "authentication failed" in errors
    assert "super-secret" not in errors

    at.radio(key="data_source").set_value("Current monitoring").run(timeout=60)
    assert not at.exception
    assert "EC" in " ".join(m.value for m in at.markdown)
