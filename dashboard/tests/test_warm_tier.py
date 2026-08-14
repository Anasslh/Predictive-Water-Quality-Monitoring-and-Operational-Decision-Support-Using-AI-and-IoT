"""Unit coverage for SQL Server Warm-tier configuration and read service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard.config.settings import Settings
from dashboard.services import db_service


_COLUMNS = [
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
]


class _Cursor:
    def __init__(self, rows, *, fail: Exception | None = None):
        self.rows = rows
        self.fail = fail
        self.description = [(name,) for name in _COLUMNS]
        self.timeout = 0
        self.executed = []
        self.closed = False

    def execute(self, query, *params):
        self.executed.append((query, params))
        if self.fail:
            raise self.fail
        return self

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def _factory(connection):
    def connect(connection_string, **kwargs):
        assert connection_string == "configured-secret"
        assert kwargs["autocommit"] is True
        return connection

    return connect


def test_warm_configuration_is_environment_driven_and_secret_repr_hidden(monkeypatch):
    monkeypatch.setenv("DB_CONN_STR", "Password=do-not-render")
    monkeypatch.setenv("WQD_WARM_SITE_ID", "Research-Site-2")
    monkeypatch.setenv("WQD_WARM_LOOKBACK_DAYS", "99999")
    monkeypatch.setenv("WQD_DB_TIMEOUT_SECONDS", "0")
    settings = Settings().normalised()
    assert settings.warm_configured is True
    assert settings.effective_warm_site_id == "Research-Site-2"
    assert settings.warm_lookback_days == 3660
    assert settings.db_timeout_seconds == 1
    assert "do-not-render" not in repr(settings)


def test_schema_is_versioned_generic_and_indexed():
    schema = (
        Path(__file__).resolve().parents[2]
        / "database"
        / "warm_tier"
        / "001_initial_schema.sql"
    ).read_text(encoding="utf-8")
    for token in (
        "PRIMARY KEY",
        "SiteId nvarchar(128)",
        "ParameterName nvarchar(128)",
        "MeasurementTimestamp datetimeoffset(7)",
        "SourceRecordHash binary(32)",
        "IX_Fact_WaterQuality_SiteParameterTime",
        "IX_Fact_WaterQuality_Anomalies",
    ):
        assert token in schema
    assert "ParameterName = N'EC'" not in schema
    assert "ParameterName = N'pH'" not in schema


def test_bounded_query_is_generic_sorted_and_preserves_nulls_and_anomalies():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        (2, start + timedelta(days=2), "NewSensor", "mg/L", 3.0, None, "[]", None, None, None, None),
        (1, start + timedelta(days=1), "NewSensor", "mg/L", 2.0, 2.5, "[]", True, 0.9, None, None),
        (3, start + timedelta(days=1), "pH", "pH", 7.1, 7.0, "[]", False, 0.1, None, None),
    ]
    cursor = _Cursor(rows)
    connection = _Connection(cursor)
    result = db_service.fetch_historical_parameters(
        "configured-secret",
        "C-1",
        start,
        start + timedelta(days=10),
        connection_factory=_factory(connection),
    )

    assert list(result.params) == ["NewSensor", "pH"]
    sensor = result.params["NewSensor"]
    assert [r.timestamp for r in sensor.records] == sorted(r.timestamp for r in sensor.records)
    assert sensor.records[-1].actual_value is None
    assert sensor.records[-1].is_anomaly is None
    assert sensor.records[0].is_anomaly is True
    assert sensor.status is None
    assert sensor.unit == "mg/L"
    query, params = cursor.executed[0]
    assert query.lstrip().startswith("SELECT")
    assert "INSERT" not in query.upper()
    assert "UPDATE" not in query.upper()
    assert "SiteId = ?" in query
    assert "MeasurementTimestamp >= ?" in query
    assert params[0] == "C-1"
    assert cursor.closed and connection.closed


def test_duplicate_timestamp_keeps_last_whole_database_row():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        (1, ts, "EC", "µS/cm", 10.0, 11.0, "[]", False, 0.1, None, None),
        (2, ts, "EC", "µS/cm", 20.0, None, "[]", None, None, None, None),
    ]
    result = db_service.fetch_historical_parameters(
        "configured-secret",
        "C-1",
        ts - timedelta(days=1),
        ts + timedelta(days=1),
        connection_factory=_factory(_Connection(_Cursor(rows))),
    )
    record = result.params["EC"].records[0]
    assert record.predicted_value == 20.0
    assert record.actual_value is None
    assert record.is_anomaly is None


def test_empty_warm_database_returns_empty_mapping():
    now = datetime.now(timezone.utc)
    result = db_service.fetch_historical_parameters(
        "configured-secret",
        "C-1",
        now - timedelta(days=1),
        now,
        connection_factory=_factory(_Connection(_Cursor([]))),
    )
    assert result.params == {}
    assert result.metrics.row_count == 0


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Login failed for user", "authentication_failed"),
        ("Invalid object name dbo.Fact_WaterQuality", "schema_missing"),
        ("HYT00 query timeout", "timeout"),
        ("Cannot open database requested by login. Login failed.", "database_unavailable"),
        ("08001 server does not exist", "database_unavailable"),
        ("unexpected driver failure", "query_failed"),
    ],
)
def test_connection_failures_are_safe_categories(message, code):
    now = datetime.now(timezone.utc)

    def fail(*args, **kwargs):
        raise RuntimeError(message)

    with pytest.raises(db_service.WarmTierError) as caught:
        db_service.fetch_historical_parameters(
            "configured-secret",
            "C-1",
            now - timedelta(days=1),
            now,
            connection_factory=fail,
        )
    assert caught.value.code == code
    assert "configured-secret" not in str(caught.value)


def test_missing_connection_is_actionable_without_driver_call():
    now = datetime.now(timezone.utc)
    with pytest.raises(db_service.WarmTierError) as caught:
        db_service.fetch_historical_parameters("", "C-1", now - timedelta(days=1), now)
    assert caught.value.code == "configuration_missing"
