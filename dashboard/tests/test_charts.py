"""Chart builder tests (headless — no Streamlit runtime needed)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go

from dashboard.components import charts
from dashboard.i18n.translator import Translator
from dashboard.models.schemas import ForecastBlock, MeasurementRecord, ParameterData
from dashboard.services.loaders import load_parameter
from dashboard.services.wqi import WqiDiagnostic

TR = Translator("en")
TR_AR = Translator("ar")


def test_actual_vs_predicted_returns_figure(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    fig = charts.actual_vs_predicted(pdata, TR)
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert TR.t("measured_series") in names
    assert TR.t("predicted_series") in names
    assert TR.t("anomaly") in names           # one anomaly present in fixture
    predicted = next(t for t in fig.data if t.name == TR.t("predicted_series"))
    assert predicted.line.dash == "dash"
    assert not fig.layout.shapes               # no operational WQI reference line


def test_actual_vs_predicted_rtl(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    fig = charts.actual_vs_predicted(pdata, TR_AR)
    assert isinstance(fig, go.Figure)


def test_actual_vs_predicted_empty():
    pdata = ParameterData(name="EC", unit="", records=[])
    fig = charts.actual_vs_predicted(pdata, TR)
    assert isinstance(fig, go.Figure)          # empty but valid


def test_chart_sorts_records_preserves_consecutive_null_gaps_and_never_substitutes():
    t = datetime(2026, 7, 20, tzinfo=timezone.utc)
    records = [
        MeasurementRecord("EC", t + timedelta(hours=3), 41.0, 40.0, is_anomaly=False),
        MeasurementRecord("EC", t, 11.0, 10.0, is_anomaly=False),
        MeasurementRecord("EC", t + timedelta(hours=2), 31.0, None, is_anomaly=None),
        MeasurementRecord("EC", t + timedelta(hours=1), 21.0, None, is_anomaly=None),
    ]
    fig = charts.actual_vs_predicted(ParameterData("EC", "µS/cm", records), TR)
    measured = next(t for t in fig.data if t.name == "Measured")
    predicted = next(t for t in fig.data if t.name == "Predicted")

    assert list(measured.x) == sorted(measured.x)
    assert list(measured.y) == [10.0, None, None, 40.0]
    assert list(predicted.y) == [11.0, 21.0, 31.0, 41.0]
    assert measured.connectgaps is False
    assert measured.hovertemplate.count("Measured") >= 1
    assert "Predicted" in measured.hovertemplate
    assert "Anomaly state" in measured.hovertemplate


def test_chart_duplicate_timestamp_keeps_last_record_without_substitution():
    t = datetime(2026, 7, 20, tzinfo=timezone.utc)
    records = [
        MeasurementRecord("EC", t, 9.0, 10.0, is_anomaly=False),
        MeasurementRecord("EC", t, 20.0, None, is_anomaly=None),
    ]
    fig = charts.actual_vs_predicted(ParameterData("EC", "µS/cm", records), TR)
    measured = next(trace for trace in fig.data if trace.name == "Measured")
    predicted = next(trace for trace in fig.data if trace.name == "Predicted")
    assert list(measured.y) == [None]
    assert list(predicted.y) == [20.0]


def test_anomaly_marker_uses_actual_measured_position():
    t = datetime(2026, 7, 20, tzinfo=timezone.utc)
    record = MeasurementRecord("EC", t, 100.0, 769.0, is_anomaly=True)
    fig = charts.actual_vs_predicted(ParameterData("EC", "µS/cm", [record]), TR)
    anomaly = next(trace for trace in fig.data if trace.name == "Anomaly")
    assert list(anomaly.y) == [769.0]
    assert anomaly.marker.symbol == "diamond"


def test_forecast_chart_none_without_forecast(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    assert charts.forecast_chart(pdata.records, TR) is None


def test_forecast_chart_renders_when_present():
    rec = MeasurementRecord(
        parameter_name="EC", timestamp=None, predicted_value=1.0, actual_value=None,
        forecast=ForecastBlock(predictions=[1.0, 2.0, 3.0], lower=[0.5, 1.0, 1.5], upper=[1.5, 3.0, 4.5]),
    )
    fig = charts.forecast_chart([rec], TR, unit="µS/cm")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_wqi_class_distribution():
    diag = WqiDiagnostic(n_rows=365, class_counts={"Excellent": 43, "Unsuitable": 274})
    fig = charts.wqi_class_distribution(diag, TR)
    assert isinstance(fig, go.Figure)
