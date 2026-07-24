"""Chart builder tests (headless — no Streamlit runtime needed)."""

from __future__ import annotations

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
    fig = charts.actual_vs_predicted(pdata, TR, water_use_profile="generalist")
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert TR.t("latest_actual") in names
    assert TR.t("latest_predicted") in names
    assert TR.t("anomaly") in names           # one anomaly present in fixture


def test_actual_vs_predicted_rtl(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    fig = charts.actual_vs_predicted(pdata, TR_AR)
    assert isinstance(fig, go.Figure)


def test_actual_vs_predicted_empty():
    pdata = ParameterData(name="EC", unit="", records=[])
    fig = charts.actual_vs_predicted(pdata, TR)
    assert isinstance(fig, go.Figure)          # empty but valid


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
