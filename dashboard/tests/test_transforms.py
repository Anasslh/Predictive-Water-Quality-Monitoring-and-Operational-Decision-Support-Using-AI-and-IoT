"""Transform tests: trend, source freshness, anomalies and model availability."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard.models.schemas import (
    MeasurementRecord,
    ParameterData,
    Performance30d,
    StatusSnapshot,
)
from dashboard.services import transforms as tx
from dashboard.services.loaders import load_parameter


def _rec(ts, actual=None, pred=None, is_anom=None):
    return MeasurementRecord(
        parameter_name="EC", timestamp=ts, predicted_value=pred,
        actual_value=actual, is_anomaly=is_anom,
    )


def test_trend_direction_up(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    direction, delta = tx.trend_direction(pdata)
    assert direction == "up"
    assert delta is not None and delta > 0


def test_trend_direction_unknown_with_one_value():
    t = datetime(2026, 7, 20, tzinfo=timezone.utc)
    pdata = ParameterData(name="EC", unit="", records=[_rec(t, actual=5.0)])
    assert tx.trend_direction(pdata) == ("unknown", None)


def test_anomaly_records_and_state(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    anoms = tx.anomaly_records(pdata)
    assert len(anoms) == 1
    assert anoms[0].actual_value == 769.0
    assert tx.latest_anomaly_state(pdata) == "anomaly"


def test_freshness_fresh_vs_stale():
    now = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    recs = [
        _rec(now - timedelta(days=3), actual=1.0),
        _rec(now - timedelta(days=2), actual=2.0),
        _rec(now - timedelta(days=1), actual=3.0),  # median gap = 1 day
    ]
    pdata = ParameterData(name="EC", unit="", records=recs)
    fresh = tx.freshness(pdata, now=now, multiplier=3.0)
    assert fresh.state == "fresh"                    # 1 day old < 3× median gap

    stale_now = now + timedelta(days=10)
    stale = tx.freshness(pdata, now=stale_now, multiplier=3.0)
    assert stale.state == "stale"


def test_freshness_unknown_single_record():
    now = datetime(2026, 7, 20, tzinfo=timezone.utc)
    pdata = ParameterData(name="EC", unit="", records=[_rec(now, actual=1.0)])
    assert tx.freshness(pdata, now=now).state == "unknown"


def test_historical_mode_disables_freshness_alert_semantics(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    far_future = datetime(2036, 7, 20, tzinfo=timezone.utc)
    result = tx.freshness(pdata, now=far_future, source_mode="historical")
    assert result.state == "historical"
    assert tx.system_freshness(
        {"EC": pdata}, source_mode="historical"
    ) == "historical"


def test_model_status_active_does_not_classify_pending_review():
    status = StatusSnapshot(parameter_name="EC", pending_approvals=1)
    pdata = ParameterData(name="EC", unit="", records=[], status=status)
    assert tx.model_status(pdata) == "active"


def test_model_status_active_does_not_classify_negative_skill():
    status = StatusSnapshot(
        parameter_name="EC",
        performance=Performance30d(skill_vs_persistence_pct=-5.0, n_measurements=20),
    )
    pdata = ParameterData(name="EC", unit="", records=[], status=status)
    assert tx.model_status(pdata) == "active"


def test_model_status_unavailable_without_status():
    pdata = ParameterData(name="EC", unit="", records=[])
    assert tx.model_status(pdata) == "unavailable"


def test_count_helpers(exports_dir):
    params = {"EC": load_parameter("EC", exports_dir, retention_days=3650)}
    assert tx.count_active_anomalies(params) == 1
    assert tx.count_pending_reviews(params) == 0


def test_anomaly_count_means_affected_parameters_not_records():
    t = datetime(2026, 7, 20, tzinfo=timezone.utc)
    pdata = ParameterData(
        name="EC",
        unit="",
        records=[
            _rec(t, actual=1.0, pred=0.0, is_anom=True),
            _rec(t + timedelta(hours=1), actual=2.0, pred=0.0, is_anom=True),
        ],
    )
    assert len(tx.anomaly_records(pdata)) == 2
    assert tx.count_active_anomalies({"EC": pdata}) == 1


def test_forecast_availability_requires_valid_predictions():
    t = datetime(2026, 7, 20, tzinfo=timezone.utc)
    absent = ParameterData(name="EC", unit="", records=[_rec(t, pred=1.0)])
    assert tx.has_forecast_data({"EC": absent}) is False

    from dashboard.models.schemas import ForecastBlock

    with_forecast = _rec(t, pred=1.0)
    with_forecast.forecast = ForecastBlock(predictions=[1.1, 1.2])
    present = ParameterData(name="EC", unit="", records=[with_forecast])
    assert tx.has_forecast_data({"EC": present}) is True
