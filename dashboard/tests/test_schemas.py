"""Schema parsing / coercion tests."""

from __future__ import annotations

from datetime import timezone

from dashboard.models.schemas import (
    ForecastBlock,
    MeasurementRecord,
    Performance30d,
    StatusSnapshot,
    as_bool,
    as_float,
    parse_shap_list,
    parse_timestamp,
)


def test_as_float():
    assert as_float("3.5") == 3.5
    assert as_float(4) == 4.0
    assert as_float(None) is None
    assert as_float(True) is None          # bool is not a number here
    assert as_float("not a number") is None
    assert as_float(float("nan")) is None
    assert as_float(float("inf")) is None


def test_as_bool():
    assert as_bool(True) is True
    assert as_bool("true") is True
    assert as_bool("YES") is True
    assert as_bool(1) is True
    assert as_bool(0) is False
    assert as_bool("false") is False
    assert as_bool(None) is None
    assert as_bool("maybe") is None


def test_parse_timestamp():
    aware = parse_timestamp("2026-07-21T21:52:06.916329+00:00")
    assert aware is not None and aware.tzinfo is not None
    naive = parse_timestamp("2026-07-21T10:00:00")
    assert naive is not None and naive.tzinfo == timezone.utc  # naive assumed UTC
    assert parse_timestamp("") is None
    assert parse_timestamp(None) is None
    assert parse_timestamp("not-a-date") is None


def test_parse_shap_list():
    raw = [
        {"feature": "a", "shap_value": 1.2, "direction": "positive"},
        {"feature": "b"},                       # missing optional fields ok
        {"shap_value": 3.0},                    # missing feature → dropped
        "garbage",                              # non-dict → dropped
    ]
    out = parse_shap_list(raw)
    assert [f.feature for f in out] == ["a", "b"]
    assert out[0].shap_value == 1.2
    assert out[1].direction is None
    assert parse_shap_list("not a list") == []


def test_measurement_record_from_dict_full():
    rec = MeasurementRecord.from_dict({
        "timestamp": "2026-07-21T12:00:00+00:00",
        "parameter_name": "EC",
        "predicted_value": 100.0,
        "actual_value": 110.0,
        "shap_top_features": [{"feature": "x", "shap_value": 1.0, "direction": "positive"}],
        "is_anomaly": True,
        "anomaly_score": 0.9,
        "retrain_alert": "queued",
    })
    assert rec.parameter_name == "EC"
    assert rec.residual == 10.0
    assert rec.is_anomaly is True
    assert rec.forecast is None                # not in contract


def test_measurement_record_missing_optionals():
    rec = MeasurementRecord.from_dict({"parameter_name": "EC", "predicted_value": 5.0})
    assert rec.actual_value is None
    assert rec.residual is None
    assert rec.shap_top_features == []
    assert rec.is_anomaly is None
    assert rec.retrain_alert is None


def test_forecast_block():
    assert ForecastBlock.from_dict(None) is None
    assert ForecastBlock.from_dict({"predictions": []}) is None
    fb = ForecastBlock.from_dict({"predictions": [1.0, 2.0], "lower": [0.5, 1.0], "upper": [1.5, 3.0]})
    assert fb is not None and fb.predictions == [1.0, 2.0]


def test_status_snapshot_from_dict(sample_status):
    st = StatusSnapshot.from_dict(sample_status)
    assert st.parameter_name == "EC"
    assert st.unit == "µS/cm"
    assert st.model_version == "xgb_ec_v1_final"
    assert st.performance.rmse == 425.64
    assert st.performance.n_measurements == 16
    assert st.performance.r2 is None           # not exported
    assert st.pending_approvals == 0


def test_status_snapshot_bad_ints_and_insufficient():
    st = StatusSnapshot.from_dict({
        "parameter_name": "EC",
        "pending_approvals": "oops",
        "performance_30d": {"insufficient_data": True, "n_measurements": None},
    })
    assert st.pending_approvals == 0
    assert st.performance.insufficient_data is True
    assert st.performance.n_measurements == 0


def test_performance_from_non_dict():
    p = Performance30d.from_dict(None)
    assert p.n_measurements == 0 and p.rmse is None
