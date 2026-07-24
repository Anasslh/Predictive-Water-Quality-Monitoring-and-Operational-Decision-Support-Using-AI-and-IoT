"""Shared pytest fixtures: synthetic exports and helper builders."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_jsonl_lines(base_time) -> list[str]:
    """Valid, varied JSONL lines: prediction-only, normal, anomaly."""
    recs = [
        {
            "timestamp": _iso(base_time),
            "parameter_name": "EC",
            "predicted_value": 805.06,
            "actual_value": None,
            "shap_top_features": [
                {"feature": "EC_roll3_mean", "shap_value": 27.77, "direction": "positive"},
                {"feature": "EC_lag1", "shap_value": -3.78, "direction": "negative"},
            ],
            "is_anomaly": None,
            "anomaly_score": None,
            "retrain_alert": None,
        },
        {
            "timestamp": _iso(base_time + timedelta(days=1)),
            "parameter_name": "EC",
            "predicted_value": 186.17,
            "actual_value": 188.17,
            "shap_top_features": [],
            "is_anomaly": False,
            "anomaly_score": 0.18,
            "retrain_alert": None,
        },
        {
            "timestamp": _iso(base_time + timedelta(days=2)),
            "parameter_name": "EC",
            "predicted_value": 186.17,
            "actual_value": 769.0,
            "shap_top_features": [
                {"feature": "EC_roll3_mean", "shap_value": -5.59, "direction": "negative"},
            ],
            "is_anomaly": True,
            "anomaly_score": 1.0,
            "retrain_alert": "Drift detected — candidate queued",
        },
    ]
    return [json.dumps(r) for r in recs]


@pytest.fixture
def sample_status() -> dict:
    return {
        "parameter_name": "EC",
        "unit": "µS/cm",
        "model_version": "xgb_ec_v1_final",
        "last_promoted_at": "2026-07-19T13:10:16+00:00",
        "last_measurement_at": "2026-07-22T12:00:00+00:00",
        "performance_30d": {
            "skill_vs_persistence_pct": 26.72,
            "rmse": 425.64,
            "mae": 311.77,
            "n_measurements": 16,
        },
        "pending_approvals": 0,
        "consecutive_rejections": 0,
        "queried_at": "2026-07-22T12:00:05+00:00",
    }


@pytest.fixture
def exports_dir(tmp_path, sample_jsonl_lines, sample_status) -> Path:
    """A well-formed exports directory for EC (valid data)."""
    d = tmp_path / "exports"
    d.mkdir()
    (d / "EC.jsonl").write_text("\n".join(sample_jsonl_lines) + "\n", encoding="utf-8")
    (d / "EC_status.json").write_text(json.dumps(sample_status), encoding="utf-8")
    return d


@pytest.fixture
def exports_dir_messy(tmp_path, sample_jsonl_lines) -> Path:
    """
    An exports directory exercising the robustness paths:
      • pH.jsonl with one corrupt line and no status file
      • a stray .tmp file and a subdirectory that must be ignored
      • no anomalies, no forecast for pH
    """
    d = tmp_path / "exports"
    d.mkdir()
    # EC valid
    (d / "EC.jsonl").write_text("\n".join(sample_jsonl_lines) + "\n", encoding="utf-8")
    # pH: one valid line + one corrupt line, no status file, no anomalies
    good = json.dumps({
        "timestamp": "2026-07-21T12:00:00+00:00",
        "parameter_name": "pH",
        "predicted_value": 7.1,
        "actual_value": 7.0,
        "shap_top_features": [],
        "is_anomaly": False,
        "anomaly_score": 0.02,
        "retrain_alert": None,
    })
    (d / "pH.jsonl").write_text(good + "\n{ this is not valid json }\n", encoding="utf-8")
    # ignored noise
    (d / "EC.jsonl.tmp").write_text("garbage", encoding="utf-8")
    sub = d / "_canary_disabled"
    sub.mkdir()
    (sub / "EC.jsonl").write_text("{}\n", encoding="utf-8")
    return d
