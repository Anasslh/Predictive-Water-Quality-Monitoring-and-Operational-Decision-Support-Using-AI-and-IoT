"""Record cleaning tests: dedupe, retention, sort."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard.models.schemas import MeasurementRecord
from dashboard.services.validation import (
    clean_records,
    dedupe,
    sort_by_time,
    within_retention,
)


def _rec(ts: datetime | None, pred=1.0, actual=1.0) -> MeasurementRecord:
    return MeasurementRecord(
        parameter_name="EC", timestamp=ts, predicted_value=pred, actual_value=actual,
    )


def test_dedupe_removes_identical():
    t = datetime(2026, 7, 20, tzinfo=timezone.utc)
    recs = [_rec(t), _rec(t), _rec(t + timedelta(days=1))]
    out = dedupe(recs)
    assert len(out) == 2


def test_within_retention_filters_old_keeps_undated():
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    recent = _rec(now - timedelta(days=5))
    old = _rec(now - timedelta(days=90))
    undated = _rec(None)
    out = within_retention([recent, old, undated], retention_days=30, now=now)
    assert recent in out
    assert old not in out
    assert undated in out          # undated records are retained


def test_sort_by_time_undated_last():
    t1 = datetime(2026, 7, 20, tzinfo=timezone.utc)
    t2 = datetime(2026, 7, 22, tzinfo=timezone.utc)
    undated = _rec(None)
    out = sort_by_time([_rec(t2), undated, _rec(t1)])
    assert out[0].timestamp == t1
    assert out[1].timestamp == t2
    assert out[2].timestamp is None


def test_clean_records_pipeline():
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    t_recent = now - timedelta(days=2)
    recs = [
        _rec(t_recent),
        _rec(t_recent),                       # duplicate
        _rec(now - timedelta(days=120)),      # out of window
    ]
    out = clean_records(recs, retention_days=30, now=now)
    assert len(out) == 1
    assert out[0].timestamp == t_recent
