"""
validation.py — Pure functions that clean a parsed record list.

Kept separate from I/O (loaders.py) so the cleaning rules can be unit-tested in
isolation. All functions are non-destructive: they return new lists and never
mutate their inputs.

Cleaning pipeline (clean_records):
    1. dedupe    — drop byte-identical duplicate records
    2. retention — keep only records within the rolling window
    3. sort      — stable chronological order, undated records last
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard.models.schemas import MeasurementRecord


def _key(rec: MeasurementRecord) -> tuple:
    """Hashable identity for exact-duplicate detection."""
    shap = tuple(
        (f.feature, f.shap_value, f.direction) for f in rec.shap_top_features
    )
    return (
        rec.timestamp,
        rec.predicted_value,
        rec.actual_value,
        rec.is_anomaly,
        rec.anomaly_score,
        rec.retrain_alert,
        shap,
    )


def dedupe(records: list[MeasurementRecord]) -> list[MeasurementRecord]:
    """Remove exact-duplicate records, preserving first-seen order."""
    seen: set[tuple] = set()
    out: list[MeasurementRecord] = []
    for rec in records:
        k = _key(rec)
        if k in seen:
            continue
        seen.add(k)
        out.append(rec)
    return out


def within_retention(
    records: list[MeasurementRecord],
    retention_days: int,
    now: datetime | None = None,
) -> list[MeasurementRecord]:
    """
    Keep records whose timestamp is within ``retention_days`` of ``now``.

    Records with no parseable timestamp are KEPT (they still carry a value and
    should not silently vanish) — this mirrors the pipeline's own read_export,
    which retains lines whose timestamp cannot be parsed.
    """
    if retention_days <= 0:
        return list(records)
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    return [r for r in records if r.timestamp is None or r.timestamp >= cutoff]


def sort_by_time(records: list[MeasurementRecord]) -> list[MeasurementRecord]:
    """
    Stable chronological sort (oldest first). Undated records are placed last
    while preserving their relative input order.
    """
    dated = [r for r in records if r.timestamp is not None]
    undated = [r for r in records if r.timestamp is None]
    dated.sort(key=lambda r: r.timestamp)  # type: ignore[arg-type,return-value]
    return dated + undated


def clean_records(
    records: list[MeasurementRecord],
    retention_days: int,
    now: datetime | None = None,
) -> list[MeasurementRecord]:
    """Full cleaning pipeline: dedupe → retention → sort."""
    return sort_by_time(within_retention(dedupe(records), retention_days, now))
