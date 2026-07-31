"""
transforms.py — Derive operational view-model values from cleaned data.

Pure functions (no I/O, no Streamlit) that turn ParameterData into the small
derived quantities the UI renders: latest state, trend, source-aware freshness,
anomaly lists, forecast availability and neutral model status. Deterministic
and unit-testable.

Freshness note
--------------
Freshness compares the age of the newest record against the series' own median
inter-arrival gap (× a configurable multiplier). This is a UI display heuristic
for "is data still arriving on cadence", NOT a scientific or safety threshold —
documented in dashboard/ASSUMPTIONS.md.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from dashboard.models.schemas import MeasurementRecord, ParameterData


# ── Trend ────────────────────────────────────────────────────────────────────

def actual_series(pdata: ParameterData) -> list[MeasurementRecord]:
    """Records that carry a measured value, chronological."""
    return [r for r in pdata.records if r.actual_value is not None]


def trend_direction(pdata: ParameterData) -> tuple[str, float | None]:
    """
    Return (direction, delta) of the latest measured value vs the previous one.

    direction ∈ {"up", "down", "flat", "unknown"}; delta is the signed change
    (None when fewer than two measured values exist).
    """
    series = actual_series(pdata)
    if len(series) < 2:
        return "unknown", None
    latest = series[-1].actual_value
    prev = series[-2].actual_value
    if latest is None or prev is None:
        return "unknown", None
    delta = latest - prev
    if delta > 0:
        return "up", delta
    if delta < 0:
        return "down", delta
    return "flat", 0.0


# ── Freshness ────────────────────────────────────────────────────────────────

def median_gap_seconds(pdata: ParameterData) -> float | None:
    """Median inter-arrival gap (seconds) across dated records, or None."""
    dated = [r.timestamp for r in pdata.records if r.timestamp is not None]
    if len(dated) < 2:
        return None
    gaps = [
        (dated[i] - dated[i - 1]).total_seconds()
        for i in range(1, len(dated))
        if (dated[i] - dated[i - 1]).total_seconds() > 0
    ]
    if not gaps:
        return None
    return statistics.median(gaps)


@dataclass(frozen=True)
class Freshness:
    state: str                    # "fresh" | "stale" | "historical" | "unknown"
    last_timestamp: datetime | None
    age_seconds: float | None


def freshness(
    pdata: ParameterData,
    now: datetime | None = None,
    multiplier: float = 3.0,
    source_mode: str = "continuous",
) -> Freshness:
    """Classify source freshness, disabling alert semantics for historical data."""
    now = now or datetime.now(timezone.utc)
    last = None
    for rec in reversed(pdata.records):
        if rec.timestamp is not None:
            last = rec.timestamp
            break
    if last is None:
        return Freshness("unknown", None, None)

    age = (now - last).total_seconds()
    if source_mode == "historical":
        return Freshness("historical", last, age)
    gap = median_gap_seconds(pdata)
    if gap is None:
        # Only one dated record: we cannot infer a cadence → don't guess "stale".
        return Freshness("unknown", last, age)
    state = "stale" if age > gap * multiplier else "fresh"
    return Freshness(state, last, age)


def system_freshness(
    params: dict[str, ParameterData],
    multiplier: float = 3.0,
    source_mode: str = "continuous",
) -> str:
    """Worst-case freshness across all parameters: stale > unknown > fresh."""
    if source_mode == "historical":
        return "historical"
    states = {
        freshness(p, multiplier=multiplier, source_mode=source_mode).state
        for p in params.values()
    }
    if "stale" in states:
        return "stale"
    if states == {"fresh"}:
        return "fresh"
    if "fresh" in states:
        return "fresh"
    return "unknown"


# ── Anomalies ────────────────────────────────────────────────────────────────

def anomaly_records(pdata: ParameterData) -> list[MeasurementRecord]:
    """Records flagged as anomalies, newest first."""
    flagged = [r for r in pdata.records if r.is_anomaly is True]
    flagged.sort(
        key=lambda r: r.timestamp or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return flagged


def latest_anomaly_state(pdata: ParameterData) -> str:
    """
    Anomaly token for the most recent *evaluated* record.

    "anomaly" | "normal" | "unknown" (unknown when no record carried an
    anomaly evaluation — e.g. prediction-only rows with no actual value).
    """
    for rec in reversed(pdata.records):
        if rec.is_anomaly is not None:
            return "anomaly" if rec.is_anomaly else "normal"
    return "unknown"


def count_active_anomalies(params: dict[str, ParameterData]) -> int:
    """Parameters whose most recent evaluated record is an anomaly."""
    return sum(1 for p in params.values() if latest_anomaly_state(p) == "anomaly")


def has_forecast_data(params: dict[str, ParameterData]) -> bool:
    """True only when at least one exported record contains valid forecast values."""
    return any(
        rec.forecast is not None and bool(rec.forecast.predictions)
        for pdata in params.values()
        for rec in pdata.records
    )


# ── Health token ─────────────────────────────────────────────────────────────

def model_status(pdata: ParameterData) -> str:
    """
    Neutral operational token for the model layer: "active" | "unavailable".

    This is deliberately NOT a health / pass-fail verdict. No approved
    model-health acceptance thresholds exist yet (see dashboard/ASSUMPTIONS.md),
    so the dashboard never claims a model is "healthy" or has "passed". It only
    states whether monitoring is active (a status snapshot exists) or not. The
    raw performance figures (RMSE, MAE, skill) and the pending-review counts are
    presented separately, as facts, on the Model health page.
    """
    return "unavailable" if pdata.status is None else "active"


def count_pending_reviews(params: dict[str, ParameterData]) -> int:
    """Total pending retraining approvals across all parameters."""
    return sum(
        (p.status.pending_approvals if p.status else 0) for p in params.values()
    )
