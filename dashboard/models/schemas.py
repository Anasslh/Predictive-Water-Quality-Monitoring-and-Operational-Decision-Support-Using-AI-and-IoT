"""
schemas.py — Typed, tolerant representations of the pipeline export contract.

These dataclasses mirror exactly what src/monitor/export.py writes:

  <param>.jsonl        -> one MeasurementRecord per line
  <param>_status.json  -> one StatusSnapshot

Every ``from_dict`` is defensive: unknown/missing/None/badly-typed fields never
raise. A malformed field becomes None (or an empty list) so a single bad record
can be skipped without taking down the view. Parsing helpers are deliberately
permissive because the source data crosses a machine boundary (exports may be
copied/synced from the edge device).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Coercion helpers ─────────────────────────────────────────────────────────

def as_float(value: Any) -> float | None:
    """Best-effort float; None on failure or non-finite."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # Reject NaN / inf so charts and comparisons stay well-defined.
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def as_bool(value: Any) -> bool | None:
    """Coerce common truthy/falsey encodings to bool; None if unknown/missing."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
    return None


def parse_timestamp(value: Any) -> datetime | None:
    """
    Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime.

    Mirrors src/monitor/export.py._parse_ts: naive timestamps are assumed UTC.
    Returns None on any failure so callers can skip the record's time axis
    without crashing.
    """
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── SHAP ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShapFeature:
    """One SHAP contribution attached to a prediction."""

    feature: str
    shap_value: float | None = None
    direction: str | None = None

    @classmethod
    def from_dict(cls, d: Any) -> "ShapFeature | None":
        if not isinstance(d, dict):
            return None
        name = d.get("feature")
        if name in (None, ""):
            return None
        direction = d.get("direction")
        return cls(
            feature=str(name),
            shap_value=as_float(d.get("shap_value")),
            direction=str(direction) if isinstance(direction, str) else None,
        )


def parse_shap_list(raw: Any) -> list[ShapFeature]:
    """Parse the shap_top_features list; always returns a list (possibly empty)."""
    if not isinstance(raw, list):
        return []
    out: list[ShapFeature] = []
    for item in raw:
        f = ShapFeature.from_dict(item)
        if f is not None:
            out.append(f)
    return out


# ── Measurement record ───────────────────────────────────────────────────────

@dataclass
class MeasurementRecord:
    """One line of a <param>.jsonl export."""

    parameter_name: str | None
    timestamp: datetime | None
    predicted_value: float | None
    actual_value: float | None
    shap_top_features: list[ShapFeature] = field(default_factory=list)
    is_anomaly: bool | None = None
    anomaly_score: float | None = None
    retrain_alert: str | None = None
    # Forecast is NOT in the current export contract. This optional field is
    # code-ready: it is populated only if a future export includes a "forecast"
    # object on the record. Never fabricated.
    forecast: "ForecastBlock | None" = None

    @property
    def residual(self) -> float | None:
        """actual − predicted, when both are present."""
        if self.actual_value is None or self.predicted_value is None:
            return None
        return self.actual_value - self.predicted_value

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasurementRecord":
        alert = d.get("retrain_alert")
        return cls(
            parameter_name=(str(d["parameter_name"]) if d.get("parameter_name") else None),
            timestamp=parse_timestamp(d.get("timestamp")),
            predicted_value=as_float(d.get("predicted_value")),
            actual_value=as_float(d.get("actual_value")),
            shap_top_features=parse_shap_list(d.get("shap_top_features")),
            is_anomaly=as_bool(d.get("is_anomaly")),
            anomaly_score=as_float(d.get("anomaly_score")),
            retrain_alert=str(alert) if isinstance(alert, str) and alert.strip() else None,
            forecast=ForecastBlock.from_dict(d.get("forecast")),
        )


# ── Forecast (code-ready, optional) ──────────────────────────────────────────

@dataclass
class ForecastBlock:
    """
    Optional forecast attached to a record. Shape anticipates the pipeline's
    ForecastResult (predictions + uncertainty). Present only if a future export
    includes it; absent today.
    """

    predictions: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    step_hours: float | None = None

    @classmethod
    def from_dict(cls, d: Any) -> "ForecastBlock | None":
        if not isinstance(d, dict):
            return None
        preds = [as_float(x) for x in d.get("predictions", []) if as_float(x) is not None]
        if not preds:
            return None
        return cls(
            predictions=preds,  # type: ignore[arg-type]
            lower=[as_float(x) for x in d.get("lower", []) if as_float(x) is not None],  # type: ignore[misc]
            upper=[as_float(x) for x in d.get("upper", []) if as_float(x) is not None],  # type: ignore[misc]
            step_hours=as_float(d.get("step_hours")),
        )


# ── Status snapshot ──────────────────────────────────────────────────────────

@dataclass
class Performance30d:
    """The performance_30d block of a status.json."""

    skill_vs_persistence_pct: float | None = None
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None                 # not currently exported → stays None
    n_measurements: int = 0
    insufficient_data: bool = False

    @classmethod
    def from_dict(cls, d: Any) -> "Performance30d":
        if not isinstance(d, dict):
            return cls()
        n = d.get("n_measurements")
        try:
            n_int = int(n) if n is not None else 0
        except (TypeError, ValueError):
            n_int = 0
        return cls(
            skill_vs_persistence_pct=as_float(d.get("skill_vs_persistence_pct")),
            rmse=as_float(d.get("rmse")),
            mae=as_float(d.get("mae")),
            r2=as_float(d.get("r2")),
            n_measurements=n_int,
            insufficient_data=bool(d.get("insufficient_data", False)),
        )


@dataclass
class StatusSnapshot:
    """A <param>_status.json file."""

    parameter_name: str | None = None
    unit: str = ""
    model_version: str | None = None
    last_promoted_at: datetime | None = None
    last_measurement_at: datetime | None = None
    performance: Performance30d = field(default_factory=Performance30d)
    pending_approvals: int = 0
    consecutive_rejections: int = 0
    queried_at: datetime | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StatusSnapshot":
        def _int(key: str) -> int:
            try:
                return int(d.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        return cls(
            parameter_name=(str(d["parameter_name"]) if d.get("parameter_name") else None),
            unit=str(d.get("unit") or ""),
            model_version=(str(d["model_version"]) if d.get("model_version") else None),
            last_promoted_at=parse_timestamp(d.get("last_promoted_at")),
            last_measurement_at=parse_timestamp(d.get("last_measurement_at")),
            performance=Performance30d.from_dict(d.get("performance_30d")),
            pending_approvals=_int("pending_approvals"),
            consecutive_rejections=_int("consecutive_rejections"),
            queried_at=parse_timestamp(d.get("queried_at")),
        )


# ── Bundled per-parameter view model ─────────────────────────────────────────

@dataclass
class ParameterData:
    """Everything the UI needs for one parameter, assembled by the loader."""

    name: str
    unit: str
    records: list[MeasurementRecord] = field(default_factory=list)  # oldest → newest
    status: StatusSnapshot | None = None
    load_errors: int = 0          # count of skipped malformed JSONL lines

    # ── Derived convenience accessors ────────────────────────────────────────
    @property
    def has_records(self) -> bool:
        return len(self.records) > 0

    @property
    def latest(self) -> MeasurementRecord | None:
        return self.records[-1] if self.records else None

    @property
    def latest_actual_record(self) -> MeasurementRecord | None:
        """Most recent record that actually carries a measured value."""
        for rec in reversed(self.records):
            if rec.actual_value is not None:
                return rec
        return None

    @property
    def display_unit(self) -> str:
        if self.unit:
            return self.unit
        return self.status.unit if self.status and self.status.unit else ""
