"""Read-only SQL Server access for the dashboard's Warm historical tier.

This module deliberately contains no Streamlit calls and never writes to the
database.  It returns the same typed ``ParameterData`` objects used by the Hot
JSONL loader, allowing every dashboard view to operate on one consistent data
source.  Connection failures are reduced to stable error codes so credentials
and driver details are never rendered in the operator UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from time import perf_counter
from typing import Any, Callable

from dashboard.models.schemas import MeasurementRecord, ParameterData
from dashboard.services.validation import clean_records

try:  # Hot mode must still import and run when the optional driver is absent.
    import pyodbc  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised through dependency check
    pyodbc = None  # type: ignore[assignment]

logger = logging.getLogger("dashboard.warm_tier")

ConnectionFactory = Callable[..., Any]


class WarmTierError(RuntimeError):
    """Safe, localized-at-the-view-boundary Warm-tier failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class WarmQueryMetrics:
    """Approximate timings for one bounded Warm query."""

    connection_ms: float = 0.0
    query_ms: float = 0.0
    transform_ms: float = 0.0
    row_count: int = 0


@dataclass(frozen=True)
class WarmQueryResult:
    """Typed data and non-sensitive performance metadata."""

    params: dict[str, ParameterData] = field(default_factory=dict)
    metrics: WarmQueryMetrics = field(default_factory=WarmQueryMetrics)


_QUERY = """
SELECT
    MeasurementId,
    CONVERT(nvarchar(50), MeasurementTimestamp, 127) AS MeasurementTimestamp,
    ParameterName,
    Unit,
    PredictedValue,
    ActualValue,
    ShapTopFeatures,
    IsAnomaly,
    AnomalyScore,
    RetrainAlert,
    ForecastJson
FROM dbo.Fact_WaterQuality
WHERE SiteId = ?
  AND MeasurementTimestamp >= ?
  AND MeasurementTimestamp < ?
ORDER BY ParameterName ASC, MeasurementTimestamp ASC, MeasurementId ASC;
"""


def driver_available() -> bool:
    """Return whether the optional SQL Server Python driver can be imported."""
    return pyodbc is not None


def fetch_historical_parameters(
    connection_string: str,
    site_id: str,
    start_utc: datetime,
    end_utc_exclusive: datetime,
    *,
    timeout_seconds: int = 5,
    connection_factory: ConnectionFactory | None = None,
) -> WarmQueryResult:
    """Load all parameters for one site and an explicit UTC date range.

    The SQL query is bounded, parameter-generic and read-only. Rows are cleaned
    with the same deterministic whole-record duplicate policy as Hot exports.
    Warm records intentionally carry no current status snapshot: model metrics
    and pending approvals are current-only until a status-history contract is
    explicitly implemented.
    """
    if not connection_string.strip():
        raise WarmTierError("configuration_missing")
    if not site_id.strip():
        raise WarmTierError("site_missing")
    if start_utc >= end_utc_exclusive:
        raise WarmTierError("invalid_range")
    if connection_factory is None:
        if pyodbc is None:
            raise WarmTierError("dependency_missing")
        connection_factory = pyodbc.connect

    start_utc = _as_utc(start_utc)
    end_utc_exclusive = _as_utc(end_utc_exclusive)
    timeout_seconds = max(1, min(int(timeout_seconds), 60))

    connection = None
    cursor = None
    connected_at = perf_counter()
    try:
        connection = connection_factory(
            connection_string,
            timeout=timeout_seconds,
            autocommit=True,
        )
        connection_ms = (perf_counter() - connected_at) * 1000

        query_started = perf_counter()
        cursor = connection.cursor()
        if hasattr(cursor, "timeout"):
            cursor.timeout = timeout_seconds
        cursor.execute(_QUERY, site_id.strip(), start_utc, end_utc_exclusive)
        columns = [str(col[0]) for col in cursor.description]
        raw_rows = cursor.fetchall()
        query_ms = (perf_counter() - query_started) * 1000

        transform_started = perf_counter()
        params = _rows_to_parameters(columns, raw_rows)
        transform_ms = (perf_counter() - transform_started) * 1000
        return WarmQueryResult(
            params=params,
            metrics=WarmQueryMetrics(
                connection_ms=connection_ms,
                query_ms=query_ms,
                transform_ms=transform_ms,
                row_count=len(raw_rows),
            ),
        )
    except WarmTierError:
        raise
    except Exception as exc:
        code = _classify_database_error(exc)
        # Never log the exception text: ODBC messages can include server,
        # database or login identifiers. The exception class is sufficient for
        # engineering logs while the UI renders a localized safe message.
        logger.warning("Warm-tier query failed [%s, %s]", code, type(exc).__name__)
        raise WarmTierError(code) from None
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _rows_to_parameters(
    columns: list[str], raw_rows: list[Any]
) -> dict[str, ParameterData]:
    """Convert DB-API rows to existing dashboard models without substitution."""
    grouped: dict[str, list[MeasurementRecord]] = {}
    units: dict[str, str] = {}
    errors: dict[str, int] = {}

    for raw in raw_rows:
        row = dict(zip(columns, raw))
        name = str(row.get("ParameterName") or "").strip()
        if not name:
            continue
        try:
            shap = _json_value(row.get("ShapTopFeatures"), default=[])
            forecast = _json_value(row.get("ForecastJson"), default=None)
            record = MeasurementRecord.from_dict(
                {
                    "timestamp": row.get("MeasurementTimestamp"),
                    "parameter_name": name,
                    "predicted_value": row.get("PredictedValue"),
                    "actual_value": row.get("ActualValue"),
                    "shap_top_features": shap,
                    "is_anomaly": row.get("IsAnomaly"),
                    "anomaly_score": row.get("AnomalyScore"),
                    "retrain_alert": row.get("RetrainAlert"),
                    "forecast": forecast,
                }
            )
            if record.timestamp is None:
                raise ValueError("invalid timestamp")
            grouped.setdefault(name, []).append(record)
            unit = str(row.get("Unit") or "").strip()
            if unit:
                units[name] = unit
        except (TypeError, ValueError, json.JSONDecodeError):
            errors[name] = errors.get(name, 0) + 1

    result: dict[str, ParameterData] = {}
    for name in sorted(grouped, key=str.casefold):
        result[name] = ParameterData(
            name=name,
            unit=units.get(name, ""),
            records=clean_records(grouped[name], retention_days=0),
            status=None,
            load_errors=errors.get(name, 0),
        )
    return result


def _json_value(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    return json.loads(str(value))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _classify_database_error(exc: Exception) -> str:
    """Map driver failures to safe operator-facing categories."""
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text or "hyt00" in text:
        return "timeout"
    if "invalid object name" in text or "42s02" in text:
        return "schema_missing"
    if (
        "cannot open database" in text
        or "server does not exist" in text
        or "server is unavailable" in text
        or "08001" in text
        or "08004" in text
    ):
        return "database_unavailable"
    if "login failed" in text or "28000" in text or "authentication" in text:
        return "authentication_failed"
    return "query_failed"
