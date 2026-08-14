"""Validated, idempotent Cold-export to SQL Server Warm-tier ingestion.

The tool is intentionally manual/CLI-driven. It consumes the flat JSONL file
produced by ``python run.py export-archive`` and upserts whole records into the
versioned SQL Server schema. It does not modify Hot exports or the Cold archive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

try:
    import pyodbc  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CLI reports a clear dependency error
    pyodbc = None  # type: ignore[assignment]


_ROOT = Path(__file__).resolve().parent
_DEFAULT_SCHEMA = _ROOT / "database" / "warm_tier" / "001_initial_schema.sql"

ConnectionFactory = Callable[..., Any]


@dataclass(frozen=True)
class ArchiveRow:
    site_id: str
    parameter_name: str
    unit: str | None
    timestamp: datetime
    predicted_value: float | None
    actual_value: float | None
    shap_json: str | None
    is_anomaly: bool | None
    anomaly_score: float | None
    retrain_alert: str | None
    forecast_json: str | None
    model_version: str | None
    source_file: str
    source_hash: bytes

    @property
    def key(self) -> tuple[str, str, datetime]:
        return self.site_id, self.parameter_name, self.timestamp

    def as_db_tuple(self) -> tuple[Any, ...]:
        return (
            self.site_id,
            self.parameter_name,
            self.unit,
            self.timestamp,
            self.predicted_value,
            self.actual_value,
            self.shap_json,
            self.is_anomaly,
            self.anomaly_score,
            self.retrain_alert,
            self.forecast_json,
            self.model_version,
            self.source_file,
            self.source_hash,
        )


@dataclass(frozen=True)
class ParseSummary:
    rows: list[ArchiveRow]
    nonempty_lines: int
    rejected_lines: int
    duplicate_timestamps: int
    rejected_line_numbers: tuple[int, ...]


@dataclass(frozen=True)
class IngestionSummary:
    staged: int
    inserted: int
    updated: int
    unchanged: int


def parse_archive_file(
    path: str | Path,
    *,
    site_id: str,
    unit_override: str | None = None,
    expected_parameter: str | None = None,
) -> ParseSummary:
    """Parse and validate an archive JSONL file.

    Invalid lines are rejected independently. For multiple valid rows sharing
    ``site + parameter + timestamp``, the last whole source record wins; fields
    are never merged. This mirrors the dashboard's deterministic duplicate
    policy and makes one ingestion batch reproducible.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    site_id = site_id.strip()
    if not site_id:
        raise ValueError("site_id is required")

    by_key: dict[tuple[str, str, datetime], ArchiveRow] = {}
    nonempty = rejected = duplicates = 0
    rejected_line_numbers: list[int] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            nonempty += 1
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record is not an object")
                row = _parse_record(
                    payload,
                    site_id=site_id,
                    unit_override=unit_override,
                    expected_parameter=expected_parameter,
                    source_file=source.name,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                rejected += 1
                rejected_line_numbers.append(line_number)
                continue
            if row.key in by_key:
                duplicates += 1
            by_key[row.key] = row

    return ParseSummary(
        rows=list(by_key.values()),
        nonempty_lines=nonempty,
        rejected_lines=rejected,
        duplicate_timestamps=duplicates,
        rejected_line_numbers=tuple(rejected_line_numbers),
    )


def apply_schema(
    connection_string: str,
    *,
    schema_path: str | Path = _DEFAULT_SCHEMA,
    timeout_seconds: int = 15,
    connection_factory: ConnectionFactory | None = None,
) -> None:
    """Apply the idempotent schema script to the configured database."""
    script_path = Path(schema_path).expanduser().resolve()
    script = script_path.read_text(encoding="utf-8")
    batches = [b.strip() for b in re.split(r"(?im)^\s*GO\s*$", script) if b.strip()]
    connection = _connect(
        connection_string,
        timeout_seconds=timeout_seconds,
        connection_factory=connection_factory,
    )
    cursor = None
    try:
        connection.autocommit = False
        cursor = connection.cursor()
        if hasattr(cursor, "timeout"):
            cursor.timeout = max(1, timeout_seconds)
        for batch in batches:
            cursor.execute(batch)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def ingest_rows(
    connection_string: str,
    rows: list[ArchiveRow],
    *,
    timeout_seconds: int = 15,
    connection_factory: ConnectionFactory | None = None,
) -> IngestionSummary:
    """Transactionally upsert validated rows into SQL Server.

    The target key is ``SiteId + ParameterName + MeasurementTimestamp``.
    Re-ingesting an identical record is a no-op. A changed record at the same
    key replaces the complete stored payload; values are never merged field by
    field. A failure rolls back the entire batch.
    """
    if not rows:
        return IngestionSummary(staged=0, inserted=0, updated=0, unchanged=0)

    connection = _connect(
        connection_string,
        timeout_seconds=timeout_seconds,
        connection_factory=connection_factory,
    )
    cursor = None
    try:
        connection.autocommit = False
        cursor = connection.cursor()
        if hasattr(cursor, "timeout"):
            cursor.timeout = max(1, timeout_seconds)
        cursor.execute(
            "SET XACT_ABORT ON; SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"
        )
        inserted = updated = unchanged = 0
        for row in rows:
            cursor.execute(
                _SELECT_HASH_SQL,
                row.site_id,
                row.parameter_name,
                row.timestamp,
            )
            existing = cursor.fetchone()
            if existing is None:
                cursor.execute(_INSERT_TARGET_SQL, *row.as_db_tuple())
                inserted += 1
            elif bytes(existing[0]) == row.source_hash:
                unchanged += 1
            else:
                cursor.execute(
                    _UPDATE_TARGET_SQL,
                    row.unit,
                    row.predicted_value,
                    row.actual_value,
                    row.shap_json,
                    row.is_anomaly,
                    row.anomaly_score,
                    row.retrain_alert,
                    row.forecast_json,
                    row.model_version,
                    row.source_file,
                    row.source_hash,
                    row.site_id,
                    row.parameter_name,
                    row.timestamp,
                )
                updated += 1
        connection.commit()
        return IngestionSummary(
            staged=len(rows), inserted=inserted, updated=updated, unchanged=unchanged
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def _parse_record(
    payload: dict[str, Any],
    *,
    site_id: str,
    unit_override: str | None,
    expected_parameter: str | None,
    source_file: str,
) -> ArchiveRow:
    parameter = str(payload.get("parameter_name") or "").strip()
    if not parameter:
        raise ValueError("parameter_name is required")
    if expected_parameter and parameter.casefold() != expected_parameter.strip().casefold():
        raise ValueError("parameter mismatch")

    timestamp = _parse_timestamp(payload.get("timestamp"))
    unit = str(unit_override or payload.get("unit") or "").strip() or None
    shap_payload = payload.get("shap_top_features", [])
    if not isinstance(shap_payload, list):
        raise ValueError("shap_top_features must be a list")
    shap_json = json.dumps(shap_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    forecast_payload = payload.get("forecast")
    if forecast_payload is not None and not isinstance(forecast_payload, dict):
        raise ValueError("forecast must be an object")
    forecast_json = (
        json.dumps(forecast_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if forecast_payload is not None
        else None
    )

    alert = payload.get("retrain_alert")
    model_version = payload.get("model_version")
    normalized = {
        "site_id": site_id,
        "parameter_name": parameter,
        "unit": unit,
        "timestamp": timestamp.isoformat(),
        "predicted_value": _as_float(payload.get("predicted_value")),
        "actual_value": _as_float(payload.get("actual_value")),
        "shap_top_features": shap_payload,
        "is_anomaly": _as_bool(payload.get("is_anomaly")),
        "anomaly_score": _as_float(payload.get("anomaly_score")),
        "retrain_alert": str(alert).strip() if isinstance(alert, str) and alert.strip() else None,
        "forecast": forecast_payload,
        "model_version": str(model_version).strip()
        if isinstance(model_version, str) and model_version.strip()
        else None,
    }
    canonical = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    return ArchiveRow(
        site_id=site_id,
        parameter_name=parameter,
        unit=unit,
        timestamp=timestamp,
        predicted_value=normalized["predicted_value"],
        actual_value=normalized["actual_value"],
        shap_json=shap_json,
        is_anomaly=normalized["is_anomaly"],
        anomaly_score=normalized["anomaly_score"],
        retrain_alert=normalized["retrain_alert"],
        forecast_json=forecast_json,
        model_version=normalized["model_version"],
        source_file=source_file,
        source_hash=hashlib.sha256(canonical).digest(),
    )


def _parse_timestamp(value: Any) -> datetime:
    if value in (None, ""):
        raise ValueError("timestamp is required")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid number") from exc
    if not math.isfinite(parsed):
        raise ValueError("non-finite number")
    return parsed


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError("invalid boolean")


def _connect(
    connection_string: str,
    *,
    timeout_seconds: int,
    connection_factory: ConnectionFactory | None,
) -> Any:
    if not connection_string.strip():
        raise ValueError("DB_CONN_STR is not configured")
    if connection_factory is None:
        if pyodbc is None:
            raise RuntimeError("pyodbc is not installed")
        connection_factory = pyodbc.connect
    return connection_factory(
        connection_string,
        timeout=max(1, min(int(timeout_seconds), 60)),
        autocommit=False,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a validated Cold archive export into SQL Server Warm storage."
    )
    parser.add_argument("input", nargs="?", help="Flat JSONL produced by run.py export-archive")
    parser.add_argument("--init-schema", action="store_true", help="Apply the idempotent schema first")
    parser.add_argument("--schema", default=str(_DEFAULT_SCHEMA), help="Schema SQL path")
    parser.add_argument(
        "--site-id",
        default=os.getenv("WQD_WARM_SITE_ID") or os.getenv("WQD_SITE_ID") or "C-1",
        help="Stable site identifier stored with every row",
    )
    parser.add_argument("--unit", default=None, help="Unit to attach when the archive has none")
    parser.add_argument("--parameter", default=None, help="Reject rows for any other parameter")
    parser.add_argument("--timeout", type=int, default=15, help="Connection/query timeout seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    connection_string = os.getenv("DB_CONN_STR", "")
    if not connection_string.strip():
        print("Warm Tier configuration is missing: set DB_CONN_STR.", file=sys.stderr)
        return 2
    if not args.init_schema and not args.input:
        print("Provide an input JSONL file or use --init-schema.", file=sys.stderr)
        return 2

    try:
        if args.init_schema:
            apply_schema(
                connection_string,
                schema_path=args.schema,
                timeout_seconds=args.timeout,
            )
            print("Warm Tier schema is ready.")

        if args.input:
            parsed = parse_archive_file(
                args.input,
                site_id=args.site_id,
                unit_override=args.unit,
                expected_parameter=args.parameter,
            )
            print(
                "Validated "
                f"{len(parsed.rows)} record(s); rejected {parsed.rejected_lines}; "
                f"resolved {parsed.duplicate_timestamps} duplicate timestamp(s)."
            )
            if parsed.rejected_line_numbers:
                shown = ", ".join(str(n) for n in parsed.rejected_line_numbers[:20])
                suffix = "…" if len(parsed.rejected_line_numbers) > 20 else ""
                print(f"Rejected source line(s): {shown}{suffix}")
            if not parsed.rows:
                if parsed.nonempty_lines:
                    print("No valid records were available for ingestion.", file=sys.stderr)
                    return 3
                print("The input file is empty; nothing was ingested.")
                return 0
            result = ingest_rows(
                connection_string,
                parsed.rows,
                timeout_seconds=args.timeout,
            )
            print(
                "Warm Tier ingestion complete: "
                f"inserted={result.inserted}, updated={result.updated}, "
                f"unchanged={result.unchanged}."
            )
        return 0
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Warm Tier input/configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Do not print driver exception text; it may expose server or login data.
        print(f"Warm Tier database operation failed ({type(exc).__name__}).", file=sys.stderr)
        return 4


_SELECT_HASH_SQL = """
SELECT SourceRecordHash
FROM dbo.Fact_WaterQuality WITH (UPDLOCK, HOLDLOCK)
WHERE SiteId = ? AND ParameterName = ? AND MeasurementTimestamp = ?;
"""

_INSERT_TARGET_SQL = """
INSERT dbo.Fact_WaterQuality
(SiteId, ParameterName, Unit, MeasurementTimestamp, PredictedValue, ActualValue,
 ShapTopFeatures, IsAnomaly, AnomalyScore, RetrainAlert, ForecastJson,
 ModelVersion, SourceFile, SourceRecordHash)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_UPDATE_TARGET_SQL = """
UPDATE dbo.Fact_WaterQuality
SET Unit = ?,
    PredictedValue = ?,
    ActualValue = ?,
    ShapTopFeatures = ?,
    IsAnomaly = ?,
    AnomalyScore = ?,
    RetrainAlert = ?,
    ForecastJson = ?,
    ModelVersion = ?,
    SourceFile = ?,
    SourceRecordHash = ?,
    IngestedAt = SYSUTCDATETIME()
WHERE SiteId = ? AND ParameterName = ? AND MeasurementTimestamp = ?;
"""


if __name__ == "__main__":
    raise SystemExit(main())
