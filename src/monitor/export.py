"""
export.py — Rolling JSONL export and status JSON for ParameterMonitor results.

Per-parameter files written to exports/:
  <param>.jsonl         — rolling window (30 days), one JSON line per measurement.
  <param>_status.json   — current snapshot: model metadata + 30-day performance.

Both files are updated after every measurement and written atomically
(write-to-.tmp then os.replace).  They are the only files the dashboard
needs: copy them to the dashboard machine without any access to the Python
code or model pickles.

This module is fully generic: no parameter names are hard-coded.

Dashboard file listing
----------------------
Use get_dashboard_export_paths(parameter_name) to retrieve both paths for
a given parameter.  Call this once per parameter and hand the dict to IE.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.monitor import MonitorResult

logger = logging.getLogger(__name__)

_DEFAULT_EXPORTS_DIR    = Path("exports")
_DEFAULT_RETENTION_DAYS = 30
_MIN_SKILL_MEASUREMENTS = 10   # mirrors MIN_PARTITION_ROWS in orchestrator.py


# ── Serialisation ──────────────────────────────────────────────────────────────

def _result_to_record(parameter_name: str, result: "MonitorResult") -> dict[str, Any]:
    """
    Convert a MonitorResult to a flat dict suitable for JSON export.

    actual_value is reconstructed from prediction + residual because
    MonitorResult stores residual (= actual − predicted), not actual directly.
    """
    ts = result.timestamp
    if ts is None:
        ts = result.processed_at
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()
    else:
        ts = str(ts)

    actual: float | None = None
    if result.prediction is not None and result.residual is not None:
        actual = result.prediction + result.residual

    return {
        "timestamp":        ts,
        "parameter_name":   parameter_name,
        "predicted_value":  result.prediction,
        "actual_value":     actual,
        "shap_top_features": result.prediction_shap or [],
        "is_anomaly":       result.anomaly_detected,
        "anomaly_score":    result.anomaly_score,
        "retrain_alert":    result.retrain_alert,
    }


# ── Timestamp parsing ──────────────────────────────────────────────────────────

def _parse_ts(line: str) -> datetime | None:
    """Extract and parse the ISO timestamp from a JSONL line."""
    try:
        obj = json.loads(line)
        ts_str = obj.get("timestamp")
        if not ts_str:
            return None
        dt = datetime.fromisoformat(str(ts_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _export_path(parameter_name: str, exports_dir: Path) -> Path:
    return exports_dir / f"{parameter_name}.jsonl"


def _status_path(parameter_name: str, exports_dir: Path) -> Path:
    return exports_dir / f"{parameter_name}_status.json"


# ── Public API ─────────────────────────────────────────────────────────────────

def append_measurement(
    parameter_name: str,
    result: "MonitorResult",
    exports_dir: str | Path = _DEFAULT_EXPORTS_DIR,
    retention_days: int     = _DEFAULT_RETENTION_DAYS,
) -> None:
    """
    Append one MonitorResult to the parameter's JSONL export file, then
    purge lines older than retention_days.

    The purge is timestamp-based (the 'timestamp' field inside each JSON
    line), not based on file modification time or line count. The file
    after this call therefore always satisfies:

        min(record.timestamp) >= now - retention_days

    Parameters
    ----------
    parameter_name : e.g. "EC", "pH", "Turbidity"  (drives the filename)
    result         : MonitorResult from process_new_measurement()
    exports_dir    : Directory for JSONL files (created if absent)
    retention_days : Sliding window length in days (default 30)
    """
    exports_dir = Path(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    path       = _export_path(parameter_name, exports_dir)
    new_line   = json.dumps(_result_to_record(parameter_name, result), default=str)
    cutoff     = datetime.now(timezone.utc) - timedelta(days=retention_days)

    # Read existing lines (empty list if file does not exist yet)
    existing: list[str] = []
    if path.exists():
        existing = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    all_lines = existing + [new_line]

    # Retain only lines whose timestamp is within the window
    kept: list[str] = []
    purged = 0
    for line in all_lines:
        ts = _parse_ts(line)
        if ts is None or ts >= cutoff:
            kept.append(line)
        else:
            purged += 1

    # Atomic write (crash-safety, not concurrency): write full content to .tmp,
    # then rename. A crash mid-write leaves the previous .jsonl intact.
    # Single-process deployment only — see model_versioning docstring for the
    # multi-process concurrency limitation.
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.replace(tmp, path)

    if purged:
        logger.info(
            "[%s] export: +1 record, purged %d line(s) older than %d days → %d kept",
            parameter_name, purged, retention_days, len(kept),
        )
    else:
        logger.debug(
            "[%s] export: +1 record → %d kept", parameter_name, len(kept)
        )


def read_export(
    parameter_name: str,
    exports_dir: str | Path = _DEFAULT_EXPORTS_DIR,
    retention_days: int     = _DEFAULT_RETENTION_DAYS,
) -> list[dict[str, Any]]:
    """
    Return all records currently within the retention window, oldest first.

    Applies the same timestamp-based filter as append_measurement() so the
    caller always sees a consistent 30-day view even if the file was not
    recently pruned.

    Parameters
    ----------
    parameter_name : e.g. "EC", "pH", "Turbidity"
    exports_dir    : Directory for JSONL files
    retention_days : Window length in days (default 30)

    Returns
    -------
    List of dicts (one per measurement), oldest first.
    """
    path = _export_path(parameter_name, Path(exports_dir))
    if not path.exists():
        return []

    cutoff  = datetime.now(timezone.utc) - timedelta(days=retention_days)
    records: list[dict] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _parse_ts(line)
        if ts is None or ts >= cutoff:
            records.append(obj)

    return records


# ── Performance metrics ────────────────────────────────────────────────────────

def _compute_performance_30d(records: list[dict]) -> dict[str, Any]:
    """
    Compute model skill vs. naive persistence baseline from JSONL records.

    Only records with both predicted_value and actual_value contribute.
    Both the model RMSE and the persistence RMSE are computed on the same
    n-1 aligned pairs (index i vs i-1) so the comparison is fair.

    Persistence baseline: predict actual[i-1] for timestep i.

    Skill formula:
        skill = (RMSE_persistence - RMSE_model) / RMSE_persistence × 100
    Positive = model beats persistence; 0 = equal; negative = worse.

    Returns a dict suitable for the "performance_30d" field of status.json.
    Returns null metrics with "insufficient_data": true when fewer than
    _MIN_SKILL_MEASUREMENTS records are available.
    """
    pairs = [
        (r["predicted_value"], r["actual_value"])
        for r in records
        if r.get("predicted_value") is not None and r.get("actual_value") is not None
    ]
    n = len(pairs)

    if n < _MIN_SKILL_MEASUREMENTS:
        return {
            "skill_vs_persistence_pct": None,
            "rmse":                     None,
            "mae":                      None,
            "n_measurements":           n,
            "insufficient_data":        True,
        }

    predicted = [p for p, _ in pairs]
    actual    = [a for _, a in pairs]

    # Align both metrics on pairs (i, i−1): n−1 pairs total
    n_pairs     = n - 1
    pred_al     = predicted[1:]
    actual_al   = actual[1:]
    actual_prev = actual[:-1]

    rmse_model = (sum((p - a) ** 2 for p, a in zip(pred_al, actual_al)) / n_pairs) ** 0.5
    mae_model  =  sum(abs(p - a)   for p, a in zip(pred_al, actual_al)) / n_pairs

    rmse_pers  = (sum((a - prev) ** 2 for a, prev in zip(actual_al, actual_prev)) / n_pairs) ** 0.5

    if rmse_pers == 0.0:
        # Degenerate: constant series → persistence is perfect, skill undefined
        skill: float | None = None
    else:
        skill = (rmse_pers - rmse_model) / rmse_pers * 100.0

    return {
        "skill_vs_persistence_pct": round(skill, 2) if skill is not None else None,
        "rmse":                     round(rmse_model, 4),
        "mae":                      round(mae_model,  4),
        "n_measurements":           n,
    }


# ── Status JSON export ─────────────────────────────────────────────────────────

def write_status_export(
    parameter_name:   str,
    unit:             str       = "",
    models_store_path: "Path | None" = None,
    exports_dir:      "str | Path"   = _DEFAULT_EXPORTS_DIR,
    retention_days:   int            = _DEFAULT_RETENTION_DAYS,
) -> None:
    """
    Write exports/<parameter_name>_status.json — dashboard-ready snapshot.

    Reads the JSONL file (same retention window as append_measurement) to
    compute 30-day performance metrics, then reads models_store_path for
    model metadata.  Written atomically so a concurrent reader always sees
    either the previous complete file or the new complete file.

    Call this after every append_measurement() so the status file is always
    fresh.

    Parameters
    ----------
    parameter_name    : e.g. "EC", "pH", "Turbidity"
    unit              : Physical unit string, e.g. "µS/cm" (from sensors_config)
    models_store_path : Directory with current_model.json + rejection_counter.json
    exports_dir       : Directory for JSONL and status files
    retention_days    : Sliding window length (days) — must match append_measurement
    """
    # Import here to avoid a top-level cycle: monitor/__init__.py imports export.py,
    # and model_versioning.py has no dependency on export.py.
    from src.retraining.model_versioning import (
        read_current_model_pointer,
        read_rejection_counter,
    )

    exports_dir = Path(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    # ── Model metadata from models_store ──────────────────────────────────────
    model_version          = None
    last_promoted_at       = None
    consecutive_rejections = 0
    pending_approvals      = 0

    store = Path(models_store_path) if models_store_path else None
    if store and store.exists():
        pointer = read_current_model_pointer(store)
        if pointer:
            model_version    = pointer.get("model_version")
            last_promoted_at = pointer.get("promoted_at")

        consecutive_rejections = read_rejection_counter(store)

        approval_dir = store / "pending_approvals"
        if approval_dir.exists():
            pending_approvals = len(list(approval_dir.glob("*.json")))

    # ── 30-day performance from the JSONL file ────────────────────────────────
    records = read_export(parameter_name, exports_dir, retention_days)

    last_measurement_at: str | None = None
    if records:
        last_measurement_at = records[-1].get("timestamp")

    performance = _compute_performance_30d(records)

    # ── Assemble and write atomically ─────────────────────────────────────────
    status: dict[str, Any] = {
        "parameter_name":         parameter_name,
        "unit":                   unit,
        "model_version":          model_version,
        "last_promoted_at":       last_promoted_at,
        "last_measurement_at":    last_measurement_at,
        "performance_30d":        performance,
        "pending_approvals":      pending_approvals,
        "consecutive_rejections": consecutive_rejections,
        "queried_at":             datetime.now(timezone.utc).isoformat(),
    }

    out_path = _status_path(parameter_name, exports_dir)
    tmp      = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)

    skill = performance.get("skill_vs_persistence_pct")
    logger.debug(
        "[%s] status.json written  skill=%s%%  n=%d",
        parameter_name,
        f"{skill:.1f}" if skill is not None else "n/a",
        performance.get("n_measurements", 0),
    )


# ── Path helper for IE dashboard integration ───────────────────────────────────

def get_dashboard_export_paths(
    parameter_name: str,
    exports_dir:    "str | Path" = _DEFAULT_EXPORTS_DIR,
) -> dict[str, Path]:
    """
    Return the two file paths the dashboard needs for this parameter.

    Both files are updated after every measurement call and are safe to
    read at any time (atomic writes guarantee no partial reads).
    Transfer / copy both to the dashboard machine together to get a
    consistent snapshot.

    Returns
    -------
    {
        "measurements": Path,   # rolling JSONL — one record per measurement (30-day window)
        "status":       Path,   # current status JSON — model metadata + 30-day metrics
    }
    """
    d = Path(exports_dir)
    return {
        "measurements": _export_path(parameter_name, d),
        "status":       _status_path(parameter_name, d),
    }
