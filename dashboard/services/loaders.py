"""
loaders.py — All file I/O for the dashboard. The only module that touches disk.

Reads the two pipeline export files per parameter and returns typed, cleaned
objects. Every function is defensive:

  • missing file            -> empty result, no exception
  • empty file              -> empty result
  • invalid JSON line       -> line skipped, counted in load_errors
  • malformed status JSON   -> None
  • parameter appears/leaves -> handled by discovery, not here

Nothing here imports Streamlit, so it is fully unit-testable and reusable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dashboard.models.schemas import MeasurementRecord, ParameterData, StatusSnapshot
from dashboard.services.validation import clean_records

logger = logging.getLogger("dashboard.loaders")


def _jsonl_path(exports_dir: Path, parameter: str) -> Path:
    return exports_dir / f"{parameter}.jsonl"


def _status_path(exports_dir: Path, parameter: str) -> Path:
    return exports_dir / f"{parameter}_status.json"


def read_jsonl_records(path: Path) -> tuple[list[MeasurementRecord], int]:
    """
    Read a <param>.jsonl file into MeasurementRecords.

    Returns (records, n_errors) where n_errors counts lines that could not be
    parsed as JSON objects. A missing or empty file yields ([], 0).
    """
    if not path.exists():
        return [], 0

    records: list[MeasurementRecord] = []
    errors = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return [], 1

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue
        if not isinstance(obj, dict):
            errors += 1
            continue
        try:
            records.append(MeasurementRecord.from_dict(obj))
        except Exception as exc:  # never let one record kill the load
            logger.debug("Skipping unparseable record in %s: %s", path, exc)
            errors += 1
    if errors:
        logger.info("%s: skipped %d malformed line(s)", path.name, errors)
    return records, errors


def read_status(path: Path) -> StatusSnapshot | None:
    """Read a <param>_status.json file. Returns None if absent or malformed."""
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse status file %s: %s", path, exc)
        return None
    if not isinstance(obj, dict):
        return None
    try:
        return StatusSnapshot.from_dict(obj)
    except Exception as exc:
        logger.warning("Malformed status file %s: %s", path, exc)
        return None


def load_parameter(
    parameter: str,
    exports_dir: str | Path,
    retention_days: int = 30,
) -> ParameterData:
    """
    Load and clean everything the UI needs for one parameter.

    Robust by construction: a broken JSONL still returns whatever records parsed,
    and a missing status file just leaves ParameterData.status as None.
    """
    exports_dir = Path(exports_dir)
    raw_records, errors = read_jsonl_records(_jsonl_path(exports_dir, parameter))
    records = clean_records(raw_records, retention_days)
    status = read_status(_status_path(exports_dir, parameter))

    unit = ""
    if status and status.unit:
        unit = status.unit

    return ParameterData(
        name=parameter,
        unit=unit,
        records=records,
        status=status,
        load_errors=errors,
    )


def load_all_parameters(
    parameters: list[str],
    exports_dir: str | Path,
    retention_days: int = 30,
) -> dict[str, ParameterData]:
    """Load every requested parameter. One bad parameter never blocks the rest."""
    out: dict[str, ParameterData] = {}
    for name in parameters:
        try:
            out[name] = load_parameter(name, exports_dir, retention_days)
        except Exception as exc:  # defensive: keep the dashboard alive
            logger.warning("Failed to load parameter %s: %s", name, exc)
            out[name] = ParameterData(name=name, unit="", records=[], status=None, load_errors=1)
    return out
