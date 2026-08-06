"""
archive.py — Permanent per-parameter measurement archive with monthly compression.

DATA LAYERS
-----------
Three complementary stores serve different purposes:

  Raw dataset  (data/processed/<param>.csv)
      Sensor measurements only — no predictions, no SHAP.
      Accumulated indefinitely; updated by the data team, not the pipeline.

  Hot export   (exports/<param>.jsonl)
      Rolling 30-day window: predictions + actuals + anomaly scores + SHAP.
      Purged automatically; designed for live dashboard consumption.

  Cold archive (archive/<param>_<YYYY-MM>.jsonl[.gz])   ← this module
      Permanent record of every measurement processed by the pipeline.
      Same schema as the Hot export; never purged.
      Current month: plain .jsonl (in progress).
      Past months: gzip-compressed .jsonl.gz (read-only, automatic).

FILE NAMING
-----------
  archive/EC_2026-07.jsonl      # current month — plain text
  archive/EC_2026-06.jsonl.gz   # past months   — gzip
  archive/EC_2026-05.jsonl.gz

ROTATION
--------
Before each append, any plain-text .jsonl files for this parameter whose
month is earlier than the current calendar month are automatically
compressed to .jsonl.gz and the originals removed.  Rotation is
idempotent: calling it when nothing needs rotating is a no-op.

READS
-----
read_archive() spans any date range across compressed and plain files.
It decompresses .gz files in memory (no temp files) and applies
timestamp-based filtering so callers never see out-of-range records.

ATOMIC WRITES
-------------
Each append uses a .tmp → os.replace() pattern identical to export.py:
crash-safe for single-process deployments, not concurrency-safe.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.monitor import MonitorResult

from src.monitor.export import _result_to_record, _parse_ts

logger = logging.getLogger(__name__)

_DEFAULT_ARCHIVE_DIR = Path("archive")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _month_tag(dt: date | datetime) -> str:
    """Return 'YYYY-MM' string for the given date or datetime."""
    return dt.strftime("%Y-%m")


def _plain_path(parameter_name: str, archive_dir: Path, month: str) -> Path:
    """Path for the plain-text current-month file."""
    return archive_dir / f"{parameter_name}_{month}.jsonl"


def _gz_path(parameter_name: str, archive_dir: Path, month: str) -> Path:
    """Path for the gzip-compressed past-month file."""
    return archive_dir / f"{parameter_name}_{month}.jsonl.gz"


def _compress_file(plain: Path) -> Path:
    """
    Compress *plain* to *plain*.gz in-place, then remove *plain*.
    Returns the path of the compressed file.
    """
    gz = plain.with_suffix(".jsonl.gz")
    data = plain.read_bytes()
    gz.write_bytes(gzip.compress(data, compresslevel=9))
    plain.unlink()
    logger.info("[archive] Compressed %s → %s (%d → %d bytes)",
                plain.name, gz.name, len(data), gz.stat().st_size)
    return gz


def _iter_lines(path: Path) -> list[str]:
    """Read all non-empty lines from a .jsonl or .jsonl.gz file."""
    if path.suffix == ".gz":
        raw = gzip.decompress(path.read_bytes()).decode("utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    return [ln for ln in raw.splitlines() if ln.strip()]


# ── Rotation ───────────────────────────────────────────────────────────────────

def _rotate_old_months(parameter_name: str, archive_dir: Path) -> None:
    """
    Compress any plain .jsonl files for *parameter_name* that belong to a
    month earlier than the current calendar month.

    This is called automatically inside append_to_archive() before writing.
    It is idempotent: multiple calls in the same month are no-ops.
    """
    current_month = _month_tag(datetime.now(timezone.utc))
    prefix = f"{parameter_name}_"

    for plain in sorted(archive_dir.glob(f"{prefix}????-??.jsonl")):
        # Extract the YYYY-MM from the filename
        stem = plain.stem          # e.g. "EC_2026-06"
        month = stem[len(prefix):]  # e.g. "2026-06"
        if month < current_month:
            try:
                _compress_file(plain)
            except Exception as exc:
                logger.error("[archive] Compression of %s failed: %s", plain.name, exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def append_to_archive(
    parameter_name: str,
    result: "MonitorResult",
    archive_dir: str | Path = _DEFAULT_ARCHIVE_DIR,
) -> None:
    """
    Append one MonitorResult to the permanent archive for *parameter_name*.

    1. Creates archive_dir if it does not exist.
    2. Compresses any previous-month plain files (rotation).
    3. Atomically appends to archive/<param>_<YYYY-MM>.jsonl.

    Parameters
    ----------
    parameter_name : e.g. "EC", "pH", "Turbidity"
    result         : MonitorResult from process_new_measurement()
    archive_dir    : Root directory for archive files (default: "archive/")
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Rotate before writing so stale plain files are compressed first
    _rotate_old_months(parameter_name, archive_dir)

    current_month = _month_tag(datetime.now(timezone.utc))
    target = _plain_path(parameter_name, archive_dir, current_month)

    new_line = json.dumps(_result_to_record(parameter_name, result), default=str)

    # Read existing lines; append new one; atomic write
    existing: list[str] = _iter_lines(target) if target.exists() else []
    all_lines = existing + [new_line]

    tmp = target.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    os.replace(tmp, target)

    logger.debug("[%s] archive: +1 record → %s (%d total)",
                 parameter_name, target.name, len(all_lines))


def read_archive(
    parameter_name: str,
    archive_dir: str | Path = _DEFAULT_ARCHIVE_DIR,
    start_date: str | date | datetime | None = None,
    end_date:   str | date | datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Return all archived records for *parameter_name* within the given date range.

    Reads across both compressed (.jsonl.gz) and plain (.jsonl) files,
    decompressing in memory as needed.  Records are returned oldest-first.

    Parameters
    ----------
    parameter_name : e.g. "EC", "pH", "Turbidity"
    archive_dir    : Root directory for archive files (default: "archive/")
    start_date     : Inclusive lower bound (ISO string, date, or datetime).
                     None = no lower bound.
    end_date       : Inclusive upper bound (ISO string, date, or datetime).
                     None = no upper bound.

    Returns
    -------
    List of dicts (one per measurement), oldest first, filtered by date range.
    """
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return []

    # Normalise bounds to tz-aware datetimes
    def _to_dt(v: str | date | datetime | None) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, date) and not isinstance(v, datetime):
            v = datetime(v.year, v.month, v.day)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v

    dt_start = _to_dt(start_date)
    dt_end   = _to_dt(end_date)

    # Collect all archive files for this parameter (plain + compressed)
    prefix = f"{parameter_name}_"
    files: list[Path] = sorted(
        list(archive_dir.glob(f"{prefix}????-??.jsonl")) +
        list(archive_dir.glob(f"{prefix}????-??.jsonl.gz"))
    )

    # Filter files by month range for speed (avoid reading unneeded files)
    if dt_start or dt_end:
        start_month = _month_tag(dt_start) if dt_start else None
        end_month   = _month_tag(dt_end)   if dt_end   else None
        filtered = []
        for f in files:
            stem  = f.name.split(".")[0]          # e.g. "EC_2026-06"
            month = stem[len(prefix):]             # e.g. "2026-06"
            if start_month and month < start_month:
                continue
            if end_month and month > end_month:
                continue
            filtered.append(f)
        files = filtered

    records: list[dict] = []
    for path in files:
        try:
            lines = _iter_lines(path)
        except Exception as exc:
            logger.warning("[archive] Could not read %s: %s", path.name, exc)
            continue

        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(line)
            if ts is not None:
                if dt_start and ts < dt_start:
                    continue
                if dt_end and ts > dt_end:
                    continue
            records.append(obj)

    return records


def export_archive_range(
    parameter_name: str,
    start_date:     "str | date | datetime",
    end_date:       "str | date | datetime",
    output_path:    "str | Path",
    archive_dir:    "str | Path" = _DEFAULT_ARCHIVE_DIR,
) -> int:
    """
    Export a time-bounded slice of the Cold archive to a single flat JSONL file.

    Intended for the IE team to bootstrap their Warm tier (12-month window)
    from the permanent Cold archive without having to implement JSONL/gzip
    reading themselves.

    The output file is a plain, self-contained .jsonl: one JSON object per
    line, oldest record first, no compression.  The caller chooses the path;
    the file is written atomically (.tmp → os.replace) so a partial run never
    leaves a truncated output.

    Parameters
    ----------
    parameter_name : e.g. "EC", "pH", "Turbidity"
    start_date     : Inclusive lower bound (ISO string, date, or datetime).
    end_date       : Inclusive upper bound (ISO string, date, or datetime).
    output_path    : Destination path for the exported JSONL file.
                     Parent directory is created if it does not exist.
    archive_dir    : Root directory of the Cold archive (default: "archive/")

    Returns
    -------
    int — number of records written to output_path.

    Raises
    ------
    ValueError if no records match the requested range (prevents writing an
    empty file silently).
    """
    records = read_archive(
        parameter_name,
        archive_dir = archive_dir,
        start_date  = start_date,
        end_date    = end_date,
    )

    if not records:
        raise ValueError(
            f"[{parameter_name}] No archive records found between "
            f"{start_date} and {end_date}. "
            f"Check that the archive directory exists and covers this range."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [json.dumps(r, default=str) for r in records]
    tmp   = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, output_path)

    logger.info(
        "[%s] export_archive_range: %d records → %s",
        parameter_name, len(records), output_path,
    )
    return len(records)


def list_archive_files(
    parameter_name: str,
    archive_dir: str | Path = _DEFAULT_ARCHIVE_DIR,
) -> list[dict[str, Any]]:
    """
    Return metadata for every archive file of *parameter_name*, newest first.

    Useful for a quick inventory without reading all records.

    Returns
    -------
    List of dicts with keys: path, month, compressed, size_bytes.
    """
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return []

    prefix = f"{parameter_name}_"
    files  = sorted(
        list(archive_dir.glob(f"{prefix}????-??.jsonl")) +
        list(archive_dir.glob(f"{prefix}????-??.jsonl.gz")),
        reverse=True,
    )

    result = []
    for f in files:
        stem  = f.name.split(".")[0]
        month = stem[len(prefix):]
        result.append({
            "path":        f,
            "month":       month,
            "compressed":  f.suffix == ".gz",
            "size_bytes":  f.stat().st_size,
        })
    return result
