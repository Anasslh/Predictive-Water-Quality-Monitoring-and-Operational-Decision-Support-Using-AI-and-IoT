"""
validation.py — Incoming-data validation for the water-quality pipeline.

DESIGN PRINCIPLES
-----------------
- Never silently reject: every discarded row is logged with a reason.
- Called before any pipeline step (onboarding, prediction, retraining).
- Returns a structured ValidationReport — callers decide whether to abort
  or proceed with the valid subset; this module never makes that call.
- Bounds come from system_config.json (validation.physical_bounds) so they
  can be tuned without touching code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RejectedRow:
    """A single row that failed at least one validation check."""
    index:    Any    # original DataFrame index
    reasons:  list[str]


@dataclass
class ValidationReport:
    """
    Structured output of validate_incoming_data().

    Attributes
    ----------
    n_input          : Number of rows received.
    n_valid          : Number of rows that passed all checks.
    n_rejected       : Number of rows rejected (len(rejected_rows)).
    rejected_rows    : List of RejectedRow, one per rejected row.
    valid_df         : DataFrame containing only the rows that passed.
    warnings         : Non-fatal issues (e.g. no timestamp column found).
    """
    n_input:        int
    n_valid:        int
    n_rejected:     int
    rejected_rows:  list[RejectedRow]
    valid_df:       pd.DataFrame
    warnings:       list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if every input row passed all checks."""
        return self.n_rejected == 0

    def summary(self) -> str:
        lines = [
            f"ValidationReport: {self.n_input} rows in → "
            f"{self.n_valid} valid / {self.n_rejected} rejected",
        ]
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for rr in self.rejected_rows[:20]:   # cap output to first 20
            lines.append(f"  REJECT  row {rr.index}: {'; '.join(rr.reasons)}")
        if self.n_rejected > 20:
            lines.append(f"  … and {self.n_rejected - 20} more rejected rows (see rejected_rows list)")
        return "\n".join(lines)


# ── Main validator ─────────────────────────────────────────────────────────────

def validate_incoming_data(
    df: pd.DataFrame,
    sensor_config: dict | None = None,
    timestamp_col: str | None = None,
    config_path: str | None = None,
) -> ValidationReport:
    """
    Validate a batch of incoming sensor readings.

    Checks performed (in order):
      1. Expected columns present (from sensor_config["sensors"] column_names).
      2. No duplicate timestamps (if timestamp column is detected or provided).
      3. Values within physical plausibility bounds (from system_config.json).

    Parameters
    ----------
    df            : Raw incoming DataFrame (rows = measurements).
    sensor_config : The dict loaded by load_sensor_config() (from sensors section
                    of system_config.json or sensors_config.json). Pass None to
                    skip column-presence check.
    timestamp_col : Name of the timestamp column. None = auto-detect from
                    sensor_config["timestamp_column_candidates"] or common names.
    config_path   : Optional path to system_config.json (for tests).

    Returns
    -------
    ValidationReport
        Always returned, even if all rows fail. Never raises on data issues.
        Raises only on programming errors (wrong argument types, etc.).
    """
    cfg        = get_config(config_path)
    bounds_cfg = cfg.validation.physical_bounds
    warnings:  list[str]        = []
    rejection_map: dict[Any, list[str]] = {}  # index → list[reason]

    def _reject(idx, reason: str) -> None:
        rejection_map.setdefault(idx, []).append(reason)

    # ── 1. Column presence check ───────────────────────────────────────────────
    if sensor_config is not None:
        expected_cols = [s["column_name"] for s in sensor_config.get("sensors", [])]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            warnings.append(
                f"Expected column(s) not found in incoming data: {missing}. "
                "Skipping per-column checks for missing columns."
            )
            logger.warning(
                "validate_incoming_data: missing expected columns %s", missing
            )

    # ── 2. Duplicate-timestamp check ───────────────────────────────────────────
    ts_col = _resolve_timestamp_col(df, sensor_config, timestamp_col)
    if ts_col is not None:
        dupes = df[df.duplicated(subset=[ts_col], keep=False)]
        if not dupes.empty:
            for idx in dupes.index:
                _reject(idx, f"duplicate timestamp in column '{ts_col}'")
            logger.warning(
                "validate_incoming_data: %d rows have duplicate timestamps",
                len(dupes),
            )
    else:
        warnings.append(
            "No timestamp column detected — duplicate-timestamp check skipped."
        )

    # ── 3. Physical-bounds check ───────────────────────────────────────────────
    # Build the set of declared sensor columns so we can distinguish "no bounds
    # defined for a known sensor" (warn) from "non-sensor column" (silent skip).
    sensor_cols: set[str] = set()
    if sensor_config is not None:
        sensor_cols = {s["column_name"] for s in sensor_config.get("sensors", [])}

    for col in df.columns:
        if col not in bounds_cfg:
            if col in sensor_cols:
                msg = (
                    f"no physical bounds defined for '{col}' in "
                    "system_config.json → validation.physical_bounds "
                    "— skipping range validation for this parameter"
                )
                warnings.append(msg)
                logger.warning("validate_incoming_data: %s", msg)
            continue
        bound = bounds_cfg[col]
        try:
            series = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            warnings.append(f"Column '{col}': could not coerce to numeric — bounds check skipped.")
            continue

        for idx, val in series.items():
            if pd.isna(val):
                _reject(idx, f"column '{col}' is NaN / non-numeric")
            elif val < bound.min:
                _reject(
                    idx,
                    f"column '{col}' = {val:.4g} below physical minimum {bound.min} {bound.unit}",
                )
            elif val > bound.max:
                _reject(
                    idx,
                    f"column '{col}' = {val:.4g} above physical maximum {bound.max} {bound.unit}",
                )

    # ── Build report ───────────────────────────────────────────────────────────
    rejected_indices = set(rejection_map.keys())
    valid_mask       = ~df.index.isin(rejected_indices)
    valid_df         = df.loc[valid_mask].copy()

    rejected_rows = [
        RejectedRow(index=idx, reasons=reasons)
        for idx, reasons in rejection_map.items()
    ]

    report = ValidationReport(
        n_input       = len(df),
        n_valid       = int(valid_mask.sum()),
        n_rejected    = len(rejected_rows),
        rejected_rows = rejected_rows,
        valid_df      = valid_df,
        warnings      = warnings,
    )

    if report.n_rejected > 0:
        logger.warning(
            "validate_incoming_data: %d / %d rows rejected — see report.rejected_rows",
            report.n_rejected, report.n_input,
        )
    else:
        logger.info(
            "validate_incoming_data: all %d rows passed validation", report.n_input
        )

    return report


# ── Private helpers ────────────────────────────────────────────────────────────

def _resolve_timestamp_col(
    df: pd.DataFrame,
    sensor_config: dict | None,
    explicit: str | None,
) -> str | None:
    """Return the name of the timestamp column, or None if not found."""
    if explicit is not None:
        return explicit if explicit in df.columns else None

    candidates: list[str] = []
    if sensor_config is not None:
        candidates = sensor_config.get("timestamp_column_candidates", [])
    # Fallback to common names
    candidates = candidates or ["Date", "timestamp", "date", "Time", "datetime"]

    for c in candidates:
        if c in df.columns:
            return c
    return None
