"""
config.py — Unified configuration loader for the water-quality monitoring pipeline.

Usage
-----
    from src.config import get_config, SystemConfig

    cfg = get_config()                    # singleton, reads config/system_config.json once
    min_rows = cfg.retraining.min_new_rows
    threshold = cfg.anomaly.score_threshold

Override at construction time (tests, per-deployment tuning):
    # All module constructors accept explicit arguments that override config defaults.
    manager = RetrainManager(..., min_new_rows=30)   # overrides cfg.retraining.min_new_rows

Schema
------
See config/system_config.json for the authoritative schema and per-field rationale.
The Python dataclasses below mirror that schema; add new fields in both places.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "system_config.json"
_SINGLETON: "SystemConfig | None" = None


# ── Section dataclasses ────────────────────────────────────────────────────────

@dataclass
class RetrainingConfig:
    min_new_rows:               int   = 60
    rmse_ratio_threshold:       float = 1.10
    tolerance:                  float = 0.02
    rejection_alert_threshold:  int   = 3
    max_history_years:          float = 5.0


@dataclass
class AnomalyConfig:
    score_threshold:                  float = 0.5
    isolation_forest_contamination:   float = 0.05
    random_state:                     int   = 42


@dataclass
class ForecastingConfig:
    default_horizon_hours: int = 72


@dataclass
class PhysicalBound:
    min:  float
    max:  float
    unit: str  = ""
    note: str  = ""


@dataclass
class ValidationConfig:
    max_duplicate_timestamp_fraction: float = 0.0
    physical_bounds: dict[str, PhysicalBound] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    log_file:       str = "logs/system.log"
    console_level:  str = "INFO"
    file_level:     str = "WARNING"
    format:         str = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    date_format:    str = "%Y-%m-%d %H:%M:%S"


@dataclass
class ExportsConfig:
    exports_dir:    str = "exports"
    retention_days: int = 30


@dataclass
class SystemConfig:
    """
    Top-level configuration object. Instantiated once by get_config().

    Attributes mirror the sections of config/system_config.json:
        retraining  — volume/drift gates, sliding window
        anomaly     — Isolation Forest + score threshold
        forecasting — recursive horizon defaults
        validation  — physical plausibility bounds
        logging     — console + file log settings
    """
    retraining:  RetrainingConfig  = field(default_factory=RetrainingConfig)
    anomaly:     AnomalyConfig     = field(default_factory=AnomalyConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    validation:  ValidationConfig  = field(default_factory=ValidationConfig)
    logging:     LoggingConfig     = field(default_factory=LoggingConfig)
    exports:     ExportsConfig     = field(default_factory=ExportsConfig)


# ── Loader ─────────────────────────────────────────────────────────────────────

def _parse_config(raw: dict) -> SystemConfig:
    r = raw.get("retraining", {})
    a = raw.get("anomaly_detection", {})
    f = raw.get("forecasting", {})
    v = raw.get("validation", {})
    lo = raw.get("logging", {})

    bounds: dict[str, PhysicalBound] = {}
    for param, spec in v.get("physical_bounds", {}).items():
        bounds[param] = PhysicalBound(
            min=float(spec.get("min", 0.0)),
            max=float(spec.get("max", float("inf"))),
            unit=spec.get("unit", ""),
            note=spec.get("note", ""),
        )

    return SystemConfig(
        retraining=RetrainingConfig(
            min_new_rows              = int(r.get("min_new_rows",              60)),
            rmse_ratio_threshold      = float(r.get("rmse_ratio_threshold",    1.10)),
            tolerance                 = float(r.get("tolerance",               0.02)),
            rejection_alert_threshold = int(r.get("rejection_alert_threshold", 3)),
            max_history_years         = float(r.get("max_history_years",       5.0)),
        ),
        anomaly=AnomalyConfig(
            score_threshold                = float(a.get("score_threshold",                  0.5)),
            isolation_forest_contamination = float(a.get("isolation_forest_contamination",   0.05)),
            random_state                   = int(a.get("random_state",                       42)),
        ),
        forecasting=ForecastingConfig(
            default_horizon_hours = int(f.get("default_horizon_hours", 72)),
        ),
        validation=ValidationConfig(
            max_duplicate_timestamp_fraction = float(v.get("max_duplicate_timestamp_fraction", 0.0)),
            physical_bounds                  = bounds,
        ),
        logging=LoggingConfig(
            log_file      = lo.get("log_file",      "logs/system.log"),
            console_level = lo.get("console_level", "INFO"),
            file_level    = lo.get("file_level",    "WARNING"),
            format        = lo.get("format",        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"),
            date_format   = lo.get("date_format",   "%Y-%m-%d %H:%M:%S"),
        ),
        exports=ExportsConfig(
            exports_dir    = raw.get("exports", {}).get("exports_dir",    "exports"),
            retention_days = int(raw.get("exports", {}).get("retention_days", 30)),
        ),
    )


def get_config(config_path: str | Path | None = None) -> SystemConfig:
    """
    Return the singleton SystemConfig, loading from disk on first call.

    Parameters
    ----------
    config_path : optional override for the JSON path (useful in tests).
                  Passing a non-None value forces a reload from that path.

    Returns
    -------
    SystemConfig
    """
    global _SINGLETON
    if config_path is not None:
        path = Path(config_path)
        with open(path, encoding="utf-8") as f:
            return _parse_config(json.load(f))

    if _SINGLETON is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                _SINGLETON = _parse_config(json.load(f))
        else:
            # Fallback to pure defaults when running outside the project root.
            _SINGLETON = SystemConfig()

    return _SINGLETON


def reload_config() -> SystemConfig:
    """Force-reload from disk (useful after editing system_config.json at runtime).

    Utility function for tests and manual reload — not called in production flow.
    """
    global _SINGLETON
    _SINGLETON = None
    return get_config()
