"""
settings.py — Centralised, environment-driven configuration for the dashboard.

No hard-coded absolute paths: every path is resolved relative to the repository
root (three levels up from this file) unless explicitly overridden by an
environment variable. This keeps the dashboard portable across machines and
CI while still allowing a deployment to point at a synced ``exports/`` folder
on a separate host.

Environment variables (all optional)
------------------------------------
WQD_EXPORTS_DIR      Directory holding ``<param>.jsonl`` / ``<param>_status.json``.
WQD_PROCESSED_DIR    Directory holding the processed CSV used by the methodology page.
WQD_SITE_NAME        Human-readable monitoring-site label shown in the header.
WQD_SITE_ID          Short site identifier.
WQD_WATER_USE        Water-use profile: ``generalist`` | ``drinking`` | ``irrigation``.
WQD_DEFAULT_LANG     Default UI language: ``en`` | ``ar``.
WQD_RETENTION_DAYS   Rolling window length in days (mirrors the pipeline export).
WQD_FRESH_MULTIPLIER Data-freshness heuristic multiplier (see ASSUMPTIONS.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root: dashboard/config/settings.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Valid water-use profiles. "generalist" makes NO drinking/irrigation safety
# claim; the others are reserved for a future, explicitly-approved deployment
# and only change how reference lines are *labelled*, never invent thresholds.
WATER_USE_PROFILES = ("generalist", "drinking", "irrigation")

# Supported UI languages.
LANGUAGES = ("en", "ar")


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


def _env_str(var: str, default: str) -> str:
    val = os.environ.get(var)
    return val if val not in (None, "") else default


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.environ[var])
    except (KeyError, ValueError):
        return default


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.environ[var])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for one dashboard session."""

    repo_root: Path = _REPO_ROOT
    exports_dir: Path = field(
        default_factory=lambda: _env_path("WQD_EXPORTS_DIR", _REPO_ROOT / "exports")
    )
    processed_dir: Path = field(
        default_factory=lambda: _env_path(
            "WQD_PROCESSED_DIR", _REPO_ROOT / "data" / "processed"
        )
    )
    reports_dir: Path = field(
        default_factory=lambda: _env_path("WQD_REPORTS_DIR", _REPO_ROOT / "reports")
    )
    site_name: str = field(default_factory=lambda: _env_str("WQD_SITE_NAME", "C-1 — Ramgarh Station"))
    site_id: str = field(default_factory=lambda: _env_str("WQD_SITE_ID", "C-1"))
    water_use_profile: str = field(
        default_factory=lambda: _env_str("WQD_WATER_USE", "generalist")
    )
    default_lang: str = field(default_factory=lambda: _env_str("WQD_DEFAULT_LANG", "en"))
    retention_days: int = field(default_factory=lambda: _env_int("WQD_RETENTION_DAYS", 30))
    # Freshness heuristic: a parameter is "stale" when the newest record is older
    # than this multiple of the series' own median inter-arrival gap. This is a
    # UI display heuristic, NOT a scientific threshold — see ASSUMPTIONS.md.
    fresh_multiplier: float = field(default_factory=lambda: _env_float("WQD_FRESH_MULTIPLIER", 3.0))

    # WQI source CSV (processed training artifact — methodology page only).
    @property
    def wqi_csv(self) -> Path:
        return self.processed_dir / "c1_with_wqi.csv"

    @property
    def wqi_methodology_txt(self) -> Path:
        return self.reports_dir / "wqi_methodology.txt"

    def normalised(self) -> "Settings":
        """Return a copy with profile/lang clamped to supported values."""
        prof = self.water_use_profile if self.water_use_profile in WATER_USE_PROFILES else "generalist"
        lang = self.default_lang if self.default_lang in LANGUAGES else "en"
        if prof == self.water_use_profile and lang == self.default_lang:
            return self
        return Settings(
            repo_root=self.repo_root,
            exports_dir=self.exports_dir,
            processed_dir=self.processed_dir,
            reports_dir=self.reports_dir,
            site_name=self.site_name,
            site_id=self.site_id,
            water_use_profile=prof,
            default_lang=lang,
            retention_days=self.retention_days,
            fresh_multiplier=self.fresh_multiplier,
        )


def get_settings() -> Settings:
    """Return the process settings (cheap to build; not cached to stay test-friendly)."""
    return Settings().normalised()
