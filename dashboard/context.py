"""
context.py — Immutable per-render context passed to every view.

Bundles the already-loaded, already-cleaned data and the session's settings /
translator so views stay pure presentation: they never touch disk or discover
parameters themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from dashboard.config.settings import Settings
from dashboard.i18n.translator import Translator
from dashboard.models.schemas import ParameterData


@dataclass(frozen=True)
class AppContext:
    settings: Settings
    tr: Translator
    params: dict[str, ParameterData] = field(default_factory=dict)
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def parameter_names(self) -> list[str]:
        return list(self.params.keys())

    def has_data(self) -> bool:
        return any(p.has_records for p in self.params.values())
