"""
translator.py — Small translation helper with graceful fallback.

Usage
-----
    tr = Translator("ar")
    tr.t("nav_overview")     # -> Arabic label, English if the key is missing in AR
    tr.is_rtl                # -> True for Arabic

The translator never raises on an unknown key: it falls back to the English
table, then to the raw key, so the UI degrades to readable text rather than
crashing or showing an empty label.
"""

from __future__ import annotations

from dataclasses import dataclass

from dashboard.i18n.strings import EN, TABLES

_RTL_LANGS = {"ar"}


@dataclass(frozen=True)
class Translator:
    """Resolve UI strings for a chosen language."""

    lang: str = "en"

    @property
    def normalised_lang(self) -> str:
        return self.lang if self.lang in TABLES else "en"

    @property
    def is_rtl(self) -> bool:
        return self.normalised_lang in _RTL_LANGS

    @property
    def dir(self) -> str:
        return "rtl" if self.is_rtl else "ltr"

    @property
    def align(self) -> str:
        return "right" if self.is_rtl else "left"

    def t(self, key: str) -> str:
        """Translate a key; fall back EN → raw key."""
        table = TABLES[self.normalised_lang]
        if key in table:
            return table[key]
        return EN.get(key, key)

    def __call__(self, key: str) -> str:  # convenience: tr("key")
        return self.t(key)
