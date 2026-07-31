"""
format.py — Presentation-only value formatting.

Numbers, units and timestamps always render left-to-right, even under an RTL
Arabic layout (mixing RTL text with LTR numerals is the correct typographic
behaviour). The em-dash "—" is the single, consistent placeholder for any
missing/None value across the whole UI.
"""

from __future__ import annotations

from datetime import datetime

from dashboard.i18n.translator import Translator

DASH = "—"


def fmt_number(value: float | int | None, digits: int = 2) -> str:
    """Format a number with fixed decimals; DASH when None."""
    if value is None:
        return DASH
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def fmt_value(value: float | None, unit: str = "", digits: int = 2) -> str:
    """Number + unit, e.g. '741.00 µS/cm'. DASH when None."""
    if value is None:
        return DASH
    num = fmt_number(value, digits)
    return f"{num} {unit}".strip()


def fmt_signed(value: float | None, unit: str = "", digits: int = 2) -> str:
    """Signed delta with explicit + / − sign; DASH when None."""
    if value is None:
        return DASH
    sign = "+" if value > 0 else ("−" if value < 0 else "")
    num = fmt_number(abs(value), digits)
    return f"{sign}{num} {unit}".strip()


def fmt_pct(value: float | None, digits: int = 1) -> str:
    """Percentage with sign, e.g. '+26.7%'. DASH when None."""
    if value is None:
        return DASH
    sign = "+" if value > 0 else ("−" if value < 0 else "")
    return f"{sign}{abs(value):.{digits}f}%"


def fmt_timestamp(dt: datetime | None, *, with_time: bool = True) -> str:
    """Compact ISO-like timestamp (always LTR). DASH when None."""
    if dt is None:
        return DASH
    return dt.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")


def fmt_age(seconds: float | None, tr: Translator) -> str:
    """Human-readable age like '3 h ago' / 'منذ 3 س'. DASH when None."""
    if seconds is None:
        return DASH
    s = int(max(0, seconds))
    is_ar = tr.is_rtl
    if s < 60:
        val, unit = s, ("ث" if is_ar else "s")
    elif s < 3600:
        val, unit = s // 60, ("د" if is_ar else "min")
    elif s < 86400:
        val, unit = s // 3600, ("س" if is_ar else "h")
    else:
        val, unit = s // 86400, ("ي" if is_ar else "d")
    return f"منذ {val} {unit}" if is_ar else f"{val} {unit} ago"


TREND_GLYPH = {"up": "▲", "down": "▼", "flat": "▬", "unknown": DASH}
