"""
reference_limits.py — Documented reference values ONLY. Nothing invented here.

Every value in this module is transcribed verbatim from a source that already
exists in the repository, with its provenance attached:

  • Physical plausibility bounds  → config/system_config.json (validation.physical_bounds)
        Explicitly documented in the repo as sensor-plausibility ranges, NOT
        calibrated water-quality/safety thresholds (see README "Known limitations").
  • WAWQI standards (Si)          → src/data/compute_wqi.py docstring
        BIS IS 10500:2012 / WHO 2022 drinking-water reference standards used to
        build the Weighted Arithmetic WQI. Cited in wqi_methodology.txt.

Display rule
------------
The drinking-water WAWQI values are exposed on the Data & Methodology page only.
They are never used as operational chart limits, anomaly thresholds, or safety
classifications. Physical plausibility bounds remain documentation only as well.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceLine:
    """A single documented reference value for one parameter."""

    value: float
    kind: str            # "standard" (WAWQI Si) | "plausibility" (sensor bound)
    label_key: str       # i18n key for the short label
    source: str          # citable provenance string (shown in tooltip / methodology)
    context: str         # the context the number was defined in (transparency)


# WAWQI Si standards — src/data/compute_wqi.py (STANDARDS dict) + module docstring.
# These are drinking-water reference standards; shown neutrally under "generalist".
_WAWQI_STANDARDS: dict[str, ReferenceLine] = {
    "pH": ReferenceLine(
        value=8.5, kind="standard", label_key="ref_wawqi_si",
        source="BIS IS 10500:2012 (desirable 6.5–8.5); WHO 2022 aesthetic upper limit",
        context="WAWQI Si (drinking-water index)",
    ),
    "EC": ReferenceLine(
        value=1500.0, kind="standard", label_key="ref_wawqi_si",
        source="BIS IS 10500:2012 — EC permissible limit 1500 µS/cm",
        context="WAWQI Si (drinking-water index)",
    ),
    "Turbidity": ReferenceLine(
        value=5.0, kind="standard", label_key="ref_wawqi_si",
        source="BIS IS 10500:2012 — turbidity permissible limit 5 NTU",
        context="WAWQI Si (drinking-water index)",
    ),
}

# Physical plausibility bounds — config/system_config.json validation.physical_bounds.
# Documented as sensor-validation ranges, NOT quality thresholds.
_PLAUSIBILITY_BOUNDS: dict[str, tuple[float, float]] = {
    "pH": (0.0, 14.0),
    "EC": (0.0, 5000.0),
    "Turbidity": (0.0, 10000.0),
}


def reference_lines(parameter: str, water_use_profile: str = "generalist") -> list[ReferenceLine]:
    """
    Return documented WAWQI references for methodology presentation only.

    Operational charts deliberately do not call this function. Physical
    plausibility bounds are wide sensor ranges, not quality references, and are
    exposed separately via ``plausibility_bounds``.
    """
    line = _WAWQI_STANDARDS.get(parameter)
    return [line] if line is not None else []


def plausibility_bounds(parameter: str) -> tuple[float, float] | None:
    """Return the documented (min, max) sensor plausibility bound, or None."""
    return _PLAUSIBILITY_BOUNDS.get(parameter)


def has_reference(parameter: str) -> bool:
    """True if any documented reference exists for the parameter."""
    return parameter in _WAWQI_STANDARDS or parameter in _PLAUSIBILITY_BOUNDS
