"""
wqi.py — Load the historical WQI diagnostic (methodology page only).

The Weighted Arithmetic WQI is NOT part of the live export contract. It exists
only as static per-row columns in the processed training artifact
``data/processed/c1_with_wqi.csv`` (WQI_new, Class_new, Qi_*), computed with
drinking-water reference standards. This module surfaces it strictly as a
documented historical diagnostic — never as a live operational verdict.

If the CSV or the methodology report is absent, every function returns None and
the methodology page simply omits the WQI block (no alarm, no placeholder).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("dashboard.wqi")

# Canonical class order used by src/data/compute_wqi.py (Brown et al., 1972).
CLASS_ORDER = ["Excellent", "Good", "Poor", "Very Poor", "Unsuitable"]


@dataclass
class WqiDiagnostic:
    """Historical WQI summary read from the processed CSV."""

    n_rows: int
    class_counts: dict[str, int] = field(default_factory=dict)
    wqi_min: float | None = None
    wqi_mean: float | None = None
    wqi_max: float | None = None
    methodology_text: str | None = None

    def class_share(self) -> list[tuple[str, int, float]]:
        """Return [(class, count, share_pct)] in canonical order."""
        out: list[tuple[str, int, float]] = []
        for cls in CLASS_ORDER:
            n = self.class_counts.get(cls, 0)
            share = (n / self.n_rows * 100.0) if self.n_rows else 0.0
            out.append((cls, n, share))
        return out


def load_wqi_diagnostic(
    csv_path: str | Path,
    methodology_path: str | Path | None = None,
) -> WqiDiagnostic | None:
    """
    Read the WQI diagnostic from the processed CSV. Returns None if unavailable.

    Uses pandas (already a repository dependency). Tolerant of a missing
    Class_new / WQI_new column: those simply produce empty summaries.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not read WQI CSV %s: %s", csv_path, exc)
        return None

    n_rows = int(len(df))
    class_counts: dict[str, int] = {}
    if "Class_new" in df.columns:
        counts = df["Class_new"].value_counts().to_dict()
        class_counts = {str(k): int(v) for k, v in counts.items()}

    wqi_min = wqi_mean = wqi_max = None
    if "WQI_new" in df.columns:
        col = df["WQI_new"].dropna()
        if len(col):
            wqi_min = float(col.min())
            wqi_mean = float(col.mean())
            wqi_max = float(col.max())

    methodology_text = None
    if methodology_path is not None:
        mp = Path(methodology_path)
        if mp.exists():
            try:
                methodology_text = mp.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not read WQI methodology %s: %s", mp, exc)

    return WqiDiagnostic(
        n_rows=n_rows,
        class_counts=class_counts,
        wqi_min=wqi_min,
        wqi_mean=wqi_mean,
        wqi_max=wqi_max,
        methodology_text=methodology_text,
    )
