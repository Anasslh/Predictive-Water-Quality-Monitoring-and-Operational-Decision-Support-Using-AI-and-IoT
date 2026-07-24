"""
discovery.py — Dynamic parameter discovery from the exports directory.

Parameters are never hard-coded. A parameter is anything with a top-level
``<name>.jsonl`` file in the exports directory. Status files (``*_status.json``)
and the deployment's temp files (``*.tmp``) are ignored, as are sub-directories
(e.g. the pipeline's ``_canary_disabled/`` folder), so only genuinely exported
parameters appear.
"""

from __future__ import annotations

from pathlib import Path


def discover_parameters(exports_dir: str | Path) -> list[str]:
    """
    Return the sorted list of parameter names that have a JSONL export.

    A missing exports directory yields an empty list (the UI then shows a calm
    "no exported parameters" state rather than crashing).
    """
    exports_dir = Path(exports_dir)
    if not exports_dir.is_dir():
        return []

    names: set[str] = set()
    for entry in exports_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix != ".jsonl":
            continue
        stem = entry.stem
        if not stem or stem.endswith("_status"):
            continue
        names.add(stem)
    return sorted(names)
