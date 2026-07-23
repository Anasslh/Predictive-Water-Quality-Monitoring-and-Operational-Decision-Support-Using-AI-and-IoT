"""
audit_repo.py — Repository hygiene audit.

Produces a text report with two sections:
  A. Empty directories (excluding .git, __pycache__, .pytest_cache)
  B. Python source files that appear to be NEVER imported anywhere else
     in the repository — candidates for manual review.

WHAT "never imported" MEANS HERE
---------------------------------
For each .py file, the script greps the rest of the codebase for:
  - "import <stem>"        (e.g.  import feature_engineering)
  - "from <stem> import"   (e.g.  from feature_engineering import ...)
  - "from .<stem> import"  (relative import from the same package)

"stem" = the filename without extension (e.g. "feature_engineering" for
feature_engineering.py).

FALSE POSITIVES are expected:
  - A module imported by a script not under src/ (e.g. a Jupyter notebook,
    a one-off script in reports/, or a config file referencing the module by
    path string rather than import).
  - A module that is an entry point (run.py, demo_*.py, examples/) —
    these are intentionally excluded from the CANDIDATE list.
  - A module whose stem happens to appear in comments or string literals
    that match the pattern.

FALSE NEGATIVES are also possible:
  - Dynamic imports (importlib.import_module("src.data." + name)) —
    these are not caught by a grep-based search.

DECISION POLICY (matches the _COVARIATES precedent)
------------------------------------------------------
This script NEVER deletes anything. It lists candidates. Before acting on
any candidate, verify manually:
  1. Search for the module name across all file types (not just .py).
  2. Check git log to understand when/why the file was last touched.
  3. Confirm with the team that no external system (e.g. a notebook, a CI
     script, a dashboard) imports it outside this repo.

USAGE
-----
    python reports/audit_repo.py
    python reports/audit_repo.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ── Config ─────────────────────────────────────────────────────────────────────

# Directories skipped entirely when walking the repo
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}

# .py files that are intentional entry points — excluded from "never imported"
# candidates because they are called directly, not imported.
ENTRY_POINT_PATTERNS = [
    # Explicit entry points at root
    lambda p, root: p.name == "run.py" and p.parent == root,
    # demo_* and example_* files
    lambda p, root: p.stem.startswith("demo_") or p.stem.startswith("example_"),
    # Anything under examples/ or notebooks/ directories (standalone scripts)
    lambda p, root: any(d in p.relative_to(root).parts for d in ("examples", "notebooks")),
    # __init__.py files (imported via their package, not by their own stem)
    lambda p, root: p.name == "__init__.py",
    # test_* files (test runners, not library code)
    lambda p, root: p.stem.startswith("test_"),
    # Everything under reports/ that is a standalone generation/audit script
    lambda p, root: p.parent.name == "reports",
    # Validation scripts that are standalone (called directly, not imported)
    lambda p, root: p.stem.startswith("validate_"),
    # Evaluation / comparison scripts (meant to be run directly)
    lambda p, root: p.parent.name == "evaluation",
    # Standalone data scripts with __main__ guard (run directly, not imported)
    lambda p, root: p.parent.name == "data" and p.stem in ("compute_wqi", "split"),
]


def _is_entry_point(path: Path, root: Path) -> bool:
    return any(pred(path, root) for pred in ENTRY_POINT_PATTERNS)


# ── Walk helpers ──────────────────────────────────────────────────────────────

def _walk_files(root: Path, extensions: set[str]) -> list[Path]:
    """Recursively collect files with given extensions, skipping SKIP_DIRS."""
    results = []
    for item in root.rglob("*"):
        if any(skip in item.parts for skip in SKIP_DIRS):
            continue
        if item.is_file() and item.suffix in extensions:
            results.append(item)
    return sorted(results)


def _is_empty_dir(path: Path) -> bool:
    """Return True if the directory contains no files at all (recursively)."""
    return not any(path.rglob("*") if True else [])


def _walk_empty_dirs(root: Path) -> list[Path]:
    results = []
    for item in root.rglob("*"):
        if any(skip in item.parts for skip in SKIP_DIRS):
            continue
        if item.is_dir():
            # Check for any file (recursively) in the dir
            has_file = any(
                f.is_file()
                for f in item.rglob("*")
                if not any(skip in f.parts for skip in SKIP_DIRS)
            )
            if not has_file:
                results.append(item)
    # Deduplicate: only report the outermost empty dirs
    results_sorted = sorted(results, key=lambda p: len(p.parts))
    outermost: list[Path] = []
    for d in results_sorted:
        if not any(d.is_relative_to(o) for o in outermost):
            outermost.append(d)
    return outermost


# ── Import detection ──────────────────────────────────────────────────────────

def _build_corpus(py_files: list[Path]) -> str:
    """Concatenate all .py file contents into one searchable string."""
    parts = []
    for f in py_files:
        try:
            parts.append(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(parts)


def _is_imported(stem: str, corpus: str) -> bool:
    """
    Return True if *stem* appears in at least one import-style reference
    in *corpus*.

    Patterns checked (word-boundary aware):
      import <stem>
      from <stem> import
      from .<stem> import
      from ..<stem> import
    """
    patterns = [
        rf"\bimport\s+{re.escape(stem)}\b",
        rf"\bfrom\s+\.{{0,2}}{re.escape(stem)}\s+import\b",
        # Also catches  from src.sub.stem import  (full dotted path ending in stem)
        rf"\bfrom\s+[\w\.]*\.{re.escape(stem)}\s+import\b",
    ]
    for pat in patterns:
        if re.search(pat, corpus):
            return True
    return False


# ── Report ────────────────────────────────────────────────────────────────────

SEP  = "─" * 64
SEP2 = "═" * 64


def _run(root: Path):
    print()
    print(SEP2)
    print("  Repository hygiene audit")
    print(f"  Root: {root}")
    print(SEP2)

    # ── A. Empty directories ──────────────────────────────────────────────────
    print()
    print(f"  {'A. EMPTY DIRECTORIES':}")
    print(f"  {SEP}")
    empty_dirs = _walk_empty_dirs(root)
    if empty_dirs:
        for d in empty_dirs:
            print(f"    {d.relative_to(root)}/")
    else:
        print("    None found.")

    # ── B. Python files never imported ────────────────────────────────────────
    print()
    print(f"  B. PYTHON FILES WITH NO DETECTED IMPORT REFERENCE")
    print(f"  {SEP}")
    print(f"  (entry points, __init__.py, test_*, demo_*, examples/ are excluded)")
    print()

    all_py = _walk_files(root, {".py"})

    # Build corpus from ALL .py files for grepping
    corpus = _build_corpus(all_py)

    candidates: list[Path] = []
    skipped_entry: list[Path] = []

    for py_file in all_py:
        if _is_entry_point(py_file, root):
            skipped_entry.append(py_file)
            continue
        stem = py_file.stem
        # Build a per-file corpus that excludes the file itself (a file
        # always contains its own stem in function/class names)
        other_py = [f for f in all_py if f != py_file]
        other_corpus = _build_corpus(other_py)
        if not _is_imported(stem, other_corpus):
            candidates.append(py_file)

    if candidates:
        print(f"  {len(candidates)} candidate(s) — verify manually before any deletion:")
        print()
        for c in sorted(candidates):
            rel = c.relative_to(root)
            print(f"    {rel}")
        print()
        print("  For each candidate, suggested checks:")
        print("    git log --oneline -- <file>      # when was it last changed?")
        print("    grep -r '<stem>' . --include='*.py' --include='*.ipynb'")
        print("    # Also check notebooks/, reports/, any CI/CD scripts")
    else:
        print("  No unimported candidates found.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(SEP2)
    print(f"  SUMMARY")
    print(f"  {SEP}")
    print(f"  Empty directories   : {len(empty_dirs)}")
    print(f"  .py files scanned   : {len(all_py)}")
    print(f"  Entry points skipped: {len(skipped_entry)}")
    print(f"  Unimported candidates: {len(candidates)}")
    print(SEP2)
    print()
    print("  REMINDER: This script never deletes anything.")
    print("  Treat candidates as a starting point for manual investigation,")
    print("  not as a definitive list of dead code.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repository hygiene audit")
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of reports/)",
    )
    args = parser.parse_args()
    _run(args.root)
