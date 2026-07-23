"""
test_batch_onboard.py — Tests for 'onboard --all' and 'review' commands.

Verifies that:
  1. onboard --all benchmarks every sensor in a test sensors_config without
     freezing any model (no current_model.json written).
  2. A pending report (.pkl + .json summary) is created for each parameter.
  3. review <param> loads the pending report, runs the interactive freeze
     (with mocked input), and writes a current_model.json.
  4. After review the pending report is archived (_archived_ prefix) and
     no unarchived .pkl remains.

Run:  python src/pipeline/test_batch_onboard.py
      (or via run_tests.sh — picked up automatically)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    symbol = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{symbol}  {label}{suffix}")
    _results.append((label, condition))


# ── Synthetic fixtures ─────────────────────────────────────────────────────────

_SENSORS_CONFIG = {
    "timestamp_column_candidates": ["Date"],
    "sensors": [
        {
            "column_name":    "Alpha",
            "parameter_name": "Alpha",
            "unit":           "mg/L",
            "lag_hours":      [24, 48],
            "rolling_hours":  [72],
            "co_variables":   [],
        },
        {
            "column_name":    "Beta",
            "parameter_name": "Beta",
            "unit":           "NTU",
            "lag_hours":      [24, 48],
            "rolling_hours":  [72],
            "co_variables":   [],
        },
    ],
}

# Minimal system_config.json — only fields required by _validate_configs.
# grid_config = None so benchmark_models() uses its hardcoded defaults.
_SYSTEM_CONFIG = {
    "retraining":        {"min_new_rows": 60, "rmse_ratio_threshold": 1.10,
                          "tolerance": 0.02, "rejection_alert_threshold": 3,
                          "max_history_years": 5.0},
    "anomaly_detection": {"score_threshold": 0.5,
                          "isolation_forest_contamination": 0.05,
                          "random_state": 42},
    "forecasting":       {"horizon_steps": 7},
    "exports":           {"exports_dir": "exports", "retention_days": 30},
    "logging":           {"log_file": "logs/system.log",
                          "console_level": "INFO", "file_level": "WARNING"},
}


def _make_dataset(n: int = 150) -> pd.DataFrame:
    """Synthetic daily dataset with Alpha and Beta columns (150 rows > MIN=100)."""
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Date":  dates.strftime("%Y-%m-%d"),
        "Alpha": (70 + 15 * rng.standard_normal(n)).round(3),
        "Beta":  (20 +  8 * rng.standard_normal(n)).round(3),
    })


def _write_fixtures(tmp: Path) -> tuple[Path, Path, Path]:
    """Write sensors_config, system_config, and dataset CSV to tmp dir."""
    sc_path  = tmp / "sensors_config.json"
    sys_path = tmp / "system_config.json"
    ds_path  = tmp / "dataset.csv"

    sc_path.write_text(json.dumps(_SENSORS_CONFIG), encoding="utf-8")
    sys_path.write_text(json.dumps(_SYSTEM_CONFIG),  encoding="utf-8")
    _make_dataset().to_csv(ds_path, index=False)

    return sc_path, sys_path, ds_path


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for CLI function calls."""
    return argparse.Namespace(**kwargs)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_batch_onboard_saves_reports_no_model_frozen() -> None:
    """
    onboard --all must:
      • create a pending .pkl + .json for each sensor
      • NOT write current_model.json for any parameter
    """
    # suppress pipeline noise during the test
    for lg in ("src.forecasting.feature_engineering_generic",
               "src.forecasting.frequency_detector",
               "src.pipeline.timestamp_detector",
               "src.pipeline.orchestrator",
               "src.pipeline.model_benchmark"):
        logging.getLogger(lg).setLevel(logging.WARNING)

    from run import _cmd_onboard_all, _pending_dir

    with tempfile.TemporaryDirectory(prefix="test_batch_") as tmp_str:
        tmp          = Path(tmp_str)
        sc, sys_cfg, ds = _write_fixtures(tmp)
        models_store    = tmp / "models_store"

        args = _make_args(
            dataset        = str(ds),
            sensors_config = str(sc),
            system_config  = str(sys_cfg),
            models_store   = str(models_store),
            all            = True,
            parameter      = None,
        )

        _cmd_onboard_all(args)

        # ── Check: pending reports exist ──────────────────────────────────
        for param in ("Alpha", "Beta"):
            pdir = _pending_dir(param, models_store)
            pkl_files = [f for f in pdir.glob("*.pkl")
                         if not f.name.startswith("_archived_")]
            check(
                f"[{param}] pending .pkl created",
                len(pkl_files) == 1,
                f"found {len(pkl_files)} file(s) in {pdir}",
            )
            json_files = [f for f in pdir.glob("*.json")
                          if not f.name.startswith("_archived_")]
            check(
                f"[{param}] human-readable .json summary created",
                len(json_files) == 1,
                f"found {len(json_files)} file(s)",
            )

        # ── Check: NO model frozen automatically ──────────────────────────
        for param in ("Alpha", "Beta"):
            pointer_path = models_store / param.lower() / "current_model.json"
            check(
                f"[{param}] current_model.json NOT written (no auto-freeze)",
                not pointer_path.exists(),
                str(pointer_path),
            )


def test_review_freezes_model_and_archives_report() -> None:
    """
    After onboard --all, review <param> must:
      • load the pending report and display the benchmark
      • freeze the model when the user confirms rank
      • write current_model.json
      • archive the pending report (no unarchived .pkl left)
    """
    from run import _cmd_onboard_all, _cmd_review, _pending_dir

    with tempfile.TemporaryDirectory(prefix="test_review_") as tmp_str:
        tmp          = Path(tmp_str)
        sc, sys_cfg, ds = _write_fixtures(tmp)
        models_store    = tmp / "models_store"

        # First run the batch onboard to create the pending report
        onboard_args = _make_args(
            dataset        = str(ds),
            sensors_config = str(sc),
            system_config  = str(sys_cfg),
            models_store   = str(models_store),
            all            = True,
            parameter      = None,
        )
        _cmd_onboard_all(onboard_args)

        # Verify pending report exists before review
        pdir_alpha = _pending_dir("Alpha", models_store)
        pre_pkls = [f for f in pdir_alpha.glob("*.pkl")
                    if not f.name.startswith("_archived_")]
        check(
            "[Alpha] pending report exists before review",
            len(pre_pkls) == 1,
            f"found {len(pre_pkls)}",
        )

        # Run review with mocked input: choose "raw" variant, then rank "1"
        review_args = _make_args(
            parameter      = "Alpha",
            sensors_config = str(sc),
            system_config  = str(sys_cfg),
            models_store   = str(models_store),
        )

        # Alpha CV is low (uniform ~70 ± 15 → CV ≈ 21%) so only "raw" prompt
        # appears; input() is called once for rank.
        with patch("builtins.input", side_effect=["1"]):
            _cmd_review(review_args)

        # ── Check: model frozen (current_model.json exists) ───────────────
        pointer_path = models_store / "alpha" / "current_model.json"
        check(
            "[Alpha] current_model.json written after review",
            pointer_path.exists(),
            str(pointer_path),
        )
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text())
            check(
                "[Alpha] pointer has model_path field",
                "model_path" in pointer,
                str(pointer.keys()),
            )
            model_file = Path(pointer.get("model_path", ""))
            check(
                "[Alpha] model .pkl file written on disk",
                model_file.exists(),
                str(model_file),
            )

        # ── Check: no unarchived .pkl left ────────────────────────────────
        active_pkls = [f for f in pdir_alpha.glob("*.pkl")
                       if not f.name.startswith("_archived_")]
        check(
            "[Alpha] no active pending .pkl after review (report archived)",
            len(active_pkls) == 0,
            f"still found: {[f.name for f in active_pkls]}",
        )
        archived_pkls = list(pdir_alpha.glob("_archived_*.pkl"))
        check(
            "[Alpha] archived .pkl exists after review",
            len(archived_pkls) == 1,
            f"found {len(archived_pkls)}",
        )

        # ── Check: second parameter (Beta) still pending ──────────────────
        pdir_beta    = _pending_dir("Beta", models_store)
        beta_active  = [f for f in pdir_beta.glob("*.pkl")
                        if not f.name.startswith("_archived_")]
        check(
            "[Beta] still has active pending report (not touched by Alpha review)",
            len(beta_active) == 1,
            f"found {len(beta_active)}",
        )
        beta_pointer = models_store / "beta" / "current_model.json"
        check(
            "[Beta] current_model.json NOT written (review not called for Beta)",
            not beta_pointer.exists(),
        )


def test_review_missing_report_exits() -> None:
    """review on a parameter with no pending report must exit with code 1."""
    from run import _cmd_review

    with tempfile.TemporaryDirectory(prefix="test_review_missing_") as tmp_str:
        tmp          = Path(tmp_str)
        models_store = tmp / "models_store"

        review_args = _make_args(
            parameter      = "NonExistent",
            sensors_config = str(ROOT / "config" / "sensors_config.json"),
            system_config  = str(ROOT / "config" / "system_config.json"),
            models_store   = str(models_store),
        )

        try:
            _cmd_review(review_args)
            check("review exits with SystemExit when no report pending", False,
                  "no exception raised")
        except SystemExit as exc:
            check(
                "review exits with SystemExit when no report pending",
                exc.code == 1,
                f"exit code={exc.code}",
            )


# ── runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  batch onboard + review — end-to-end tests")
    print(SEP2)

    # Suppress noisy pipeline logs during tests
    for lg in ("src.forecasting.feature_engineering_generic",
               "src.forecasting.frequency_detector",
               "src.pipeline.timestamp_detector",
               "src.pipeline.orchestrator",
               "src.pipeline.model_benchmark",
               "src.retraining.model_versioning"):
        logging.getLogger(lg).setLevel(logging.WARNING)

    test_batch_onboard_saves_reports_no_model_frozen()
    test_review_freezes_model_and_archives_report()
    test_review_missing_report_exits()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
