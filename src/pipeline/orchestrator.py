"""
orchestrator.py — Generic sensor onboarding pipeline.

RESPONSIBILITY
--------------
This module wires together all pipeline components (timestamp detection,
frequency detection, feature engineering, data splitting, model benchmarking,
and human-gated approval) into a single, parameter-agnostic workflow:

    1. Load raw data.
    2. Detect timestamp column (via sensors_config.json candidates).
    3. Detect measurement frequency.
    4. Run quick noise diagnostic + auto-flag log-transform need.
    5. Build features using the time-aware feature engineering function.
    6. Split chronologically (70 / 15 / 15).
    7. Benchmark RF / XGBoost / SVR on the RAW target.
    7b. If signal CV > LOG_CV_THRESHOLD: automatically also benchmark on
        log1p(target), back-transform predictions, and present both reports.
    8. Human explicitly chooses a model rank AND a variant → freeze.

DESIGN CONSTRAINTS (enforced, not advisory)
--------------------------------------------
  • NO automatic model selection or variant choice. Both reports are
    presented; the human decides which variant (raw or log) to use and
    which rank to submit via submit_benchmark_choice().
  • NO silent decisions. Every step prints a clear log line.
  • NO parameter-specific code. Parameter name injected from sensors_config.json.
  • Chronological split only — never random.

PIPELINE ENTRY POINTS
----------------------
  onboard_new_parameter(cfg, df, parameter_name, ...)
      → OnboardingResult containing report_raw, report_log (or None),
        cv_signal, log_evaluated, and recommended variant.
        Does NOT freeze anything.

  submit_benchmark_choice(report, chosen_rank, models_store_path, X_all, y_all)
      → Human-confirmed freeze. Accepts report_raw OR report_log from an
        OnboardingResult — the caller decides which report to submit from.

  run_manual_benchmark(...)
      → Alias for onboard_new_parameter (re-evaluation context).
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.split import chronological_split
from src.forecasting.feature_engineering_generic import build_features_time_aware
from src.forecasting.frequency_detector import detect_frequency
from src.pipeline.model_benchmark import (
    BenchmarkReport,
    LOG_CV_THRESHOLD,
    benchmark_models,
    compute_signal_cv,
    format_benchmark_report,
    quick_noise_diagnostic,
    rerank_log_report_to_original_scale,
)
from src.pipeline.timestamp_detector import detect_timestamp_column, parse_timestamp_column
from src.retraining.model_versioning import (
    save_model_version,
    write_current_model_pointer,
)

logger = logging.getLogger(__name__)

# ── Return type for onboard_new_parameter ─────────────────────────────────────

@dataclass
class OnboardingResult:
    """
    Full output of onboard_new_parameter().

    Contains both the raw and (if applicable) the log-transformed benchmark
    reports so that the human reviewer can choose between variants before
    calling submit_benchmark_choice().

    Fields
    ------
    parameter_name : Sensor parameter name.
    cv_signal      : Signal coefficient of variation (%) of the raw target.
    log_evaluated  : True if CV exceeded LOG_CV_THRESHOLD and a log benchmark
                     was run. False if only the raw benchmark was run.
    report_raw     : BenchmarkReport on the raw target — always present.
    report_log     : BenchmarkReport on log1p(target), back-transformed to
                     original scale — present only when log_evaluated=True.
    recommended    : "raw" or "log" — whichever variant has a lower rank-1
                     original-scale Val RMSE. Advisory only; human decides.
    """
    parameter_name: str
    cv_signal:      float
    log_evaluated:  bool
    report_raw:     BenchmarkReport
    report_log:     BenchmarkReport | None
    recommended:    str           # "raw" | "log"

    @property
    def best_report(self) -> BenchmarkReport:
        """Return the recommended variant's BenchmarkReport."""
        if self.recommended == "log" and self.report_log is not None:
            return self.report_log
        return self.report_raw

    @property
    def best(self):
        """Shortcut: best ModelResult from the recommended report."""
        return self.best_report.best


# ── Sensor config loader ───────────────────────────────────────────────────────

def load_sensor_config(config_path: str | Path) -> dict:
    """Load and return the parsed sensors_config.json."""
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def get_sensor_entry(config: dict, parameter_name: str) -> dict:
    """
    Return the sensors[] entry for the requested parameter_name.

    Raises ValueError if the parameter is not listed in sensors_config.json.
    """
    for entry in config["sensors"]:
        if entry["parameter_name"] == parameter_name:
            return entry
    available = [e["parameter_name"] for e in config["sensors"]]
    raise ValueError(
        f"Parameter '{parameter_name}' not found in sensors_config.json.\n"
        f"Available: {available}\n"
        f"Add an entry to sensors_config.json to onboard a new parameter."
    )


# ── Core pipeline steps ────────────────────────────────────────────────────────

def _prepare_data(
    config: dict,
    df: pd.DataFrame,
    sensor: dict,
) -> tuple[pd.DataFrame, Any]:
    """
    Detect timestamp column, parse it, detect measurement frequency.

    Returns the cleaned DataFrame (sorted, datetime column) and the
    detected timedelta frequency.
    """
    param    = sensor["parameter_name"]
    col_name = sensor["column_name"]

    if col_name not in df.columns:
        raise ValueError(
            f"[{param}] Expected column '{col_name}' not found in DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )

    candidates = config["timestamp_column_candidates"]
    ts_col = detect_timestamp_column(df, candidates)
    logger.info("[orchestrator] [%s] Timestamp column detected: '%s'", param, ts_col)

    df        = parse_timestamp_column(df, ts_col)
    frequency = detect_frequency(df, date_col=ts_col)
    freq_hours = frequency.total_seconds() / 3600.0
    logger.info("[orchestrator] [%s] Measurement frequency: %.1f h", param, freq_hours)

    return df, frequency


def _build_and_split(
    df: pd.DataFrame,
    sensor: dict,
    frequency: Any,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Feature engineering → chronological 70/15/15 split.

    The sensor dict's "parameter_name" field is used as the target column.
    Pass a modified sensor dict (with a different "parameter_name" and
    "column_name") to build features for a log-transformed target.

    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    """
    param         = sensor["parameter_name"]
    lag_hours     = sensor.get("lag_hours",     [24, 48, 72])
    rolling_hours = sensor.get("rolling_hours", [72, 168])
    co_variables  = sensor.get("co_variables")

    logger.info("[orchestrator] [%s] Building features ...", param)
    X, y = build_features_time_aware(
        df            = df,
        target        = param,
        frequency     = frequency,
        lag_hours     = lag_hours,
        rolling_hours = rolling_hours,
        co_variables  = co_variables,
    )

    logger.info("[orchestrator] [%s] Splitting chronologically ...", param)
    return chronological_split(X, y)


# ── Public entry points ────────────────────────────────────────────────────────

def onboard_new_parameter(
    config: dict,
    df: pd.DataFrame,
    parameter_name: str,
    models_store_path: str | Path | None = None,
) -> OnboardingResult:
    """
    Run the full onboarding pipeline for a sensor parameter.

    Steps
    -----
    1.  Validate the sensor entry in sensors_config.json.
    2.  Detect timestamp column and measurement frequency.
    3.  Run quick noise diagnostic. Compute signal CV.
    4.  Build time-aware features (lag, rolling, co-variables) on RAW target.
    5.  Chronological 70/15/15 split.
    6.  Benchmark RF / XGBoost / SVR on the RAW target.
    6b. If signal CV > LOG_CV_THRESHOLD (default 60%):
          • Build features on log1p(target).
          • Benchmark RF / XGBoost / SVR on the log target.
          • Re-rank log models by original-scale RMSE (via expm1).
          • Print both reports side by side.
    7.  Return OnboardingResult — no model is saved.

    This function does NOT select or save any model. The caller must read
    the printed report(s), choose a variant (raw vs log) and a rank, then
    call submit_benchmark_choice().

    Parameters
    ----------
    config            : Parsed sensors_config.json.
    df                : Raw DataFrame — will be copied, not modified.
    parameter_name    : Must match a "parameter_name" field in sensors_config.
    models_store_path : Accepted for call-site symmetry with
                        submit_benchmark_choice(); unused by this step.

    Returns
    -------
    OnboardingResult
    """
    sensor = get_sensor_entry(config, parameter_name)
    unit   = sensor.get("unit", "")

    logger.info("=" * 62)
    logger.info("  ONBOARDING PIPELINE — %s  (%s)", parameter_name, unit)
    logger.info("=" * 62)

    df, frequency = _prepare_data(config, df.copy(), sensor)

    # ── Noise diagnostic + CV check ────────────────────────────────────────
    noise_diag = quick_noise_diagnostic(df, sensor["column_name"])
    logger.info("[orchestrator] [%s] Quick noise diagnostic:\n%s", parameter_name, noise_diag)

    cv_signal     = compute_signal_cv(df, sensor["column_name"])
    log_evaluated = cv_signal > LOG_CV_THRESHOLD

    # ── Raw benchmark ──────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = _build_and_split(
        df, sensor, frequency
    )

    logger.info("[orchestrator] [%s] Running RAW benchmark ...", parameter_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_raw = benchmark_models(
            X_train, y_train, X_val, y_val,
            parameter_name=parameter_name,
            unit=unit,
        )

    logger.info("\n%s", format_benchmark_report(report_raw))

    # ── Log benchmark (automatic when CV > threshold) ──────────────────────
    report_log: BenchmarkReport | None = None

    if log_evaluated:
        col_name = sensor["column_name"]
        col_log  = col_name + "_log"

        logger.info(
            "[orchestrator] [%s] Signal CV=%.1f%% > %.0f%% → auto-running LOG benchmark ...",
            parameter_name, cv_signal, LOG_CV_THRESHOLD,
        )

        df_log       = df.copy()
        df_log[col_log] = np.log1p(df_log[col_name])

        sensor_log = dict(sensor)
        sensor_log["column_name"]    = col_log
        sensor_log["parameter_name"] = col_log

        Xtr_l, ytr_l, Xv_l, yv_l, Xt_l, yt_l = _build_and_split(
            df_log, sensor_log, frequency
        )
        y_val_orig = np.expm1(yv_l.values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report_log_scale = benchmark_models(
                Xtr_l, ytr_l, Xv_l, yv_l,
                parameter_name=col_log,
                unit=f"log-{unit}",
            )

        report_log = rerank_log_report_to_original_scale(
            report_log_scale, Xv_l, y_val_orig, unit
        )

        logger.info(
            "[orchestrator] [%s] LOG benchmark (back-transformed to %s):\n%s",
            parameter_name, unit, format_benchmark_report(report_log),
        )

        # Head-to-head summary
        raw_r1  = report_raw.best.val_rmse
        log_r1  = report_log.best.val_rmse
        winner  = "log" if log_r1 < raw_r1 else "raw"
        pct     = abs(raw_r1 - log_r1) / raw_r1 * 100
        logger.info(
            "[orchestrator] [%s] Raw rank-1: %.4f %s  |  Log rank-1: %.4f %s  →  %s wins by %.1f%%",
            parameter_name, raw_r1, unit, log_r1, unit, winner, pct,
        )
    else:
        logger.info(
            "[orchestrator] [%s] Signal CV=%.1f%% ≤ %.0f%% → log benchmark skipped.",
            parameter_name, cv_signal, LOG_CV_THRESHOLD,
        )

    # ── Recommended variant ────────────────────────────────────────────────
    if log_evaluated and report_log is not None:
        recommended = "log" if report_log.best.val_rmse < report_raw.best.val_rmse else "raw"
    else:
        recommended = "raw"

    logger.info(
        "[orchestrator] [%s] ⚠  Benchmark complete — NO model saved yet.\n"
        "  Recommended variant: %s  (advisory — human decides).\n"
        "  Call submit_benchmark_choice(result.report_%s, "
        "chosen_rank=N, models_store_path=..., X_all=..., y_all=...) to proceed.",
        parameter_name, recommended.upper(), recommended,
    )

    return OnboardingResult(
        parameter_name = parameter_name,
        cv_signal      = cv_signal,
        log_evaluated  = log_evaluated,
        report_raw     = report_raw,
        report_log     = report_log,
        recommended    = recommended,
    )


# run_manual_benchmark is the same workflow as onboard_new_parameter.
# Renamed entry point for re-evaluation context (e.g. after data drift).
run_manual_benchmark = onboard_new_parameter


def submit_benchmark_choice(
    report: BenchmarkReport,
    chosen_rank: int,
    models_store_path: str | Path,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    pending_dir: str | Path | None = None,
    max_versions: int = 3,
) -> dict:
    """
    Freeze the human-chosen model to the versioned store.

    This is the ONLY path that writes a model file. Must be called
    explicitly by a human who has read the benchmark report and chosen
    a rank and a variant.

    Parameters
    ----------
    report      : BenchmarkReport from OnboardingResult.report_raw OR
                  OnboardingResult.report_log — caller's choice.
    chosen_rank : Integer rank (1 = best Val RMSE). Must be valid.
    models_store_path : Directory for versioned pkl files.
    X_all       : Feature matrix for the final refit (train + val).
                  Must be from the SAME variant (raw or log) as report.
    y_all       : Target aligned with X_all. Same variant as report.
                  For log models: pass the log-transformed y (not original),
                  since the model was trained in log space.
    pending_dir : Reserved; unused.
    max_versions: Checkpoint files to retain per model (default 3).

    Note on y_all for log models
    ----------------------------
    When submitting a LOG-variant model, y_all must be log1p(target) —
    NOT the original-scale values. The model predicts in log space; the
    final refit must use the same scale. Inference output is then
    post-processed via expm1() at the API layer.

    Returns
    -------
    dict with keys:
        parameter_name, chosen_rank, algorithm, hyperparams,
        val_rmse, saved_path, pointer_path, timestamp.
    """
    n_results = len(report.results)
    if not (1 <= chosen_rank <= n_results):
        raise ValueError(
            f"chosen_rank={chosen_rank} is out of range. "
            f"Valid range: 1–{n_results}."
        )

    chosen = report.results[chosen_rank - 1]
    param  = report.parameter_name.removesuffix("_log")   # strip log suffix for display

    logger.info(
        "[orchestrator] [%s] Freezing rank %d — %s  hyperparams=%s  "
        "val_rmse=%.4f %s  refitting on %d rows ...",
        param, chosen_rank, chosen.algorithm, chosen.hyperparams,
        chosen.val_rmse, report.unit, len(X_all),
    )

    chosen.fitted_model.fit(X_all, y_all)

    store      = Path(models_store_path)
    saved_path = save_model_version(
        model             = chosen.fitted_model,
        models_store_path = store,
        max_versions      = max_versions,
    )

    approval_id = (
        f"{param}_{chosen.fitted_model.model_version}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_benchmark_freeze"
    )
    pointer_path = write_current_model_pointer(
        models_store_path = store,
        model_pkl_path    = saved_path,
        approval_id       = approval_id,
        parameter_name    = param,
        model_version     = chosen.fitted_model.model_version,
    )

    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        "[orchestrator] [%s] Model frozen.  saved=%s  pointer=%s  approval_id=%s",
        param, saved_path, pointer_path, approval_id,
    )

    return {
        "parameter_name": param,
        "chosen_rank":    chosen_rank,
        "algorithm":      chosen.algorithm,
        "hyperparams":    chosen.hyperparams,
        "val_rmse":       chosen.val_rmse,
        "saved_path":     str(saved_path),
        "pointer_path":   str(pointer_path),
        "timestamp":      ts,
    }
