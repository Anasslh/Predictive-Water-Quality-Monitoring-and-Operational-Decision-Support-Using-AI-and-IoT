#!/usr/bin/env python3
"""
run.py — Unified CLI entry point for the water-quality monitoring pipeline.

Sub-commands
------------
  onboard   Benchmark RF / XGBoost / SVR for a new sensor parameter.
            Single-parameter mode (--parameter): interactive, freezes model immediately.
            Batch mode (--all): benchmarks every parameter in sensors_config.json
            without interruption, saves pending reports, never freezes automatically.
  review    Load a pending benchmark report (saved by 'onboard --all') and run
            the interactive rank-selection / model-freeze workflow for one parameter.
  monitor   Pass one new measurement through the production model:
            predict, SHAP explanation, anomaly score, rolling JSONL export.
  status    Show current operational state for one or all parameters.
  approve   Review / approve / reject pending model promotion candidates.

Config validation
-----------------
Before any pipeline module is imported, run.py checks that
  • config/system_config.json  exists, is valid JSON, has required fields
  • config/sensors_config.json exists, is valid JSON, has required fields
A clear human-readable error is printed and the process exits if anything
is wrong — no cryptic traceback further down the pipeline.

Default paths follow the standard project layout and can all be overridden
via command-line arguments.

Usage examples
--------------
  # Single-parameter onboard (interactive, freezes immediately)
  python run.py onboard --dataset data/processed/c1_with_wqi.csv \\
                        --parameter Turbidity

  # Batch onboard (runs all sensors, saves reports, NO interactive freeze)
  python run.py onboard --all --dataset data/processed/c1_with_wqi.csv

  # Review a pending benchmark report and choose a rank to freeze
  python run.py review EC
  python run.py review pH

  python run.py monitor --parameter EC \\
                        --new-row data/processed/ec_X_test.csv

  python run.py status
  python run.py status --parameter EC

  python run.py approve
  python run.py approve <approval_id>
  python run.py approve --reject-all-stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Project root (this file lives at the project root) ────────────────────────
ROOT = Path(__file__).resolve().parent

# Default config paths
_DEFAULT_SYSTEM_CONFIG  = ROOT / "config" / "system_config.json"
_DEFAULT_SENSORS_CONFIG = ROOT / "config" / "sensors_config.json"
_DEFAULT_MODELS_STORE   = ROOT / "models_store"


# ══════════════════════════════════════════════════════════════════════════════
# Config validation — runs BEFORE any src import
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_CONFIG_REQUIRED_FIELDS  = ("retraining", "anomaly_detection",
                                    "forecasting", "exports", "logging")
_SENSORS_CONFIG_REQUIRED_FIELDS = ("sensors", "timestamp_column_candidates")


def _validate_configs(
    system_config_path: Path,
    sensors_config_path: Path,
) -> None:
    """
    Validate both config files before any pipeline code runs.

    Checks performed
    ----------------
    1. File exists at the specified path.
    2. File is valid JSON (no parse error).
    3. All required top-level fields are present.

    On any failure: print a clear diagnostic and sys.exit(1).
    """
    errors: list[str] = []

    for path, label, required_fields in (
        (system_config_path,  "system_config.json",  _SYSTEM_CONFIG_REQUIRED_FIELDS),
        (sensors_config_path, "sensors_config.json", _SENSORS_CONFIG_REQUIRED_FIELDS),
    ):
        if not path.exists():
            errors.append(
                f"{label} not found at: {path}\n"
                f"    Expected location: {path}\n"
                f"    Create the file or pass --system-config / --sensors-config "
                f"to point to an alternative location."
            )
            continue   # no point checking fields if file is absent

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{label}: invalid JSON — {exc}")
            continue

        for field in required_fields:
            if field not in raw:
                errors.append(
                    f"{label}: missing required field '{field}'\n"
                    f"    Found fields: {list(raw.keys())}"
                )

        if label == "sensors_config.json":
            sensors = raw.get("sensors", [])
            if not isinstance(sensors, list):
                errors.append("sensors_config.json: field 'sensors' must be a list")
            elif len(sensors) == 0:
                errors.append("sensors_config.json: 'sensors' list is empty — "
                              "add at least one sensor entry")

    if errors:
        print("\n[run.py] Configuration error(s) — cannot proceed:\n")
        for i, err in enumerate(errors, 1):
            for j, line in enumerate(err.splitlines()):
                prefix = f"  {i}. " if j == 0 else "     "
                print(prefix + line)
        print()
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Pending benchmark report helpers (used by --all and review)
# ══════════════════════════════════════════════════════════════════════════════

_PENDING_DIR = "pending_benchmark_reports"


def _pending_dir(param: str, models_store: Path) -> Path:
    return models_store / param.lower() / _PENDING_DIR


def _save_pending_report(
    param: str,
    result: object,
    X_all_raw: object,
    y_all_raw: object,
    X_all_log: object,
    y_all_log: object,
    models_store: Path,
    frequency: object = None,
    sensor_entry: object = None,
    ts_col: str = "Date",
) -> Path:
    """
    Persist a pending benchmark report to disk.

    Saves two files under models_store/<param>/pending_benchmark_reports/:
      <timestamp>.pkl  — full payload (OnboardingResult + X_all/y_all for
                          both variants); the only file needed by 'review'.
      <timestamp>.json — human-readable summary (best RMSE, variant, etc.)
                          for quick inspection without loading the pickle.

    The .pkl format (not JSON) is required because BenchmarkReport contains
    fitted sklearn/XGBoost objects that are not JSON-serialisable.

    frequency / sensor_entry / ts_col are optional metadata saved so that
    'review' can train and save the forecaster config after the model is frozen.
    """
    import json as _json
    import pickle
    from datetime import datetime, timezone

    pdir = _pending_dir(param, models_store)
    pdir.mkdir(parents=True, exist_ok=True)

    ts_str  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pkl_path  = pdir / f"{ts_str}.pkl"
    json_path = pdir / f"{ts_str}.json"

    payload = {
        "result":           result,
        "X_all_raw":        X_all_raw,
        "y_all_raw":        y_all_raw,
        "X_all_log":        X_all_log,
        "y_all_log":        y_all_log,
        "frequency_seconds": (int(frequency.total_seconds()) if frequency is not None else None),
        "sensor_entry":      sensor_entry,
        "ts_col":            ts_col,
    }
    with open(pkl_path, "wb") as _f:
        pickle.dump(payload, _f)

    # Human-readable summary (not used by code — for inspection only)
    summary: dict = {
        "parameter_name":  param,
        "saved_at":        datetime.now(timezone.utc).isoformat(),
        "cv_signal":       result.cv_signal,
        "log_evaluated":   result.log_evaluated,
        "recommended":     result.recommended,
        "best_raw_rmse":   result.report_raw.best.val_rmse,
        "best_raw_unit":   result.report_raw.unit,
        "best_log_rmse":   (result.report_log.best.val_rmse
                            if result.report_log else None),
        "n_models":        len(result.report_raw.results),
        "pkl_file":        pkl_path.name,
    }
    with open(json_path, "w", encoding="utf-8") as _f:
        _json.dump(summary, _f, indent=2)

    return pkl_path


def _load_latest_pending_report(
    param: str,
    models_store: Path,
) -> "tuple[Path, dict] | tuple[None, None]":
    """
    Load the most recent .pkl pending benchmark report for a parameter.

    Files are named <ISO-timestamp>.pkl; lexicographic sort gives the latest.
    Returns (pkl_path, payload_dict) or (None, None) if nothing is pending.
    Only considers files that do NOT start with '_archived_'.
    """
    import pickle

    pdir = _pending_dir(param, models_store)
    if not pdir.exists():
        return None, None

    candidates = sorted(
        f for f in pdir.glob("*.pkl") if not f.name.startswith("_archived_")
    )
    if not candidates:
        return None, None

    latest = candidates[-1]
    with open(latest, "rb") as _f:
        payload = pickle.load(_f)
    return latest, payload


def _archive_pending_report(pkl_path: Path) -> None:
    """
    Mark a pending report as processed by prepending '_archived_' to its name.

    Both the .pkl and any sibling .json summary are renamed so they are
    invisible to future _load_latest_pending_report() calls.
    """
    def _rename(p: Path) -> None:
        dst = p.parent / ("_archived_" + p.name)
        p.rename(dst)

    _rename(pkl_path)
    json_path = pkl_path.with_suffix(".json")
    if json_path.exists():
        _rename(json_path)


def _train_and_save_forecaster(
    model: object,
    X_all: object,
    y_all: object,
    sensor: dict,
    frequency: object,
    store_path: Path,
    date_col: str = "Date",
) -> None:
    """
    Compute per-step uncertainty bands from the validation tail of X_all, then
    persist a forecaster_config.json alongside the frozen model.

    Called immediately after submit_benchmark_choice() so the forecaster config
    is always calibrated on the same data as the frozen model.

    The forecaster itself is reconstructed at load time from this config + the
    current model file — no separate pickle of the forecaster object is needed.
    """
    import math
    import numpy as np
    from src.forecasting.recursive_forecaster import save_forecaster_config

    target        = sensor["parameter_name"]
    lag_hours     = sensor.get("lag_hours", [24, 48, 72])
    rolling_hours = sensor.get("rolling_hours", [72, 168])
    co_variables  = sensor.get("co_variables") or []

    # Derive n_steps from the sensor's forecast horizon and detected frequency.
    # Default horizon: 72 h (= 3 steps at daily frequency, 72 steps at hourly).
    forecast_horizon_hours = sensor.get("forecast_horizon_hours", 72)
    freq_hours = frequency.total_seconds() / 3600.0
    n_steps = max(1, math.floor(forecast_horizon_hours / freq_hours))

    # Estimate step-1 std from the last 15 % of X_all (approximate val set)
    n         = len(y_all)
    val_size  = max(5, int(n * 0.15))
    X_v_approx = X_all.iloc[-val_size:]
    y_v_approx = y_all.iloc[-val_size:]
    step1_std  = float(np.std(y_v_approx.values - model.predict(X_v_approx)))

    # Propagate uncertainty by sqrt(step) — random-walk approximation
    residual_std_by_step = [round(step1_std * (k + 1) ** 0.5, 6) for k in range(n_steps)]

    cfg_path = save_forecaster_config(
        target               = target,
        frequency            = frequency,
        lag_hours            = lag_hours,
        rolling_hours        = rolling_hours,
        co_variables         = co_variables,
        residual_std_by_step = residual_std_by_step,
        models_store_path    = store_path,
        n_steps              = n_steps,
        date_col             = date_col,
    )
    print(f"[forecast] Forecaster config saved → {cfg_path} "
          f"(horizon={forecast_horizon_hours}h / freq={freq_hours:.0f}h"
          f" → {n_steps} steps, step-1 σ={step1_std:.2f})")


def _train_and_save_anomaly_detector(
    model: object,
    X_all: object,
    y_all: object,
    store_path: Path,
) -> None:
    """
    Train an ECAnomalyDetector on train+val residuals and persist it alongside
    the production model.

    Called immediately after submit_benchmark_choice() so the detector is always
    calibrated against the same data distribution as the frozen model.
    """
    import numpy as np
    from src.anomaly.detector import ECAnomalyDetector, save_anomaly_detector
    from src.config import get_config

    cfg       = get_config()
    residuals = y_all.values - model.predict(X_all)
    detector  = ECAnomalyDetector(
        contamination = cfg.anomaly.isolation_forest_contamination,
        threshold     = cfg.anomaly.score_threshold,
        random_state  = cfg.anomaly.random_state,
    )
    detector.fit(residuals)
    det_path = save_anomaly_detector(detector, store_path)
    print(f"[anomaly] Detector trained on {len(residuals)} residuals → {det_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Sub-command implementations
# ══════════════════════════════════════════════════════════════════════════════

def _cmd_onboard(args: argparse.Namespace) -> None:
    """
    onboard: benchmark + interactive freeze for a single sensor parameter.

    With --all  : dispatches to _cmd_onboard_all() — no interactive prompt.
    With --parameter: single-parameter interactive flow (unchanged behaviour).

    Steps (single-parameter mode)
    ------------------------------
    1. Load the dataset CSV.
    2. Call onboard_new_parameter() → prints benchmark report(s).
    3. Ask the user to choose a variant (raw / log) and a rank.
    4. Rebuild X_all / y_all for the chosen variant (train + val concatenated).
    5. Call submit_benchmark_choice() to freeze the model.
    """
    if getattr(args, "all", False):
        _cmd_onboard_all(args)
        return

    import numpy as np
    import pandas as pd
    from src.pipeline.orchestrator import (
        _build_and_split,
        _prepare_data,
        load_sensor_config,
        onboard_new_parameter,
        submit_benchmark_choice,
    )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"\n[onboard] Dataset not found: {dataset_path}\n")
        sys.exit(1)

    print(f"\n[onboard] Loading dataset:  {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        print(f"[onboard] Failed to load dataset: {exc}\n")
        sys.exit(1)

    print(f"[onboard] Rows: {len(df)}  |  Parameter: {args.parameter}")

    sensors_config = load_sensor_config(args.sensors_config)
    models_store   = Path(args.models_store) / args.parameter.lower()

    with open(args.system_config, encoding="utf-8") as _f:
        _sys_raw = json.load(_f)
    grid_config = _sys_raw.get("model_benchmark")

    result = onboard_new_parameter(
        config            = sensors_config,
        df                = df,
        parameter_name    = args.parameter,
        models_store_path = models_store,
        grid_config       = grid_config,
    )

    # ── Interactive variant + rank selection ───────────────────────────────
    print("\n" + "═" * 62)
    print("  HUMAN REVIEW REQUIRED")
    print("═" * 62)

    if result.log_evaluated:
        print(f"  Two variants evaluated (signal CV={result.cv_signal:.1f}% > 60%):")
        print(f"    raw : rank-1 val RMSE = {result.report_raw.best.val_rmse:.4f}")
        print(f"    log : rank-1 val RMSE = {result.report_log.best.val_rmse:.4f}  "
              f"(back-transformed)")
        print(f"  Recommended: {result.recommended.upper()}")
        print()
        variant_input = (
            input(f"  Choose variant [raw / log]  (default: {result.recommended}): ")
            .strip().lower() or result.recommended
        )
        if variant_input not in ("raw", "log"):
            print(f"  Unknown variant '{variant_input}'. Aborting.")
            sys.exit(1)
    else:
        variant_input = "raw"
        print(f"  Only RAW variant evaluated (CV={result.cv_signal:.1f}% ≤ 60%).")

    chosen_report = result.report_raw if variant_input == "raw" else result.report_log
    n_results     = len(chosen_report.results)
    print(f"\n  Report has {n_results} ranked model(s). Enter the rank to freeze "
          f"(1 = best val RMSE).")
    try:
        rank_input = int(input(f"  Chosen rank [1–{n_results}]: ").strip())
    except ValueError:
        print("  Invalid rank. Aborting.")
        sys.exit(1)

    # ── Rebuild X_all / y_all for the chosen variant ───────────────────────
    sensor_entry = next(
        s for s in sensors_config["sensors"]
        if s["parameter_name"] == args.parameter
    )
    df_clean, frequency, _ts_col_ob = _prepare_data(sensors_config, df.copy(), sensor_entry)

    if variant_input == "log":
        col    = sensor_entry["column_name"]
        col_lg = col + "_log"
        df_log = df_clean.copy()
        df_log[col_lg] = np.log1p(df_log[col])
        sensor_log = {**sensor_entry,
                      "column_name": col_lg, "parameter_name": col_lg}
        X_tr, y_tr, X_v, y_v, *_ = _build_and_split(df_log, sensor_log, frequency,
                                                      date_col=_ts_col_ob)
    else:
        X_tr, y_tr, X_v, y_v, *_ = _build_and_split(df_clean, sensor_entry, frequency,
                                                      date_col=_ts_col_ob)

    X_all = pd.concat([X_tr, X_v]).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_v]).reset_index(drop=True)

    models_store.mkdir(parents=True, exist_ok=True)
    out = submit_benchmark_choice(
        report            = chosen_report,
        chosen_rank       = rank_input,
        models_store_path = models_store,
        X_all             = X_all,
        y_all             = y_all,
    )

    frozen_model = chosen_report.results[rank_input - 1].fitted_model
    _train_and_save_anomaly_detector(frozen_model, X_all, y_all, models_store)
    _train_and_save_forecaster(frozen_model, X_all, y_all, sensor_entry, frequency, models_store,
                               date_col=_ts_col_ob)

    print(f"\n[onboard] Model frozen.")
    print(f"  Algorithm  : {out['algorithm']}")
    print(f"  Val RMSE   : {out['val_rmse']:.4f}")
    print(f"  Saved to   : {out['saved_path']}")
    print(f"  Pointer    : {out['pointer_path']}\n")


def _cmd_onboard_all(args: argparse.Namespace) -> None:
    """
    onboard --all: benchmark every sensor in sensors_config.json without
    any interactive prompt. Saves a pending report for each parameter so
    'python run.py review <param>' can be run separately to freeze each one.

    Nothing is written to production (no current_model.json, no .pkl model
    file) until the human runs 'review'.
    """
    import numpy as np
    import pandas as pd
    from src.pipeline.orchestrator import (
        _build_and_split,
        _prepare_data,
        load_sensor_config,
        onboard_new_parameter,
    )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"\n[onboard --all] Dataset not found: {dataset_path}\n")
        sys.exit(1)

    print(f"\n[onboard --all] Loading dataset: {dataset_path}")
    try:
        df_master = pd.read_csv(dataset_path)
    except Exception as exc:
        print(f"[onboard --all] Failed to load dataset: {exc}\n")
        sys.exit(1)

    sensors_config = load_sensor_config(args.sensors_config)
    sensors        = sensors_config["sensors"]
    models_store   = Path(args.models_store)

    with open(args.system_config, encoding="utf-8") as _f:
        grid_config = json.load(_f).get("model_benchmark")

    SEP2 = "═" * 62
    print(f"\n{SEP2}")
    print(f"  BATCH ONBOARDING — {len(sensors)} parameter(s) from sensors_config")
    print(SEP2)

    # (param, unit, best_rmse_or_None, status_string)
    batch_summary: list[tuple[str, str, float | None, str]] = []

    for sensor in sensors:
        param = sensor["parameter_name"]
        unit  = sensor.get("unit", "")

        print(f"\n  ── [{param}] benchmarking ──")

        try:
            result = onboard_new_parameter(
                config            = sensors_config,
                df                = df_master.copy(),
                parameter_name    = param,
                models_store_path = models_store / param.lower(),
                grid_config       = grid_config,
            )
        except ValueError as exc:
            msg = str(exc).splitlines()[0][:80]
            print(f"  [{param}] ERROR — {msg}")
            batch_summary.append((param, unit, None, f"ERROR: {msg}"))
            continue

        # Build X_all / y_all for BOTH variants so 'review' can freeze either
        df_clean, frequency, _ts_col_oa = _prepare_data(
            sensors_config, df_master.copy(), sensor
        )
        X_tr, y_tr, X_v, y_v, *_ = _build_and_split(df_clean, sensor, frequency,
                                                      date_col=_ts_col_oa)
        X_all_raw = pd.concat([X_tr, X_v]).reset_index(drop=True)
        y_all_raw = pd.concat([y_tr, y_v]).reset_index(drop=True)

        X_all_log = y_all_log = None
        if result.log_evaluated:
            col_name = sensor["column_name"]
            col_log  = col_name + "_log"
            df_log   = df_clean.copy()
            df_log[col_log] = np.log1p(df_log[col_name])
            sensor_log = {**sensor, "column_name": col_log,
                          "parameter_name": col_log}
            Xtr_l, ytr_l, Xv_l, yv_l, *_ = _build_and_split(
                df_log, sensor_log, frequency, date_col=_ts_col_oa
            )
            X_all_log = pd.concat([Xtr_l, Xv_l]).reset_index(drop=True)
            y_all_log = pd.concat([ytr_l, yv_l]).reset_index(drop=True)

        pkl_path = _save_pending_report(
            param         = param,
            result        = result,
            X_all_raw     = X_all_raw,
            y_all_raw     = y_all_raw,
            X_all_log     = X_all_log,
            y_all_log     = y_all_log,
            models_store  = models_store,
            frequency     = frequency,
            sensor_entry  = sensor,
            ts_col        = _ts_col_oa,
        )
        best_rmse = result.best.val_rmse
        print(f"  [{param}] Done.  Best val RMSE = {best_rmse:.2f} {unit}")
        print(f"  [{param}] Report saved → {pkl_path.relative_to(models_store)}")
        batch_summary.append((param, unit, best_rmse, "awaiting review"))

    # ── Batch summary ──────────────────────────────────────────────────────
    col_w = max((len(p) for p, *_ in batch_summary), default=10) + 2
    print(f"\n{SEP2}")
    print(f"  BATCH ONBOARDING COMPLETE — {len(sensors)} parameter(s)")
    print(SEP2)
    for param, unit, best_rmse, status in batch_summary:
        if best_rmse is not None:
            print(f"  {param:<{col_w}} best={best_rmse:.2f} {unit:<10} {status}")
        else:
            print(f"  {param:<{col_w}} {status}")
    print()
    print("  Run 'python run.py review <parameter>' to choose a rank and freeze")
    print("  each model — nothing has been saved to production yet.")
    print(SEP2)
    print()


def _cmd_review(args: argparse.Namespace) -> None:
    """
    review: load the latest pending benchmark report for a parameter and run
    the interactive variant-selection + rank-selection + model-freeze flow.

    After a successful freeze the pending report is archived (renamed with
    an '_archived_' prefix) so it is invisible to future 'review' calls.
    """
    import numpy as np
    import pandas as pd
    from src.pipeline.orchestrator import submit_benchmark_choice
    from src.pipeline.model_benchmark import format_benchmark_report

    param        = args.parameter
    models_store = Path(args.models_store)

    pkl_path, payload = _load_latest_pending_report(param, models_store)

    if payload is None:
        print(f"\n[review] No pending benchmark report found for '{param}'.")
        print(f"  Run 'python run.py onboard --all --dataset <path>' first.\n")
        sys.exit(1)

    result             = payload["result"]
    X_all_raw          = payload["X_all_raw"]
    y_all_raw          = payload["y_all_raw"]
    X_all_log          = payload.get("X_all_log")
    y_all_log          = payload.get("y_all_log")
    _freq_seconds      = payload.get("frequency_seconds")
    _sensor_entry_pkg  = payload.get("sensor_entry")
    _ts_col_rev        = payload.get("ts_col", "Date")

    # ── Display report(s) ──────────────────────────────────────────────────
    print(f"\n[review] Pending report for '{param}': {pkl_path}")
    print()
    print(format_benchmark_report(result.report_raw))
    if result.log_evaluated and result.report_log:
        print()
        print(format_benchmark_report(result.report_log))

    # ── Interactive variant + rank selection (same logic as _cmd_onboard) ──
    print("\n" + "═" * 62)
    print("  HUMAN REVIEW REQUIRED")
    print("═" * 62)

    if result.log_evaluated:
        print(f"  Two variants evaluated (signal CV={result.cv_signal:.1f}% > 60%):")
        print(f"    raw : rank-1 val RMSE = {result.report_raw.best.val_rmse:.4f}")
        print(f"    log : rank-1 val RMSE = {result.report_log.best.val_rmse:.4f}  "
              f"(back-transformed)")
        print(f"  Recommended: {result.recommended.upper()}")
        print()
        variant_input = (
            input(f"  Choose variant [raw / log]  (default: {result.recommended}): ")
            .strip().lower() or result.recommended
        )
        if variant_input not in ("raw", "log"):
            print(f"  Unknown variant '{variant_input}'. Aborting.")
            sys.exit(1)
    else:
        variant_input = "raw"
        print(f"  Only RAW variant evaluated (CV={result.cv_signal:.1f}% ≤ 60%).")

    chosen_report = result.report_raw if variant_input == "raw" else result.report_log
    n_results     = len(chosen_report.results)
    print(f"\n  Report has {n_results} ranked model(s). Enter the rank to freeze "
          f"(1 = best val RMSE).")
    try:
        rank_input = int(input(f"  Chosen rank [1–{n_results}]: ").strip())
    except ValueError:
        print("  Invalid rank. Aborting.")
        sys.exit(1)

    # ── Select X_all / y_all for the chosen variant ────────────────────────
    if variant_input == "log":
        if X_all_log is None or y_all_log is None:
            print("  ERROR: log variant data not found in pending report. Aborting.")
            sys.exit(1)
        X_all, y_all = X_all_log, y_all_log
    else:
        X_all, y_all = X_all_raw, y_all_raw

    store_path = models_store / param.lower()
    store_path.mkdir(parents=True, exist_ok=True)

    out = submit_benchmark_choice(
        report            = chosen_report,
        chosen_rank       = rank_input,
        models_store_path = store_path,
        X_all             = X_all,
        y_all             = y_all,
    )

    frozen_model = chosen_report.results[rank_input - 1].fitted_model
    _train_and_save_anomaly_detector(frozen_model, X_all, y_all, store_path)
    if _freq_seconds is not None and _sensor_entry_pkg is not None:
        from datetime import timedelta as _td
        _train_and_save_forecaster(
            frozen_model, X_all, y_all,
            _sensor_entry_pkg, _td(seconds=_freq_seconds), store_path,
            date_col=_ts_col_rev,
        )
    else:
        print("[forecast] No frequency/sensor metadata in pending report — "
              "run 'onboard' again to generate a forecaster config.")

    # ── Archive the pending report ─────────────────────────────────────────
    _archive_pending_report(pkl_path)

    print(f"\n[review] Model frozen.")
    print(f"  Algorithm  : {out['algorithm']}")
    print(f"  Val RMSE   : {out['val_rmse']:.4f}")
    print(f"  Saved to   : {out['saved_path']}")
    print(f"  Pointer    : {out['pointer_path']}")
    print(f"  Report     : archived → _archived_{pkl_path.name}\n")


def _cmd_monitor(args: argparse.Namespace) -> None:
    """
    monitor: run one raw sensor measurement through the production model.

    Stages active by default
    ------------------------
    1. Predict + SHAP   (always on)
    2. Anomaly detection (on unless --no-anomaly; skipped if no detector trained)
    3. Retrain check    (off unless --check-retrain; requires --actual-value)
    4. Forecast         (on unless --no-forecast; skipped if no forecaster config)

    new-row format
    --------------
    CSV with the raw sensor values for the timestep being processed — the same
    columns as the onboarding dataset (Date, EC, pH, Turbidity, …).  Do NOT
    pre-compute lag/rolling features; this command does that automatically.

    workflow
    --------
    1. Load historical CSV from --dataset (same format as onboarding).
    2. Validate the new raw row (physical bounds check).
    3. Append new row to history, run build_features_time_aware() with the
       sensor's lag/rolling/co-variable config, extract the last feature row.
    4. Run predict + SHAP + anomaly + (optionally) retrain check + forecast.
    5. Atomically append the new raw row to --dataset so history self-enriches.

    Note: --dataset grows by one row after every successful monitor call.
    """
    import os
    import pandas as pd
    from src.retraining.model_versioning import load_current_model, read_current_model_pointer
    from src.monitor import ParameterMonitor, prepare_feature_row_from_raw
    from src.pipeline.orchestrator import load_sensor_config, get_sensor_entry
    from src.data.validation import validate_incoming_data
    from src.config import get_config

    param      = args.parameter
    store_path = Path(args.models_store) / param.lower()
    pointer    = read_current_model_pointer(store_path)

    if pointer is None:
        print(f"\n[monitor] No production model found for '{param}' in {store_path}")
        print(f"  Run 'python run.py onboard --parameter {param}' first.\n")
        sys.exit(1)

    print(f"\n[monitor] Parameter     : {param}")
    print(f"[monitor] Model version : {pointer.get('model_version', '?')}")
    print(f"[monitor] Model path    : {pointer.get('model_path', '?')}")

    model = load_current_model(store_path)
    if model is None:
        print(f"\n[monitor] Could not load model from {store_path}\n")
        sys.exit(1)

    # ── Stage 2: Anomaly detector ──────────────────────────────────────────
    from src.anomaly.detector import load_anomaly_detector
    if not getattr(args, "no_anomaly", False):
        _detector = load_anomaly_detector(store_path)
        if _detector is None:
            print(f"[monitor] Anomaly detection: not yet trained — skipping "
                  f"(run 'onboard' or 'review' to train the detector)")
        else:
            print(f"[monitor] Anomaly detection: active")
    else:
        _detector = None
        print(f"[monitor] Anomaly detection: disabled (--no-anomaly)")

    # ── Stage 4: Forecaster ────────────────────────────────────────────────
    from src.forecasting.recursive_forecaster import load_forecaster
    if not getattr(args, "no_forecast", False):
        _forecaster = load_forecaster(store_path, model)
        if _forecaster is None:
            print(f"[monitor] Forecasting: no config found — skipping "
                  f"(run 'onboard' or 'review' to generate the forecaster config)")
        else:
            _freq_h = _forecaster.frequency.total_seconds() / 3600
            print(f"[monitor] Forecasting: active ({_freq_h:.0f}h interval)")
    else:
        _forecaster = None
        print(f"[monitor] Forecasting: disabled (--no-forecast)")

    # ── Load historical dataset ────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"\n[monitor] --dataset file not found: {dataset_path}\n")
        sys.exit(1)
    df_history = pd.read_csv(dataset_path)
    print(f"[monitor] Historical dataset: {dataset_path}  ({len(df_history)} rows)")

    # ── Load new raw measurement row ───────────────────────────────────────
    row_path = Path(args.new_row)
    if not row_path.exists():
        print(f"\n[monitor] --new-row file not found: {row_path}\n")
        sys.exit(1)
    new_raw_row = pd.read_csv(row_path)
    if len(new_raw_row) > 1:
        print(f"[monitor] --new-row has {len(new_raw_row)} rows; using the last row.")
        new_raw_row = new_raw_row.tail(1).reset_index(drop=True)

    # ── Validate raw measurement (physical bounds) ─────────────────────────
    sensor_config = load_sensor_config(Path(args.sensors_config))
    sensor        = get_sensor_entry(sensor_config, param)
    val_report    = validate_incoming_data(new_raw_row, sensor_config=sensor_config)
    if not val_report.is_clean:
        reasons = "; ".join(
            f"row {r.index}: {', '.join(r.reasons)}"
            for r in val_report.rejected_rows
        )
        print(f"\n[monitor] New measurement rejected by validation — aborting: {reasons}\n")
        sys.exit(1)

    # ── Detect timestamp column once from config candidates ───────────────────
    # Used for timestamp extraction, retrain-check frequency detection, and
    # forecaster window conversion — never hardcoded to "Date".
    from src.pipeline.timestamp_detector import detect_timestamp_column as _detect_ts_col
    _ts_col = _detect_ts_col(df_history, sensor_config["timestamp_column_candidates"])

    # ── Extract measurement timestamp before feature engineering removes it ──
    # prepare_feature_row_from_raw() drops the timestamp column when building
    # lag/rolling features. Capture it here so it can be injected into the
    # result after process_new_measurement(), ensuring the JSONL stores the
    # true measurement date rather than the processing wall-clock time.
    _raw_timestamp = new_raw_row[_ts_col].iloc[0] if _ts_col in new_raw_row.columns else None

    # ── Build feature row from raw history + new raw row ──────────────────
    try:
        feature_row = prepare_feature_row_from_raw(
            df_history, new_raw_row, sensor, sensor_config
        )
    except ValueError as exc:
        print(f"\n[monitor] {exc}\n")
        sys.exit(1)

    print(f"[monitor] Feature columns computed: {list(feature_row.columns)}")

    # ── Prepare updated history (used for forecast context + optional retrain) ─
    # Computed here (before process_new_measurement) so the forecaster sees
    # the most recent measurement in its historical window.
    updated_hist = pd.concat([df_history, new_raw_row], ignore_index=True)

    # ── Stage 3: Retrain check (opt-in only — can be slow) ────────────────
    cfg = get_config()
    _retrain_manager = None
    _all_data        = None

    if getattr(args, "check_retrain", False):
        print(f"[monitor] Retrain check: active (--check-retrain)")
        from src.retraining.retrain_manager import RetrainManager
        from src.forecasting.feature_engineering_generic import build_features_time_aware
        from src.forecasting.frequency_detector import detect_frequency

        _lag_h   = sensor.get("lag_hours",     [24, 48, 72])
        _roll_h  = sensor.get("rolling_hours", [72, 168])
        _covars  = sensor.get("co_variables") or []
        _param   = sensor["parameter_name"]

        _df_hist_ts = df_history.copy()
        _df_hist_ts[_ts_col] = pd.to_datetime(_df_hist_ts[_ts_col], errors="coerce")
        _freq = detect_frequency(_df_hist_ts, date_col=_ts_col)

        def _retrain_feature_fn(df):
            return build_features_time_aware(
                df, target=_param, frequency=_freq,
                lag_hours=_lag_h, rolling_hours=_roll_h,
                co_variables=_covars if _covars else None,
                date_col=_ts_col,
            )

        _retrain_manager = RetrainManager(
            model_class               = type(model),
            feature_fn                = _retrain_feature_fn,
            min_new_rows              = cfg.retraining.min_new_rows,
            tolerance                 = cfg.retraining.tolerance,
            models_store_path         = store_path,
            rejection_alert_threshold = cfg.retraining.rejection_alert_threshold,
            max_history_years         = cfg.retraining.max_history_years,
            rmse_ratio_threshold      = cfg.retraining.rmse_ratio_threshold,
        )
        _retrain_manager.initialize(df_history, model)
        _all_data = updated_hist
    else:
        print(f"[monitor] Retrain check: skipped (add --check-retrain to enable)")

    # ── Build monitor ──────────────────────────────────────────────────────
    monitor = ParameterMonitor(
        parameter_name    = param,
        prediction_model  = model,
        feature_fn        = lambda df: (None, None),
        retrain_manager   = _retrain_manager,
        anomaly_detector  = _detector,
        forecaster        = _forecaster,
        models_store_path = store_path,
        config            = cfg,
    )

    # Provide historical context for forecasting (Stage 4)
    if _forecaster is not None:
        _window = updated_hist.copy()
        _window[_ts_col] = pd.to_datetime(_window[_ts_col], errors="coerce")
        monitor._historical_window = _window

    actual   = float(args.actual_value) if args.actual_value is not None else None
    _n_steps = getattr(_forecaster, "_config_n_steps", None) if _forecaster else None
    result   = monitor.process_new_measurement(
        feature_row,
        all_data       = _all_data,
        actual_value   = actual,
        forecast_steps = _n_steps,
        timestamp      = _raw_timestamp,
    )

    # ── Atomically persist updated historical dataset ──────────────────────
    tmp_path = dataset_path.with_suffix(".tmp")
    updated_hist.to_csv(tmp_path, index=False)
    os.replace(tmp_path, dataset_path)
    print(f"[monitor] Dataset updated: {dataset_path}  ({len(updated_hist)} rows, +1)")

    # ── Display result ─────────────────────────────────────────────────────
    SEP = "─" * 58
    print(f"\n{SEP}")
    print(f"  MONITOR RESULT — {param}")
    print(SEP)
    print(f"  Timestamp       : {result.timestamp}")
    if result.prediction is not None:
        print(f"  Prediction      : {result.prediction:.4f}")
    else:
        print(f"  Prediction      : —")
    if result.residual is not None:
        print(f"  Actual value    : {actual:.4f}")
        print(f"  Residual        : {result.residual:+.4f}")

    # Top-3 SHAP features with direction and magnitude
    if result.prediction_shap:
        print(f"  Top SHAP drivers:")
        for i, entry in enumerate(result.prediction_shap[:3], 1):
            feat  = entry.get("feature", "?")
            val   = entry.get("shap_value", 0.0)
            dirn  = "UP  " if entry.get("direction") == "positive" else "DOWN"
            print(f"    {i}. {feat:<30} pushed prediction {dirn} by {abs(val):.4f}")

    # Anomaly result
    if result.anomaly_detected is not None:
        threshold = cfg.anomaly.score_threshold
        if result.anomaly_detected:
            print(f"  ⚠ ANOMALY DETECTED — score={result.anomaly_score:.3f} "
                  f"(threshold={threshold:.3f})")
            print(f"    residual={result.residual:+.2f}  "
                  f"(model expected {result.prediction:.2f}, got {actual:.2f})")
            if result.anomaly_shap:
                print(f"  What the model relied on (feature value → SHAP contribution):")
                for i, entry in enumerate(result.anomaly_shap[:3], 1):
                    feat     = entry.get("feature", "?")
                    shap_val = entry.get("shap_value", 0.0)
                    dirn     = "▲" if entry.get("direction") == "positive" else "▼"
                    feat_val = feature_row[feat].iloc[0] if feat in feature_row.columns else float("nan")
                    print(f"    {i}. {feat:<28} = {feat_val:>8.2f}   SHAP {dirn} {abs(shap_val):.4f}")
        else:
            print(f"  Anomaly check   : OK  (score={result.anomaly_score:.3f}, "
                  f"threshold={threshold:.3f})")
    elif _detector is None:
        print(f"  Anomaly check   : skipped "
              f"({'--no-anomaly' if getattr(args, 'no_anomaly', False) else 'no detector trained'})")

    # Forecast trajectory
    if result.forecast is not None:
        fc    = result.forecast
        freq_h = fc.frequency.total_seconds() / 3600
        unit   = "D" if freq_h >= 20 else ("H" if freq_h <= 2 else f"{freq_h:.0f}h")
        print(f"  Forecast ({fc.n_steps} steps, {freq_h:.0f}h interval):")
        for k, (pred, std) in enumerate(
            zip(fc.predictions, fc.uncertainty_std), start=1
        ):
            band = f"  ± {std:.2f}" if std is not None else ""
            print(f"    {unit}+{k:<2}  {pred:>10.2f}{band}")
    elif _forecaster is None:
        print(f"  Forecast        : skipped "
              f"({'--no-forecast' if getattr(args, 'no_forecast', False) else 'no config found'})")

    # Retrain check result
    if result.retrain_alert:
        print(f"  Retrain alert   : {result.retrain_alert}")
    elif getattr(args, "check_retrain", False):
        outcome = (result.retrain_result or {}).get("outcome", "volume gate not reached")
        print(f"  Retrain check   : {outcome}")

    print(SEP)
    exports_dir = Path(cfg.exports.exports_dir)
    print(f"  Export appended : {exports_dir / (param + '.jsonl')}")
    print()


def _cmd_status(args: argparse.Namespace) -> None:
    """
    status: print current operational state for one or all parameters.

    Without --parameter: scans models_store/ for every sub-directory that
    contains a current_model.json and prints status for each.  Also surfaces
    any parameter whose benchmark ran but whose model has not been frozen yet
    (pending_benchmark_reports/ present, current_model.json absent), so the
    operator knows to run 'python run.py review <param>'.
    """
    from src.monitor import get_system_status

    store_root = Path(args.models_store)
    SEP  = "─" * 58
    SEP2 = "═" * 58

    if args.parameter:
        params     = [args.parameter]
        store_dirs = [store_root / args.parameter.lower()]
    else:
        if not store_root.exists():
            print(f"\n[status] models_store not found at: {store_root}\n")
            sys.exit(1)
        store_dirs = sorted(
            d for d in store_root.iterdir()
            if d.is_dir() and (d / "current_model.json").exists()
        )
        params = [d.name for d in store_dirs]

    # ── Detect parameters awaiting review (benchmark done, model not frozen) ──
    def _pending_display_name(store_dir: Path) -> str | None:
        """
        Return the original-casing parameter name stored in the pending JSON
        summary, or None if no pending report exists.
        """
        import json as _json_mod
        pdir = store_dir / _PENDING_DIR
        if not pdir.is_dir():
            return None
        jsons = sorted(
            f for f in pdir.glob("*.json")
            if not f.name.startswith("_archived_")
        )
        if jsons:
            try:
                data = _json_mod.loads(jsons[-1].read_text(encoding="utf-8"))
                return data.get("parameter_name") or store_dir.name
            except Exception:
                pass
        # Fall back to the pkl existence check; display dir name
        return store_dir.name if any(pdir.glob("*.pkl")) else None

    if args.parameter:
        sd = store_root / args.parameter.lower()
        display = _pending_display_name(sd)
        pending_params = (
            [display]
            if display is not None and not (sd / "current_model.json").exists()
            else []
        )
    else:
        pending_params = []
        for d in sorted(store_root.iterdir()):
            if not d.is_dir() or (d / "current_model.json").exists():
                continue
            display = _pending_display_name(d)
            if display is not None:
                pending_params.append(display)

    if not params and not pending_params:
        print(f"\n[status] No models found in {store_root}\n")
        print(f"  Run 'python run.py onboard --parameter <name>' to onboard a parameter.\n")
        return

    if params:
        print(f"\n{SEP2}")
        print(f"  SYSTEM STATUS  ({len(params)} parameter(s))")
        print(SEP2)

        for param, store_dir in zip(params, store_dirs):
            s = get_system_status(param.upper() if args.parameter else param, store_dir)
            print(f"\n  Parameter             : {s.parameter_name}")
            print(f"  Model version         : {s.model_version or '— not set'}")
            print(f"  Last promoted         : {s.last_promoted_at or '— never'}")
            print(f"  Consecutive rejections: {s.consecutive_rejections}")
            print(f"  Pending approvals     : {s.pending_approvals}")
            print(f"  Last prediction       : "
                  f"{s.last_prediction:.4f}" if s.last_prediction is not None
                  else "  Last prediction       : — (no measurement processed yet)")
            print(f"  Queried at            : {s.queried_at}")
            print(SEP)

    if pending_params:
        print(f"\n{SEP2}")
        print(f"  AWAITING REVIEW  ({len(pending_params)} parameter(s))")
        print(f"  Benchmark complete — model not frozen yet.")
        print(SEP2)
        for p in pending_params:
            print(f"\n  {p:<20} → python run.py review {p}")
        print()


def _cmd_export_archive(args: argparse.Namespace) -> None:
    """
    export-archive: export a time-bounded slice of the Cold archive to a
    single flat JSONL file for IE Warm-tier bootstrapping.

    Accepts either --months N (N full calendar months back from today)
    or explicit --start-date / --end-date (ISO 8601).
    """
    from datetime import datetime, timezone
    from src.monitor.archive import export_archive_range

    param       = args.parameter
    output_path = Path(args.output)
    archive_dir = args.archive_dir

    if args.months is not None:
        now   = datetime.now(timezone.utc)
        m     = now.month - args.months
        y     = now.year
        while m <= 0:
            m += 12
            y -= 1
        start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
        end_dt   = now
        range_label = f"last {args.months} month(s)"
    else:
        if not args.start_date or not args.end_date:
            print("\n[export-archive] Provide either --months N or both "
                  "--start-date and --end-date.\n")
            sys.exit(1)
        start_dt = datetime.fromisoformat(args.start_date)
        end_dt   = datetime.fromisoformat(args.end_date)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        range_label = f"{args.start_date} → {args.end_date}"

    print(f"\n[export-archive] Parameter : {param}")
    print(f"[export-archive] Range     : {range_label}")
    print(f"[export-archive] Source    : {archive_dir}")
    print(f"[export-archive] Output    : {output_path}")

    try:
        n = export_archive_range(
            parameter_name = param,
            start_date     = start_dt,
            end_date       = end_dt,
            output_path    = output_path,
            archive_dir    = archive_dir,
        )
    except ValueError as exc:
        print(f"\n[export-archive] {exc}\n")
        sys.exit(1)

    size_kb = output_path.stat().st_size / 1024
    print(f"\n[export-archive] Done — {n} records exported ({size_kb:.1f} KB)")
    print(f"[export-archive] File: {output_path.resolve()}\n")


def _cmd_approve(args: argparse.Namespace) -> None:
    """
    approve: delegate entirely to cli_approve.main().

    Forwards all unknown arguments verbatim so every cli_approve option
    (--reject-all-stale, --stale-days, --models-store, approval_id) works
    exactly as documented in src/retraining/cli_approve.py.
    """
    from src.retraining.cli_approve import main as _approve_main

    _approve_main(
        argv         = args.approve_args,
        models_store = Path(args.models_store),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python run.py",
        description = "Water-quality monitoring pipeline — unified entry point.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # Global overrides (accepted by all sub-commands)
    parser.add_argument(
        "--system-config",
        default = str(_DEFAULT_SYSTEM_CONFIG),
        metavar = "PATH",
        help    = f"Path to system_config.json (default: {_DEFAULT_SYSTEM_CONFIG})",
    )
    parser.add_argument(
        "--sensors-config",
        default = str(_DEFAULT_SENSORS_CONFIG),
        metavar = "PATH",
        help    = f"Path to sensors_config.json (default: {_DEFAULT_SENSORS_CONFIG})",
    )
    parser.add_argument(
        "--models-store",
        default = str(_DEFAULT_MODELS_STORE),
        metavar = "PATH",
        help    = f"Root of models_store/ directory (default: {_DEFAULT_MODELS_STORE})",
    )

    subs = parser.add_subparsers(dest="command", metavar="<command>")
    subs.required = True

    # ── onboard ───────────────────────────────────────────────────────────
    p_onboard = subs.add_parser(
        "onboard",
        help        = "Benchmark RF/XGBoost/SVR for one or all sensor parameters.",
        description = (
            "Two modes — exactly one of --parameter / --all is required:\n\n"
            "  --parameter NAME  Single-parameter interactive mode.\n"
            "                    Runs benchmark, then immediately asks for a rank\n"
            "                    and freezes the chosen model.\n\n"
            "  --all             Batch mode: benchmarks every sensor in\n"
            "                    sensors_config.json without any interactive\n"
            "                    prompt. Saves a pending report for each\n"
            "                    parameter. Run 'python run.py review <param>'\n"
            "                    afterwards to choose and freeze each one."
        ),
    )
    p_onboard.add_argument(
        "--dataset", required=True, metavar="PATH",
        help="CSV dataset with Date and sensor columns.",
    )
    _ob_group = p_onboard.add_mutually_exclusive_group(required=True)
    _ob_group.add_argument(
        "--parameter", metavar="NAME",
        help="Parameter to onboard (single-parameter interactive mode).",
    )
    _ob_group.add_argument(
        "--all", dest="all", action="store_true",
        help="Benchmark all sensors in sensors_config.json (batch, non-interactive).",
    )

    # ── review ────────────────────────────────────────────────────────────
    p_review = subs.add_parser(
        "review",
        help        = "Load a pending benchmark report and interactively freeze a model.",
        description = (
            "Loads the most recent pending benchmark report saved by\n"
            "'onboard --all' for the given parameter, displays the full\n"
            "ranking table, and asks you to choose a variant and rank.\n\n"
            "After a successful freeze the pending report is archived so\n"
            "future 'review' calls do not re-process it.\n\n"
            "Nothing is committed to production until you confirm the rank."
        ),
    )
    p_review.add_argument(
        "parameter", metavar="PARAMETER",
        help="Parameter name whose pending report to review (e.g. EC, pH, Turbidity).",
    )

    # ── monitor ───────────────────────────────────────────────────────────
    p_monitor = subs.add_parser(
        "monitor",
        help        = "Process one new raw sensor measurement through the production model.",
        description = (
            "Accepts a RAW sensor measurement (Date + sensor values, same format\n"
            "as the onboarding CSV) — NOT pre-computed features.\n\n"
            "Workflow:\n"
            "  1. Validate the new raw row (physical bounds).\n"
            "  2. Append it to --dataset, rebuild lag/rolling features.\n"
            "  3. Run predict + SHAP + anomaly detection + JSONL export.\n"
            "  4. Write the new row back to --dataset (atomic, +1 row).\n\n"
            "Use data/processed/ec_new_row_example.csv as a format reference.\n"
            "Note: --dataset grows by one row after every successful call."
        ),
    )
    p_monitor.add_argument(
        "--parameter", required=True, metavar="NAME",
        help="Parameter name (e.g. EC, pH, Turbidity).",
    )
    p_monitor.add_argument(
        "--dataset", required=True, metavar="PATH",
        help=(
            "Path to the historical CSV (same format as the onboarding dataset). "
            "Provides the context window for lag/rolling feature computation. "
            "One raw row is appended to this file after each successful monitor call."
        ),
    )
    p_monitor.add_argument(
        "--new-row", required=True, metavar="PATH",
        help=(
            "Path to a single-row CSV with the new raw sensor values "
            "(Date + sensor columns, same format as --dataset)."
        ),
    )
    p_monitor.add_argument(
        "--actual-value", default=None, type=float, metavar="FLOAT",
        help="Actual measured value (optional). Enables residual and anomaly scoring.",
    )
    p_monitor.add_argument(
        "--check-retrain", dest="check_retrain", action="store_true", default=False,
        help=(
            "Enable the retrain-check stage (Stage 3). Loads a RetrainManager, "
            "runs the volume + drift gates, and may trigger a retraining candidate. "
            "Off by default because it can take several seconds. "
            "Requires --actual-value for meaningful drift scoring."
        ),
    )
    p_monitor.add_argument(
        "--no-anomaly", dest="no_anomaly", action="store_true", default=False,
        help="Disable anomaly detection (Stage 2) for this monitor call.",
    )
    p_monitor.add_argument(
        "--no-forecast", dest="no_forecast", action="store_true", default=False,
        help="Disable multi-step forecasting (Stage 4) for this monitor call.",
    )

    # ── status ────────────────────────────────────────────────────────────
    p_status = subs.add_parser(
        "status",
        help        = "Show current operational status for one or all parameters.",
        description = (
            "Without --parameter: scans models_store/ and shows status for every\n"
            "parameter that has a current_model.json."
        ),
    )
    p_status.add_argument(
        "--parameter", default=None, metavar="NAME",
        help="If given, show status for this parameter only.",
    )

    # ── export-archive ────────────────────────────────────────────────────
    p_export = subs.add_parser(
        "export-archive",
        help        = "Export a time-bounded slice of the Cold archive to a single JSONL file.",
        description = (
            "IE team: use this command to bootstrap your Warm tier from the Cold archive.\n\n"
            "Reads across compressed (.jsonl.gz) and plain (.jsonl) monthly files,\n"
            "decompresses in memory, and writes a single flat JSONL to --output.\n\n"
            "Date range: use --months N for the last N calendar months (recommended),\n"
            "or provide explicit --start-date / --end-date (ISO 8601).\n\n"
            "Examples:\n"
            "  python run.py export-archive --parameter EC --months 12 \\\n"
            "      --output warm_tier/EC_last_12m.jsonl\n\n"
            "  python run.py export-archive --parameter pH \\\n"
            "      --start-date 2025-08-01 --end-date 2026-07-31 \\\n"
            "      --output warm_tier/pH_fy2026.jsonl"
        ),
    )
    p_export.add_argument(
        "--parameter", required=True, metavar="NAME",
        help="Parameter name (e.g. EC, pH, Turbidity).",
    )
    _date_group = p_export.add_mutually_exclusive_group(required=True)
    _date_group.add_argument(
        "--months", type=int, metavar="N",
        help="Export the last N full calendar months (start = first day of month N months ago).",
    )
    _date_group.add_argument(
        "--start-date", dest="start_date", metavar="YYYY-MM-DD",
        help="Explicit start date (ISO 8601, inclusive). Requires --end-date.",
    )
    p_export.add_argument(
        "--end-date", dest="end_date", metavar="YYYY-MM-DD",
        help="Explicit end date (ISO 8601, inclusive). Required with --start-date.",
    )
    p_export.add_argument(
        "--output", required=True, metavar="PATH",
        help="Destination JSONL file path (created with its parent directories).",
    )
    p_export.add_argument(
        "--archive-dir", dest="archive_dir", default="archive", metavar="PATH",
        help="Root Cold archive directory (default: archive/).",
    )

    # ── approve ───────────────────────────────────────────────────────────
    p_approve = subs.add_parser(
        "approve",
        help        = "Review / approve / reject pending model candidates.",
        description = (
            "Delegates to src/retraining/cli_approve.py.\n"
            "Without arguments: lists all pending approvals.\n"
            "With an ID:        interactive y/n decision for that approval.\n"
            "--reject-all-stale [--stale-days N]: list approvals older than N days."
        ),
    )
    p_approve.add_argument(
        "approve_args",
        nargs   = argparse.REMAINDER,
        help    = "Arguments forwarded verbatim to cli_approve (approval_id, --reject-all-stale, …).",
    )

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    # ── Bootstrap: ensure project root is on sys.path ─────────────────────
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── Config validation (before any src import) ─────────────────────────
    # export-archive only reads the Cold archive files — no model store,
    # no system config needed.  Skip validation so IE can run it standalone.
    if args.command != "export-archive":
        _validate_configs(
            system_config_path  = Path(args.system_config),
            sensors_config_path = Path(args.sensors_config),
        )
        from src._logging import setup_logging_from_config
        setup_logging_from_config(config_path=Path(args.system_config))

    # ── Dispatch ──────────────────────────────────────────────────────────
    dispatch = {
        "onboard":        _cmd_onboard,
        "review":         _cmd_review,
        "monitor":        _cmd_monitor,
        "status":         _cmd_status,
        "approve":        _cmd_approve,
        "export-archive": _cmd_export_archive,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
