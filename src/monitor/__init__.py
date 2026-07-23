"""
monitor/__init__.py — Single-parameter operational monitor for the water-quality pipeline.

Architecture
------------
ParameterMonitor wraps all four pipeline stages for one sensor parameter
(predict → anomaly check → retrain check → forecast) and exposes a single
entry-point: process_new_measurement(new_row).

After each successful call to process_new_measurement(), the result is
automatically appended to a rolling JSONL export file via
src.monitor.export.append_measurement().  Export path and retention window
are read from system_config.json (section "exports").

Design constraints
------------------
- Generic: zero knowledge of EC / pH / Turbidity. All parameter-specific
  behaviour is injected via the ParameterModel, RetrainManager,
  ECAnomalyDetector, and MultiStepForecaster provided at construction.
- Config defaults come from system_config.json via src.config.get_config().
  Explicit constructor arguments always override config.
- Stateless per measurement: each call to process_new_measurement() is
  independent. Historical state lives in the injected components (retrain
  manager, anomaly detector).

Usage (sketch)
--------------
    from src.monitor import ParameterMonitor, build_ec_monitor

    monitor = build_ec_monitor(historical_df)
    result  = monitor.process_new_measurement(new_row_df)

    if result.anomaly_detected:
        print("ANOMALY", result.anomaly_score, result.anomaly_shap)
    if result.retrain_alert:
        print("RETRAIN NEEDED", result.retrain_alert)
    print("Prediction:", result.prediction, result.prediction_shap)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import get_config, SystemConfig
from src.models.base import ParameterModel
from src.pipeline.orchestrator import load_sensor_config, get_sensor_entry
from src.anomaly.detector import ECAnomalyDetector
from src.anomaly.residual import compute_residuals
from src.forecasting.recursive_forecaster import MultiStepForecaster, ForecastResult
from src.retraining.retrain_manager import RetrainManager
from src.retraining.model_versioning import read_current_model_pointer, read_rejection_counter
from src.xai.shap_wrapper import compute_shap_explanation
from src.data.validation import validate_incoming_data as _validate_incoming_data
from src.monitor.export import (
    append_measurement     as _append_measurement,
    write_status_export    as _write_status_export,
    get_dashboard_export_paths,
)

logger = logging.getLogger(__name__)

# Default path for sensors_config.json — same as run.py's _DEFAULT_SENSORS_CONFIG
_DEFAULT_SENSORS_CONFIG = Path(__file__).resolve().parents[2] / "config" / "sensors_config.json"


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class MonitorResult:
    """
    Combined output of process_new_measurement() for one new sensor row.

    All fields are optional at the type level; the ones that are filled
    depend on which pipeline stages ran (some are skipped if prerequisites
    are missing, e.g. anomaly detection needs a fitted detector).

    Attributes
    ----------
    parameter_name      : e.g. "EC", "pH", "Turbidity".
    timestamp           : Timestamp of the processed row (or None if not found).
    prediction          : Model prediction for the new row.
    prediction_shap     : Top-k SHAP features for the prediction.
    residual            : actual - predicted (None if actual not in new_row).
    anomaly_score       : Normalized anomaly score in [0, 1] (None if detector not fitted).
    anomaly_detected    : True if score >= threshold (None if detector not fitted).
    anomaly_shap        : Top-k SHAP features explaining the anomaly (None if not anomalous).
    retrain_needed      : True if volume+drift gates both fired.
    retrain_result      : Full dict from RetrainManager.attempt_retrain() (None if not triggered).
    retrain_alert       : Human-readable alert string (None if no alert this cycle).
    forecast            : ForecastResult for the next N steps (None if forecaster not set).
    processed_at        : UTC timestamp of when this result was produced.
    """
    parameter_name:   str
    timestamp:        Any                 = None
    prediction:       float | None        = None
    prediction_shap:  list[dict]          = field(default_factory=list)
    residual:         float | None        = None
    anomaly_score:    float | None        = None
    anomaly_detected: bool | None         = None
    anomaly_shap:     list[dict] | None   = None
    retrain_needed:   bool                = False
    retrain_result:   dict | None         = None
    retrain_alert:    str | None          = None
    forecast:         ForecastResult | None = None
    processed_at:     datetime            = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class SystemStatus:
    """
    Snapshot of the current operational state for one parameter.

    Returned by ParameterMonitor.get_system_status().
    """
    parameter_name:           str
    model_version:            str | None
    last_promoted_at:         str | None
    model_path:               str | None
    last_train_size:          int | None
    consecutive_rejections:   int
    pending_approvals:        int
    last_prediction:          float | None
    last_anomaly_score:       float | None
    last_anomaly_detected:    bool | None
    queried_at:               str


# ── Core monitor class ─────────────────────────────────────────────────────────

class ParameterMonitor:
    """
    Operational monitor for one sensor parameter.

    Parameters
    ----------
    parameter_name    : Human-readable parameter name (e.g. "EC").
    prediction_model  : Fitted ParameterModel for one-step prediction.
    feature_fn        : Feature engineering function (same used at training).
                        Signature: (df: pd.DataFrame) -> (X, y).
    retrain_manager   : Configured RetrainManager (optional — skip retrain checks
                        if not provided).
    anomaly_detector  : Fitted ECAnomalyDetector (optional — skip anomaly stage
                        if not provided or not yet fitted).
    forecaster        : Fitted MultiStepForecaster (optional).
    models_store_path : Path to the parameter's model store (for status queries).
    config            : SystemConfig. Defaults to get_config().
    top_k_shap        : Number of top SHAP features to return (default 3).
    """

    def __init__(
        self,
        parameter_name:   str,
        prediction_model: ParameterModel,
        feature_fn:       Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
        retrain_manager:  RetrainManager | None       = None,
        anomaly_detector: ECAnomalyDetector | None    = None,
        forecaster:       MultiStepForecaster | None  = None,
        models_store_path:   str | Path | None          = None,
        config:              SystemConfig | None         = None,
        top_k_shap:          int                         = 3,
        sensors_config_path: str | Path | None           = None,
    ) -> None:
        self.parameter_name    = parameter_name
        self.prediction_model  = prediction_model
        self.feature_fn        = feature_fn
        self.retrain_manager   = retrain_manager
        self.anomaly_detector  = anomaly_detector
        self.forecaster        = forecaster
        self.models_store_path = Path(models_store_path) if models_store_path else None
        self.cfg               = config or get_config()
        self.top_k_shap        = top_k_shap

        # Physical unit — read directly from sensors_config.json (single source of truth)
        _sc_path = Path(sensors_config_path) if sensors_config_path else _DEFAULT_SENSORS_CONFIG
        self._unit: str = ""
        if _sc_path.exists():
            try:
                _sc = load_sensor_config(_sc_path)
                self._unit = get_sensor_entry(_sc, self.parameter_name).get("unit", "")
            except ValueError:
                logger.warning(
                    "[%s] not found in %s — unit will be empty in status export",
                    parameter_name, _sc_path,
                )
        else:
            logger.warning(
                "[%s] sensors_config.json not found at %s — unit will be empty in status export",
                parameter_name, _sc_path,
            )

        # Rolling state updated after each measurement
        self._last_prediction:       float | None = None
        self._last_anomaly_score:    float | None = None
        self._last_anomaly_detected: bool | None  = None
        self._historical_window:     pd.DataFrame | None = None  # for forecast context

        logger.info(
            "[%s] ParameterMonitor initialised (anomaly=%s, retrain=%s, forecast=%s)",
            parameter_name,
            anomaly_detector is not None,
            retrain_manager is not None,
            forecaster is not None,
        )

    # ── Main entry point ───────────────────────────────────────────────────────

    def process_new_measurement(
        self,
        new_row: pd.DataFrame,
        all_data: pd.DataFrame | None = None,
        actual_value: float | None    = None,
        forecast_steps: int | None    = None,
        timestamp: Any | None         = None,
    ) -> MonitorResult:
        """
        Run the full pipeline for one new sensor row.

        Parameters
        ----------
        new_row        : Single-row DataFrame containing the new measurement.
                         Must have the same feature columns as the training set
                         (produced by the same feature_fn).
        all_data       : Full historical dataset (raw, not feature-engineered).
                         Required for retrain-trigger check. Can be None to skip.
        actual_value   : The true measured value for this timestep (if available).
                         Used to compute the residual for anomaly scoring.
                         If None, anomaly scoring is skipped.
        forecast_steps : Number of future steps to forecast. If None, uses
                         cfg.forecasting.default_horizon_hours / step duration.
                         Ignored if forecaster is None.
        timestamp      : Explicit measurement timestamp to store in the result and
                         the JSONL export. When provided, overrides any Date/
                         timestamp column detected in new_row (which may be absent
                         when new_row contains pre-computed features rather than
                         raw sensor values). Pass the value extracted from the
                         original raw row before feature engineering.

        Returns
        -------
        MonitorResult
        """
        result = MonitorResult(parameter_name=self.parameter_name)

        # ── Stage 0: Input validation ─────────────────────────────────────────
        # Check physical bounds before any computation. new_row is already
        # feature-engineered, so column-presence checking is skipped (no
        # sensor_config passed). Physical bounds apply to any column whose
        # name matches system_config.json validation.physical_bounds.
        val_report = _validate_incoming_data(new_row)
        if not val_report.is_clean:
            reasons = "; ".join(
                f"row {r.index}: {', '.join(r.reasons)}"
                for r in val_report.rejected_rows
            )
            logger.warning(
                "[%s] Measurement row rejected by Stage 0 validation — skipping pipeline: %s",
                self.parameter_name, reasons,
            )
            return result

        # ── Timestamp ─────────────────────────────────────────────────────────
        # First try to detect a date column in new_row (works when new_row is a
        # raw sensor row). Then let an explicit `timestamp` argument override —
        # this is the correct path when new_row contains pre-computed features
        # and the Date column has been dropped by feature engineering.
        for tc in ("Date", "timestamp", "date", "Time"):
            if tc in new_row.columns:
                result.timestamp = new_row[tc].iloc[0]
                break
        if timestamp is not None:
            result.timestamp = timestamp

        # ── Stage 1: Prediction + SHAP ────────────────────────────────────────
        try:
            result.prediction     = self._predict(new_row)
            result.prediction_shap = self._explain(new_row)
            self._last_prediction  = result.prediction
            logger.info(
                "[%s] Prediction=%.4f  (top feature: %s)",
                self.parameter_name, result.prediction,
                result.prediction_shap[0]["feature"] if result.prediction_shap else "n/a",
            )
        except Exception as exc:
            logger.error("[%s] Prediction failed: %s", self.parameter_name, exc)

        # ── Stage 2: Anomaly detection ────────────────────────────────────────
        if actual_value is not None and result.prediction is not None:
            result.residual = float(actual_value) - result.prediction
            if self.anomaly_detector is not None and self.anomaly_detector._is_fitted:
                try:
                    residuals_1d       = np.array([result.residual])
                    score              = float(self.anomaly_detector.score(residuals_1d)[0])
                    result.anomaly_score    = score
                    result.anomaly_detected = score >= self.cfg.anomaly.score_threshold
                    self._last_anomaly_score    = score
                    self._last_anomaly_detected = result.anomaly_detected

                    if result.anomaly_detected:
                        result.anomaly_shap = self._explain(new_row)
                        logger.warning(
                            "[%s] ANOMALY detected  score=%.3f >= threshold=%.2f  "
                            "residual=%.4f  top_feature=%s",
                            self.parameter_name, score,
                            self.cfg.anomaly.score_threshold, result.residual,
                            result.anomaly_shap[0]["feature"] if result.anomaly_shap else "n/a",
                        )
                    else:
                        logger.info(
                            "[%s] Anomaly score=%.3f (below threshold %.2f)",
                            self.parameter_name, score, self.cfg.anomaly.score_threshold,
                        )
                except Exception as exc:
                    logger.error("[%s] Anomaly scoring failed: %s", self.parameter_name, exc)

        # ── Stage 3: Retrain check ────────────────────────────────────────────
        if all_data is not None and self.retrain_manager is not None:
            try:
                current_size = len(all_data)
                if self.retrain_manager.should_check_retrain(current_size):
                    logger.info(
                        "[%s] Volume gate passed (%d rows) — attempting retrain check",
                        self.parameter_name, current_size,
                    )
                    retrain_result = self.retrain_manager.attempt_retrain(all_data)
                    result.retrain_result  = retrain_result
                    result.retrain_needed  = retrain_result.get("drift_detected", False)
                    result.retrain_alert   = retrain_result.get("alert")

                    if result.retrain_alert:
                        logger.warning(
                            "[%s] %s", self.parameter_name, result.retrain_alert
                        )
                    else:
                        logger.info(
                            "[%s] Retrain outcome: %s",
                            self.parameter_name,
                            retrain_result.get("outcome", "no_action"),
                        )
            except Exception as exc:
                logger.error("[%s] Retrain check failed: %s", self.parameter_name, exc)

        # ── Stage 4: Forecast ─────────────────────────────────────────────────
        if self.forecaster is not None and self._historical_window is not None:
            try:
                n_steps = forecast_steps or self.cfg.forecasting.default_horizon_hours
                result.forecast = self.forecaster.forecast(
                    self._historical_window, n_steps=n_steps
                )
                logger.info(
                    "[%s] Forecast %d steps: first=%.4f",
                    self.parameter_name, n_steps,
                    result.forecast.predictions[0] if result.forecast.predictions else float("nan"),
                )
            except Exception as exc:
                logger.error("[%s] Forecast failed: %s", self.parameter_name, exc)

        # ── Stage 5: Rolling JSONL export ─────────────────────────────────────
        try:
            _append_measurement(
                self.parameter_name,
                result,
                exports_dir    = self.cfg.exports.exports_dir,
                retention_days = self.cfg.exports.retention_days,
            )
        except Exception as exc:
            logger.warning("[%s] Export write failed (non-fatal): %s", self.parameter_name, exc)

        # ── Stage 6: Status JSON (overwritten after every measurement) ─────────
        try:
            _write_status_export(
                self.parameter_name,
                unit              = self._unit,
                models_store_path = self.models_store_path,
                exports_dir       = self.cfg.exports.exports_dir,
                retention_days    = self.cfg.exports.retention_days,
            )
        except Exception as exc:
            logger.warning(
                "[%s] Status export write failed (non-fatal): %s", self.parameter_name, exc
            )

        return result

    # ── System status ──────────────────────────────────────────────────────────

    def get_system_status(self) -> SystemStatus:
        """
        Return a snapshot of the current operational state for this parameter.

        Reads disk state (current_model.json, rejection_counter.json,
        pending_approvals/) — does NOT depend on in-memory manager state,
        so it is safe to call even before process_new_measurement().

        Returns
        -------
        SystemStatus
        """
        model_version    = None
        last_promoted_at = None
        model_path       = None

        if self.models_store_path and self.models_store_path.exists():
            pointer = read_current_model_pointer(self.models_store_path)
            if pointer:
                model_version    = pointer.get("model_version")
                last_promoted_at = pointer.get("promoted_at")
                model_path       = pointer.get("model_path")

            consecutive_rejections = read_rejection_counter(self.models_store_path)

            approval_dir = self.models_store_path / "pending_approvals"
            pending_approvals = (
                len(list(approval_dir.glob("*.json")))
                if approval_dir.exists() else 0
            )
        else:
            consecutive_rejections = (
                self.retrain_manager.consecutive_rejections
                if self.retrain_manager else 0
            )
            pending_approvals = 0

        last_train_size = (
            self.retrain_manager.last_train_size
            if self.retrain_manager else None
        )

        return SystemStatus(
            parameter_name          = self.parameter_name,
            model_version           = model_version,
            last_promoted_at        = last_promoted_at,
            model_path              = model_path,
            last_train_size         = last_train_size,
            consecutive_rejections  = consecutive_rejections,
            pending_approvals       = pending_approvals,
            last_prediction         = self._last_prediction,
            last_anomaly_score      = self._last_anomaly_score,
            last_anomaly_detected   = self._last_anomaly_detected,
            queried_at              = datetime.now(timezone.utc).isoformat(),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _predict(self, X_row: pd.DataFrame) -> float:
        return float(self.prediction_model.predict(X_row)[0])

    def _explain(self, X_row: pd.DataFrame) -> list[dict]:
        try:
            return compute_shap_explanation(
                self.prediction_model, X_row, top_k=self.top_k_shap
            )
        except Exception as exc:
            logger.warning("[%s] SHAP explanation failed: %s", self.parameter_name, exc)
            return []


# ── Convenience status query (no monitor instance needed) ──────────────────────

def prepare_feature_row_from_raw(
    df_history: pd.DataFrame,
    new_raw_row: pd.DataFrame,
    sensor: dict,
    sensor_config: dict,
) -> pd.DataFrame:
    """
    Append a new raw measurement to the historical dataset and return the
    last feature row ready for model.predict().

    This is the entry point for the IoT use-case where a sensor station
    delivers raw values (Date, EC, pH, …) rather than a pre-computed feature
    vector.  The function replicates the exact same feature engineering used
    at training time so the column set is guaranteed to match the frozen model.

    Parameters
    ----------
    df_history  : Historical dataset in the same column format as the
                  onboarding CSV (raw values, not feature-engineered).
                  Must contain at least max(lag_steps, rolling_rows) rows
                  so that the last feature row can be computed without NaN.
    new_raw_row : Single-row DataFrame with the new raw sensor values
                  (same columns as df_history; Date column required).
    sensor      : sensors_config.json entry for the parameter being monitored
                  (output of get_sensor_entry()).
    sensor_config : Full parsed sensors_config dict (for
                    timestamp_column_candidates).

    Returns
    -------
    pd.DataFrame — single-row feature DataFrame whose columns match those
                   used when the frozen model was trained.

    Raises
    ------
    ValueError  if there are not enough rows in the combined dataset to
                compute the required lag / rolling features — e.g. at the
                very beginning of a deployment when fewer than
                max(lag_steps, rolling_rows) historical measurements exist.
    """
    import math
    from src.forecasting.feature_engineering_generic import build_features_time_aware
    from src.forecasting.frequency_detector import detect_frequency
    from src.pipeline.timestamp_detector import detect_timestamp_column, parse_timestamp_column

    param         = sensor["parameter_name"]
    lag_hours     = sensor.get("lag_hours",     [24, 48, 72])
    rolling_hours = sensor.get("rolling_hours", [72, 168])
    co_variables  = sensor.get("co_variables") or []
    candidates    = sensor_config.get(
        "timestamp_column_candidates", ["Date", "timestamp", "date", "Time"]
    )

    # Parse timestamps in history (detect_timestamp_column enforces monotone)
    ts_col     = detect_timestamp_column(df_history, candidates)
    df_history = parse_timestamp_column(df_history, ts_col)

    # Parse timestamp in the new row using the same column name
    new_raw_row = new_raw_row.copy()
    if ts_col in new_raw_row.columns:
        new_raw_row[ts_col] = pd.to_datetime(new_raw_row[ts_col])

    # Detect frequency from history only (before appending the new row)
    frequency = detect_frequency(df_history, date_col=ts_col)

    # Append new row and sort (guard against out-of-order delivery)
    df_combined = (
        pd.concat([df_history, new_raw_row], ignore_index=True)
        .sort_values(ts_col)
        .reset_index(drop=True)
    )

    # Build features — same call as _build_and_split() in orchestrator.py
    X, _ = build_features_time_aware(
        df_combined,
        param,
        frequency,
        lag_hours     = lag_hours,
        rolling_hours = rolling_hours,
        co_variables  = co_variables if co_variables else None,
    )

    if len(X) == 0:
        freq_h    = frequency.total_seconds() / 3600.0
        max_lag   = max(max(1, math.floor(h / freq_h)) for h in lag_hours)
        max_roll  = max(max(1, math.floor(h / freq_h)) for h in rolling_hours)
        min_rows  = max(max_lag, max_roll) + 1
        raise ValueError(
            f"[{param}] Not enough history to compute features: "
            f"{len(df_combined)} row(s) available after appending the new "
            f"measurement, but {min_rows} are required "
            f"(max rolling window = {max_roll} rows at "
            f"{freq_h:.0f} h frequency). "
            f"Collect more historical data before running monitor."
        )

    return X.tail(1).reset_index(drop=True)


def get_system_status(
    parameter_name: str,
    models_store_path: str | Path,
) -> SystemStatus:
    """
    Query system status for a parameter without instantiating a full monitor.

    Parameters
    ----------
    parameter_name     : e.g. "EC", "pH", "Turbidity".
    models_store_path  : Directory containing current_model.json and
                         rejection_counter.json for this parameter.

    Returns
    -------
    SystemStatus
    """
    store = Path(models_store_path)
    pointer              = read_current_model_pointer(store) if store.exists() else None
    consecutive_rejections = read_rejection_counter(store) if store.exists() else 0
    approval_dir         = store / "pending_approvals"
    pending_approvals    = len(list(approval_dir.glob("*.json"))) if approval_dir.exists() else 0

    return SystemStatus(
        parameter_name         = parameter_name,
        model_version          = pointer.get("model_version")   if pointer else None,
        last_promoted_at       = pointer.get("promoted_at")     if pointer else None,
        model_path             = pointer.get("model_path")      if pointer else None,
        last_train_size        = None,
        consecutive_rejections = consecutive_rejections,
        pending_approvals      = pending_approvals,
        last_prediction        = None,
        last_anomaly_score     = None,
        last_anomaly_detected  = None,
        queried_at             = datetime.now(timezone.utc).isoformat(),
    )
