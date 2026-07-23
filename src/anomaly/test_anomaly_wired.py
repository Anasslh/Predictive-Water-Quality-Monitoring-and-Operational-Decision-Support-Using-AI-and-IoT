"""
test_anomaly_wired.py — Verify that the anomaly detector is correctly wired
into ParameterMonitor and produces expected output for extreme residuals.

Scenario
--------
  1. Generate 150 days of synthetic EC data with mean=200, std=15.
  2. Train a RandomForest model on the first 85% of rows (train+val).
  3. Train an ECAnomalyDetector on the train+val residuals.
  4. Build a feature row that the model predicts as ~200 µS/cm.
  5. Pass actual_value=769 → residual ≈ +569 → clearly anomalous.
  6. Assert is_anomaly=True, anomaly_score > 0.9, SHAP provided.

Run:  python src/anomaly/test_anomaly_wired.py
      (also picked up automatically by run_tests.sh)
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── Test helpers ───────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    symbol = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{symbol}  {label}{suffix}")
    _results.append((label, condition))


# ── Synthetic dataset ──────────────────────────────────────────────────────────

def _make_synthetic_df(n: int = 150) -> pd.DataFrame:
    """Daily EC series, mean=200 µS/cm, std=15 (low CV — no log transform)."""
    rng   = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    ec    = (200 + 15 * rng.standard_normal(n)).round(2)
    return pd.DataFrame({"Date": dates, "EC": ec})


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_massive_residual_flags_anomaly() -> None:
    """
    End-to-end test: a measurement with actual_value far from prediction must
    produce is_anomaly=True with a high anomaly score.
    """
    # Suppress logger noise during the test
    for lg in ("src.monitor", "src.forecasting.feature_engineering_generic",
               "src.forecasting.frequency_detector", "src.pipeline.orchestrator"):
        logging.getLogger(lg).setLevel(logging.WARNING)

    from src.forecasting.feature_engineering_generic import build_features_time_aware
    from src.forecasting.frequency_detector import detect_frequency
    from src.data.split import chronological_split
    from src.pipeline.model_benchmark import _GenericRFModel
    from src.anomaly.detector import ECAnomalyDetector
    from src.monitor import ParameterMonitor

    df = _make_synthetic_df(150)

    # ── Feature engineering ────────────────────────────────────────────────
    freq = detect_frequency(df, date_col="Date")
    X, y = build_features_time_aware(
        df, target="EC", frequency=freq,
        lag_hours=[24, 48], rolling_hours=[72], co_variables=[],
    )
    X_tr, y_tr, X_v, y_v, X_te, y_te = chronological_split(X, y)

    X_all = pd.concat([X_tr, X_v]).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_v]).reset_index(drop=True)

    # ── Train prediction model ─────────────────────────────────────────────
    model = _GenericRFModel("EC", "rf_test", n_estimators=50)
    model.fit(X_all, y_all)

    # ── Train anomaly detector on train+val residuals ──────────────────────
    residuals = y_all.values - model.predict(X_all)
    detector  = ECAnomalyDetector(contamination=0.05, threshold=0.5, random_state=42)
    detector.fit(residuals)

    check(
        "Detector fitted without error",
        detector._is_fitted,
    )

    # ── Feature row from test set (model predicts ~200 µS/cm) ─────────────
    feature_row    = X_te.iloc[[-1]].reset_index(drop=True)
    predicted_val  = float(model.predict(feature_row)[0])

    check(
        "Model prediction is within normal range (100–300)",
        100 < predicted_val < 300,
        f"got {predicted_val:.1f}",
    )

    # ── Wire into ParameterMonitor (no models_store_path — export disabled) ─
    with tempfile.TemporaryDirectory(prefix="test_anomaly_wired_") as tmp_str:
        tmp = Path(tmp_str)

        monitor = ParameterMonitor(
            parameter_name    = "EC",
            prediction_model  = model,
            feature_fn        = lambda _df: (None, None),
            anomaly_detector  = detector,
            models_store_path = tmp,
        )

        # actual_value = 769 → massive residual regardless of predicted_val
        actual_massive = 769.0
        result = monitor.process_new_measurement(
            feature_row, actual_value=actual_massive,
        )

    residual_actual = actual_massive - result.prediction

    check(
        "Residual is massive (> 400 µS/cm)",
        abs(residual_actual) > 400,
        f"residual={residual_actual:.1f}",
    )

    check(
        "anomaly_score is not None",
        result.anomaly_score is not None,
        str(result.anomaly_score),
    )

    check(
        "anomaly_score > 0.90 (clear outlier)",
        result.anomaly_score is not None and result.anomaly_score > 0.90,
        f"got {result.anomaly_score}",
    )

    check(
        "anomaly_detected = True",
        result.anomaly_detected is True,
        f"got {result.anomaly_detected}",
    )

    check(
        "SHAP explanation provided (anomaly stage runs explain)",
        bool(result.prediction_shap),
        f"shap={result.prediction_shap}",
    )


def test_normal_measurement_not_flagged() -> None:
    """A residual near zero must not be flagged as an anomaly."""
    from src.forecasting.feature_engineering_generic import build_features_time_aware
    from src.forecasting.frequency_detector import detect_frequency
    from src.data.split import chronological_split
    from src.pipeline.model_benchmark import _GenericRFModel
    from src.anomaly.detector import ECAnomalyDetector
    from src.monitor import ParameterMonitor

    df = _make_synthetic_df(150)
    freq = detect_frequency(df, date_col="Date")
    X, y = build_features_time_aware(
        df, target="EC", frequency=freq,
        lag_hours=[24, 48], rolling_hours=[72], co_variables=[],
    )
    X_tr, y_tr, X_v, y_v, X_te, y_te = chronological_split(X, y)
    X_all = pd.concat([X_tr, X_v]).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_v]).reset_index(drop=True)

    model = _GenericRFModel("EC", "rf_test", n_estimators=50)
    model.fit(X_all, y_all)
    residuals = y_all.values - model.predict(X_all)
    detector  = ECAnomalyDetector(contamination=0.05, threshold=0.5, random_state=42)
    detector.fit(residuals)

    feature_row   = X_te.iloc[[-1]].reset_index(drop=True)
    predicted_val = float(model.predict(feature_row)[0])

    # Pass actual_value ≈ predicted → residual ≈ 0 → clearly normal
    actual_normal = predicted_val + 2.0

    with tempfile.TemporaryDirectory(prefix="test_anomaly_normal_") as tmp_str:
        monitor = ParameterMonitor(
            parameter_name    = "EC",
            prediction_model  = model,
            feature_fn        = lambda _df: (None, None),
            anomaly_detector  = detector,
            models_store_path = Path(tmp_str),
        )
        result = monitor.process_new_measurement(
            feature_row, actual_value=actual_normal,
        )

    check(
        "Normal measurement: anomaly_detected = False",
        result.anomaly_detected is False,
        f"score={result.anomaly_score}",
    )


def test_save_load_detector_roundtrip() -> None:
    """save_anomaly_detector + load_anomaly_detector must preserve scores exactly."""
    from src.anomaly.detector import ECAnomalyDetector, save_anomaly_detector, load_anomaly_detector

    rng       = np.random.default_rng(1)
    residuals = rng.standard_normal(100)
    detector  = ECAnomalyDetector(contamination=0.05, threshold=0.5, random_state=42)
    detector.fit(residuals)

    probe = np.array([0.0, 100.0, -100.0])  # normal, and two extremes
    scores_before = detector.score(probe)

    with tempfile.TemporaryDirectory(prefix="test_anomaly_persist_") as tmp_str:
        tmp = Path(tmp_str)
        saved_path = save_anomaly_detector(detector, tmp)

        check(
            "anomaly_detector.pkl created",
            saved_path.exists() and saved_path.name == "anomaly_detector.pkl",
            str(saved_path),
        )

        loaded = load_anomaly_detector(tmp)

    check(
        "Loaded detector is ECAnomalyDetector",
        isinstance(loaded, ECAnomalyDetector),
        str(type(loaded)),
    )
    scores_after = loaded.score(probe)
    check(
        "Scores identical after round-trip (max diff < 1e-9)",
        np.allclose(scores_before, scores_after, atol=1e-9),
        f"before={scores_before}, after={scores_after}",
    )

    check(
        "load_anomaly_detector returns None when file absent",
        load_anomaly_detector(Path("/tmp/does_not_exist_xyz")) is None,
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  anomaly detection wiring — end-to-end tests")
    print(SEP2)

    for lg in ("src.monitor", "src.forecasting.feature_engineering_generic",
               "src.forecasting.frequency_detector", "src.pipeline.orchestrator",
               "src.retraining.model_versioning"):
        logging.getLogger(lg).setLevel(logging.WARNING)

    test_save_load_detector_roundtrip()
    print(SEP)
    test_normal_measurement_not_flagged()
    print(SEP)
    test_massive_residual_flags_anomaly()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
