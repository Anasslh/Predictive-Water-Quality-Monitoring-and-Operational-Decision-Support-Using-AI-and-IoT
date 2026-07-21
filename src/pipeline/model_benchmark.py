"""
model_benchmark.py — Generic multi-model benchmark for any sensor parameter.

DESIGN PRINCIPLE — NO SILENT SELECTION
---------------------------------------
benchmark_models() trains and evaluates several algorithms over a restricted
hyperparameter grid, then returns a BenchmarkReport containing the full
ranking. It does NOT select or promote any model automatically. The caller
(orchestrator.py → human reviewer) decides which model to submit for approval.

ALGORITHMS BENCHMARKED
-----------------------
  RF      — RandomForestRegressor. Tree-based ensemble, robust to outliers,
             no feature scaling needed.
  XGBoost — XGBRegressor. Gradient-boosted trees, strong on tabular data with
             lag features. Same algorithm family as the reference pipeline.
  SVR     — Support Vector Regression (RBF kernel). Scales X and y internally.
             Historically dominant on water quality indices (Kazapoe et al. 2024).

HYPERPARAMETER GRID (restricted, validated on the reference pipeline)
----------------------------------------------------------------------
  RF:      n_estimators ∈ {50, 100, 200}  ×  max_depth ∈ {3, 5, None}
  XGBoost: max_depth ∈ {2, 3, 5}  ×  n_estimators ∈ {50, 100}
           × learning_rate ∈ {0.01, 0.05, 0.1}
  SVR:     C ∈ {1, 10, 100}  ×  epsilon ∈ {0.1, 1.0}

All models are evaluated on the validation set (not cross-validated) to remain
consistent with the rest of the pipeline's evaluation methodology.

GENERIC MODEL CLASSES
---------------------
_GenericXGBoostModel, _GenericRFModel, _GenericSVRModel are private subclasses
of ParameterModel. They accept parameter_name and model_version as constructor
arguments so they can be used for ANY sensor without hardcoding. They are
exported as pickle objects by submit_benchmark_choice() in orchestrator.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.models.base import ParameterModel


# ── Generic ParameterModel subclasses (parameter-agnostic) ────────────────────

class _GenericXGBoostModel(ParameterModel):
    """XGBoost wrapper usable for any sensor parameter."""

    def __init__(self, parameter_name: str, model_version: str, **xgb_kwargs):
        super().__init__(parameter_name=parameter_name, model_version=model_version)
        self.model = XGBRegressor(random_state=42, verbosity=0, **xgb_kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_fitted()
        return self.model.predict(X)

    def explain(self, X: pd.DataFrame) -> Any:
        self.check_fitted()
        return self.model


class _GenericRFModel(ParameterModel):
    """RandomForest wrapper usable for any sensor parameter."""

    def __init__(self, parameter_name: str, model_version: str, **rf_kwargs):
        super().__init__(parameter_name=parameter_name, model_version=model_version)
        self.model = RandomForestRegressor(random_state=42, **rf_kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_fitted()
        return self.model.predict(X)

    def explain(self, X: pd.DataFrame) -> Any:
        self.check_fitted()
        return self.model


class _PredictWrapper:
    """Thin wrapper so _GenericSVRModel.explain() exposes raw-X → raw-y predict."""
    def __init__(self, pipeline, y_scaler):
        self._pipeline = pipeline
        self._y_scaler = y_scaler

    def predict(self, X):
        y_scaled = self._pipeline.predict(X)
        return self._y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


class _GenericSVRModel(ParameterModel):
    """
    SVR wrapper usable for any sensor parameter.
    Scales X (via Pipeline) and y (separate StandardScaler) to avoid SVR
    sensitivity to feature magnitude — same strategy as the reference pipeline.
    """

    def __init__(self, parameter_name: str, model_version: str, **svr_kwargs):
        super().__init__(parameter_name=parameter_name, model_version=model_version)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel="rbf", **svr_kwargs)),
        ])
        self.y_scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        y_scaled = self.y_scaler.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel()
        self.pipeline.fit(X_train, y_scaled)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_fitted()
        y_scaled = self.pipeline.predict(X)
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def explain(self, X: pd.DataFrame) -> Any:
        self.check_fitted()
        return _PredictWrapper(self.pipeline, self.y_scaler)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Evaluation result for a single (algorithm, hyperparameter) combination."""
    rank:            int
    label:           str               # e.g. "XGBoost  depth=3  lr=0.01  n=100"
    algorithm:       str               # "XGBoost" | "RF" | "SVR"
    hyperparams:     dict
    val_rmse:        float
    val_mae:         float
    train_rmse:      float
    fit_seconds:     float
    fitted_model:    ParameterModel    # ready for predict() or approval submission


@dataclass
class BenchmarkReport:
    """
    Full output of benchmark_models(). Not a decision — a report to be read.

    The caller must explicitly choose a model (by rank or label) before
    calling orchestrator.submit_benchmark_choice().
    """
    parameter_name: str
    unit:           str
    n_train:        int
    n_val:          int
    n_features:     int
    timestamp:      str
    results:        list[ModelResult]   # sorted by val_rmse ascending

    @property
    def best(self) -> ModelResult:
        return self.results[0]


# ── Public API ─────────────────────────────────────────────────────────────────

def benchmark_models(
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    X_val:          pd.DataFrame,
    y_val:          pd.Series,
    parameter_name: str,
    unit:           str = "",
) -> BenchmarkReport:
    """
    Train and evaluate RF / XGBoost / SVR over a restricted hyperparameter grid.

    All models are evaluated on the provided validation set. The grid is
    intentionally restricted to keep runtime below 60 s for datasets of the
    size expected in this project (≤ 500 training rows).

    Parameters
    ----------
    X_train, y_train : Training split.
    X_val,   y_val   : Validation split — used for model selection ranking.
                       NOT used for final test evaluation.
    parameter_name   : Sensor parameter name (used for model versioning).
    unit             : Physical unit string for display (e.g. "µS/cm").

    Returns
    -------
    BenchmarkReport  — ranked list of all evaluated models. No automatic
                       selection. Pass to format_benchmark_report() for display,
                       then to orchestrator.submit_benchmark_choice().
    """
    param_lower = parameter_name.lower()
    candidates: list[tuple[str, ParameterModel]] = []

    # ── RF grid ───────────────────────────────────────────────────────────────
    for n_est in [50, 100, 200]:
        for depth in [3, 5, None]:
            depth_tag = str(depth) if depth is not None else "full"
            label = f"RF         n={n_est:<3}  depth={depth_tag}"
            mv    = f"rf_{param_lower}_bench_d{depth_tag}_n{n_est}"
            model = _GenericRFModel(parameter_name, mv,
                                    n_estimators=n_est, max_depth=depth)
            candidates.append((label, model, {"n_estimators": n_est, "max_depth": depth}))

    # ── XGBoost grid ──────────────────────────────────────────────────────────
    for depth in [2, 3, 5]:
        for n_est in [50, 100]:
            for lr in [0.01, 0.05, 0.1]:
                label = f"XGBoost    depth={depth}  lr={lr}   n={n_est}"
                mv    = f"xgb_{param_lower}_bench_d{depth}_n{n_est}_lr{str(lr).replace('.','')}"
                model = _GenericXGBoostModel(
                    parameter_name, mv,
                    max_depth=depth, n_estimators=n_est, learning_rate=lr,
                )
                candidates.append((label, model, {"max_depth": depth, "n_estimators": n_est, "learning_rate": lr}))

    # ── SVR grid ──────────────────────────────────────────────────────────────
    for C in [1, 10, 100]:
        for eps in [0.1, 1.0]:
            label = f"SVR        C={C:<5}  eps={eps}"
            mv    = f"svr_{param_lower}_bench_C{C}_eps{str(eps).replace('.','')}"
            model = _GenericSVRModel(parameter_name, mv, C=C, epsilon=eps)
            candidates.append((label, model, {"C": C, "epsilon": eps}))

    # ── Fit & evaluate ────────────────────────────────────────────────────────
    raw_results: list[ModelResult] = []

    for label, model, hparams in candidates:
        algo = label.strip().split()[0]
        t0   = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        train_preds = model.predict(X_train)
        val_preds   = model.predict(X_val)

        train_rmse = float(np.sqrt(np.mean((y_train.values - train_preds) ** 2)))
        val_rmse   = float(np.sqrt(np.mean((y_val.values   - val_preds)   ** 2)))
        val_mae    = float(np.mean(np.abs(y_val.values - val_preds)))

        raw_results.append(ModelResult(
            rank         = 0,      # assigned after sort
            label        = label.strip(),
            algorithm    = algo,
            hyperparams  = hparams,
            val_rmse     = round(val_rmse,   4),
            val_mae      = round(val_mae,    4),
            train_rmse   = round(train_rmse, 4),
            fit_seconds  = round(elapsed,    3),
            fitted_model = model,
        ))

    raw_results.sort(key=lambda r: r.val_rmse)
    for rank, r in enumerate(raw_results, start=1):
        r.rank = rank

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    return BenchmarkReport(
        parameter_name = parameter_name,
        unit           = unit,
        n_train        = len(X_train),
        n_val          = len(X_val),
        n_features     = X_train.shape[1],
        timestamp      = ts,
        results        = raw_results,
    )


def format_benchmark_report(report: BenchmarkReport, top_n: int = 10) -> str:
    """
    Format a BenchmarkReport as a human-readable ranking table.

    Shows the top_n models (default 10). The complete list is always in
    report.results for programmatic inspection.
    """
    SEP  = "─" * 76
    SEP2 = "═" * 76

    lines = [
        SEP2,
        f"  BENCHMARK REPORT — {report.parameter_name}  ({report.unit})",
        SEP2,
        f"  Generated   : {report.timestamp}",
        f"  Train rows  : {report.n_train}  |  Val rows : {report.n_val}"
        f"  |  Features : {report.n_features}",
        f"  Models evaluated : {len(report.results)}"
        f"  (RF + XGBoost + SVR, restricted grid)",
        SEP,
        f"  {'Rank':>4}  {'Algorithm':<10}  {'Val RMSE':>10}  {'Val MAE':>9}"
        f"  {'Train RMSE':>11}  {'Fit (s)':>7}  Key params",
        SEP,
    ]

    for r in report.results[:top_n]:
        params_str = "  ".join(f"{k}={v}" for k, v in r.hyperparams.items())
        lines.append(
            f"  {r.rank:>4}  {r.algorithm:<10}  {r.val_rmse:>10.2f}  "
            f"{r.val_mae:>9.2f}  {r.train_rmse:>11.2f}  {r.fit_seconds:>7.3f}  "
            f"{params_str}"
        )

    if len(report.results) > top_n:
        lines.append(f"  ... {len(report.results) - top_n} more models not shown (all in report.results)")

    best = report.best
    lines += [
        SEP,
        f"  Best candidate (rank 1): {best.algorithm}",
        f"    Val RMSE  = {best.val_rmse:.4f} {report.unit}",
        f"    Val MAE   = {best.val_mae:.4f} {report.unit}",
        f"    Params    : {best.hyperparams}",
        SEP,
        "  ⚠  NO MODEL SELECTED AUTOMATICALLY. Human decision required.",
        "  Call submit_benchmark_choice(report, chosen_rank=N, ...) to proceed.",
        SEP2,
    ]

    return "\n".join(lines)


def quick_noise_diagnostic(df: pd.DataFrame, target_col: str) -> str:
    """
    Rapid noise profile summary for a sensor column.

    Computes a 7-row rolling standard deviation (shift(1) to avoid leakage),
    characterises noise level via CV, and flags whether the log-transform
    evaluation threshold is exceeded.

    Returns
    -------
    str — multi-line diagnostic report, printable directly.
    """
    series   = df[target_col].dropna().astype(float)
    roll_std = series.shift(1).rolling(7, min_periods=4).std().dropna()

    mean_signal = float(series.mean())
    std_signal  = float(series.std(ddof=1))
    mean_rstd   = float(roll_std.mean())
    median_rstd = float(roll_std.median())
    cv_signal   = std_signal / abs(mean_signal) * 100 if mean_signal != 0 else float("nan")
    cv_noise    = mean_rstd  / abs(mean_signal) * 100 if mean_signal != 0 else float("nan")

    if cv_noise < 5:
        noise_class = "Very low"
        rec = "Signal is highly stable. Minimal lag depth may suffice (lag1 alone)."
    elif cv_noise < 20:
        noise_class = "Low–moderate"
        rec = "Noise profile similar to typical water-quality sensors. Standard feature set recommended."
    elif cv_noise < 40:
        noise_class = "Moderate–high"
        rec = "Above-average variability. Consider adding EWMA features or longer rolling windows."
    else:
        noise_class = "High"
        rec = "High variability. Consider log-transform, differencing, or more robust features."

    log_flag = (
        f"  ⚠  Signal CV={cv_signal:.1f}% > {LOG_CV_THRESHOLD:.0f}% → "
        f"log-transform will be auto-evaluated (see benchmark)."
        if cv_signal > LOG_CV_THRESHOLD
        else
        f"  ✓  Signal CV={cv_signal:.1f}% ≤ {LOG_CV_THRESHOLD:.0f}% → "
        f"log-transform NOT evaluated (raw features sufficient)."
    )

    SEP = "─" * 62
    lines = [
        SEP,
        f"  NOISE DIAGNOSTIC — {target_col}",
        SEP,
        f"  Signal mean            : {mean_signal:>10.3f}",
        f"  Signal std             : {std_signal:>10.3f}",
        f"  Signal CV              : {cv_signal:>9.1f} %",
        f"  Rolling-7 std (mean)   : {mean_rstd:>10.3f}",
        f"  Rolling-7 std (median) : {median_rstd:>10.3f}",
        f"  Noise CV               : {cv_noise:>9.1f} %  ({noise_class})",
        SEP,
        f"  Recommendation: {rec}",
        log_flag,
        SEP,
    ]
    return "\n".join(lines)


# ── Log-transform helpers ──────────────────────────────────────────────────────

#: Signal CV threshold (%) above which log1p-transform is automatically evaluated.
#: Empirically calibrated on C-1 dataset: Turbidity CV=86.6% (triggers) vs
#: EC CV=23.4% and pH CV=11.9% (do not trigger).
LOG_CV_THRESHOLD: float = 60.0


def compute_signal_cv(df: pd.DataFrame, target_col: str) -> float:
    """
    Compute coefficient of variation (%) of a sensor column.

    Used by orchestrator.py to decide whether to run the automatic
    log-transform evaluation during onboarding.

    Returns float("nan") if the signal mean is zero.
    """
    series = df[target_col].dropna().astype(float)
    mean_v = float(series.mean())
    if mean_v == 0:
        return float("nan")
    return float(series.std(ddof=1) / abs(mean_v) * 100)


def rerank_log_report_to_original_scale(
    report_log: BenchmarkReport,
    X_val_log: pd.DataFrame,
    y_val_orig: np.ndarray,
    unit: str,
) -> BenchmarkReport:
    """
    Re-rank a log-space BenchmarkReport by original-scale RMSE.

    Models were trained on log1p(target). This function converts predictions
    back via expm1() and recomputes RMSE/MAE in the original unit, then
    re-ranks all models accordingly so the report is directly comparable to
    the raw-target BenchmarkReport.

    Parameters
    ----------
    report_log  : BenchmarkReport from benchmark_models() on log-scale target.
    X_val_log   : Validation features used for the log benchmark.
    y_val_orig  : Original-scale (NOT log-transformed) validation targets.
    unit        : Physical unit string for the relabelled report (e.g. "NTU").

    Returns
    -------
    BenchmarkReport with val_rmse and val_mae in original scale, re-ranked.
    The parameter_name is suffixed with "_log" to distinguish from the raw report.
    """
    y_orig = np.asarray(y_val_orig, dtype=float)
    new_results: list[ModelResult] = []

    for r in report_log.results:
        y_pred_orig = np.expm1(r.fitted_model.predict(X_val_log))
        rmse_orig   = float(np.sqrt(np.mean((y_orig - y_pred_orig) ** 2)))
        mae_orig    = float(np.mean(np.abs(y_orig - y_pred_orig)))
        new_results.append(ModelResult(
            rank         = 0,
            label        = r.label,
            algorithm    = r.algorithm,
            hyperparams  = r.hyperparams,
            val_rmse     = round(rmse_orig, 4),
            val_mae      = round(mae_orig,  4),
            train_rmse   = r.train_rmse,   # still in log scale; less meaningful
            fit_seconds  = r.fit_seconds,
            fitted_model = r.fitted_model,
        ))

    new_results.sort(key=lambda r: r.val_rmse)
    for rank, r in enumerate(new_results, start=1):
        r.rank = rank

    # Strip trailing "_log" from parameter_name before re-adding it,
    # so double-suffixing never occurs.
    base_name = report_log.parameter_name.removesuffix("_log")
    return BenchmarkReport(
        parameter_name = base_name + "_log",
        unit           = unit,
        n_train        = report_log.n_train,
        n_val          = report_log.n_val,
        n_features     = report_log.n_features,
        timestamp      = report_log.timestamp,
        results        = new_results,
    )
