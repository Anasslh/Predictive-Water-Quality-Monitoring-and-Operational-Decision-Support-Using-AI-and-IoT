"""
retrain_manager.py — Orchestrates the full retraining decision loop.

DESIGN PRINCIPLES
-----------------
- Generic: no knowledge of EC, pH, or turbidity. The parameter-specific
  model class and feature engineering function are injected at construction.
- Two-gate policy: a retrain is attempted only when BOTH conditions hold:
    1. Enough new rows have accumulated (volume gate: min_new_rows).
    2. A drift signal has been detected (quality gate: drift_detector).
  Volume alone is insufficient — a new model trained on an unchanged
  distribution is unlikely to outperform the current one, and the compute
  cost is wasted. Drift without volume is also insufficient — the recent
  window is too small for a reliable KS test or RMSE estimate.
- Conservative acceptance: a candidate is accepted only if its RMSE on the
  shared test set is no worse than the current model's RMSE * (1 + tolerance).
  This prevents degradation from noisy retraining episodes.
"""

from __future__ import annotations

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

from src.models.base import ParameterModel
from src.data.split import chronological_split
from src.retraining.drift_detector import check_drift
from src.retraining.approval import (
    APPROVAL_DIR_NAME,
    PendingApproval,
    submit_for_approval,
)
from src.retraining.model_versioning import (
    read_rejection_counter,
    write_rejection_counter,
)


logger = logging.getLogger(__name__)


class RetrainManager:
    """
    Manages the lifecycle of a ParameterModel: detects drift, retrains
    candidates, and decides whether to promote them.

    Parameters
    ----------
    model_class : type
        The ParameterModel subclass to instantiate for new candidates.
        Example: ECModelXGBoost. Must be callable with no required arguments.
        The manager never imports any specific model — it only calls the
        class reference it receives.
    feature_fn : Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]]
        A function that takes the raw accumulated DataFrame and returns
        (X, y) after applying the appropriate feature engineering.
        Example: lambda df: build_features(df, "EC")
        This is where the parameter name is injected — the manager itself
        never mentions "EC", "pH", or "Turbidity".
    min_new_rows : int
        Minimum number of new feature-engineered rows since the last training
        before drift is even checked. Default 60.
        Rationale: below ~60 observations, the RMSE ratio estimate is noisy
        enough that random fluctuations in a small window can exceed the 1.10
        threshold by chance (empirically validated on the C-1 daily dataset).
        60 rows ≈ 2 months of daily data — sufficient for a stable estimate
        while remaining responsive to real seasonal drift. Adapt to the
        station's reporting frequency: faster stations need larger windows in
        absolute row count to achieve the same statistical stability.
    tolerance : float
        A candidate model is accepted if its test RMSE satisfies:
          new_rmse <= current_rmse * (1 + tolerance)
        Default 0.02 (2%). Allows marginal regression not to trigger a
        rollback — XGBoost with stochastic elements may vary by ±1-2%
        across runs on the same data.
    """

    # Minimum new rows before drift check triggers.
    # Documented as a class constant so it is visible at the module level;
    # the constructor accepts an override.
    #
    # Why 60 and not 30?
    # With n_recent = 30, the RMSE ratio estimate has high variance: a random
    # fluctuation in a small window can push the ratio above the 1.10 threshold
    # by chance, triggering a spurious retrain. Empirical power analysis on the
    # C-1 station showed that n ≥ 60 stabilises the RMSE ratio estimate enough
    # to reduce such false triggers. For a daily-granularity station this means
    # waiting ~2 months of new data before checking — an acceptable delay given
    # that model drift on this dataset is slow (seasonal, not abrupt).
    # For higher-frequency stations (hourly) or faster-drifting sensors, this
    # value should be adapted to the actual data rate and expected drift speed.
    MIN_NEW_ROWS_DEFAULT: int = 60
    TOLERANCE_DEFAULT: float = 0.02
    REJECTION_ALERT_THRESHOLD_DEFAULT: int = 3

    # ── Sliding-window cap on training history ────────────────────────────────
    #
    # THEORETICAL CHOICE — NOT empirically validated on the C-1 dataset.
    # Rationale documented here; must be revalidated once multi-year data is
    # available (see also the per-call docstring of _apply_history_window).
    #
    # 5 years was chosen as a compromise between two competing objectives:
    #   • Enough seasonal cycles: water-quality parameters (EC, pH, turbidity)
    #     show strong annual cycles driven by rainfall, temperature, and
    #     agricultural runoff. A single year is insufficient to distinguish
    #     seasonal variability from inter-annual variability. Multiple years
    #     (≥3) are needed for the model to generalise across wet/dry season
    #     contrasts without overfitting to one particular year's pattern.
    #   • Regime freshness: mining-site water chemistry can shift structurally
    #     over a decade (new extraction areas, treatment plant upgrades,
    #     environmental regulations). Training on a 10- or 20-year window
    #     risks including data from an obsolete operational regime that
    #     dilutes, not enriches, the signal for the current regime.
    #
    # Weighting: UNIFORM across the window (no time-decay / exponential
    # smoothing). Decay weights add a free hyper-parameter that cannot be
    # calibrated without at least 2–3 full seasonal cycles of post-drift
    # data. With only 1 year of C-1 data, any decay constant would be
    # arbitrary. Uniform weights are the conservative, calibration-free
    # choice until multi-year data is available.
    #
    # TO REVALIDATE when ≥3 years of real data are collected:
    #   1. Plot hold-out RMSE vs window length (1, 2, 3, 5, 10 years).
    #   2. Check if a shorter window (e.g. 3 years) gives comparable or
    #      better RMSE with faster drift responsiveness.
    #   3. If inter-annual variability is high, consider exponential decay
    #      with a half-life tuned on the validation period.
    MAX_HISTORY_YEARS_DEFAULT: float = 5.0

    def __init__(
        self,
        model_class: type,
        feature_fn: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
        min_new_rows: int = MIN_NEW_ROWS_DEFAULT,
        tolerance: float = TOLERANCE_DEFAULT,
        models_store_path: str | Path | None = None,
        rejection_alert_threshold: int = REJECTION_ALERT_THRESHOLD_DEFAULT,
        max_history_years: float = MAX_HISTORY_YEARS_DEFAULT,
        date_col: str | None = None,
    ) -> None:
        self.model_class      = model_class
        self.feature_fn       = feature_fn
        self.min_new_rows     = min_new_rows
        self.tolerance        = tolerance
        self.rejection_alert_threshold = rejection_alert_threshold
        # Root of the model store (versioned production files + pending_approvals/).
        # Defaults to models_store/ relative to the working directory.
        self._models_store_path = (
            Path(models_store_path) if models_store_path is not None
            else Path("models_store")
        )

        # Sliding-window cap — see MAX_HISTORY_YEARS_DEFAULT docstring above.
        # Set to float("inf") or 0 to disable windowing entirely.
        self.max_history_years: float = max_history_years
        # Explicit date column name. None = auto-detect from all_data each call.
        self._date_col: str | None = date_col

        # Consecutive-rejection counter — incremented on rejection, reset on acceptance.
        # Loaded from disk so it survives process restarts.
        self.consecutive_rejections: int = read_rejection_counter(self._models_store_path)

        # State — populated by initialize() before attempt_retrain() is callable
        self.current_model: ParameterModel | None = None
        self.reference_residuals: np.ndarray | None = None
        self.last_train_size: int = 0    # number of feature-engineered rows at last fit

    # ------------------------------------------------------------------
    def initialize(
        self,
        initial_data: pd.DataFrame,
        initial_model: ParameterModel,
    ) -> None:
        """
        Set the starting model and record its reference residual distribution.

        Call this once before the first attempt_retrain(). It stores:
          - the current production model
          - the residuals of that model on the train+val portion of the
            initial data (used later as the reference distribution for KS)
          - the number of feature-engineered rows in the initial data

        Parameters
        ----------
        initial_data : pd.DataFrame
            Raw data available at the time the model was originally trained.
        initial_model : ParameterModel
            The already-fitted production model.
        """
        X, y = self.feature_fn(initial_data)
        X_train, y_train, X_val, y_val, _, _ = chronological_split(X, y)

        X_tv = pd.concat([X_train, X_val], ignore_index=True)
        y_tv = pd.concat([y_train, y_val], ignore_index=True)

        preds = initial_model.predict(X_tv)
        self.reference_residuals = y_tv.values - preds
        self.current_model       = initial_model
        self.last_train_size     = len(X)

    # ------------------------------------------------------------------
    def should_check_retrain(self, current_data_size: int) -> bool:
        """
        Return True if enough new rows have accumulated since the last fit.

        Parameters
        ----------
        current_data_size : int
            Number of feature-engineered rows in the current accumulated dataset.

        Returns
        -------
        bool
        """
        new_rows = current_data_size - self.last_train_size
        return new_rows >= self.min_new_rows

    # ------------------------------------------------------------------
    def attempt_retrain(self, all_data: pd.DataFrame) -> dict:
        """
        Run the full retrain decision loop on the current accumulated data.

        Steps
        -----
        a. Feature-engineer all accumulated data and check volume gate.
        b. Compute residuals of the current model on the new rows only
           (rows after last_train_size in feature space).
        c. Run drift detection (KS + RMSE ratio). If no drift: return early.
        d. If drift detected: train a candidate on train+val of all data.
        e. Evaluate candidate vs current model on the SAME test set.
        f. Accept the candidate if it meets the tolerance criterion.
           If accepted, update internal state (model, reference residuals,
           last_train_size).

        Parameters
        ----------
        all_data : pd.DataFrame
            The complete accumulated raw dataset (old rows + new rows).
            Must be compatible with feature_fn.

        Returns
        -------
        dict with keys:
            retrained              bool   — whether a candidate was trained
            accepted               bool   — whether the candidate passed the acceptance gate
            old_rmse               float  — current model RMSE on test (if retrained)
            new_rmse               float  — candidate RMSE on test (if retrained)
            drift                  dict   — output of check_drift() (if drift was checked)
            reason                 str    — human-readable summary of the decision
            consecutive_rejections int    — counter value after this attempt (if retrained)
            window_applied         bool   — True if history was capped to max_history_years
            training_rows          int    — feature-engineered rows used for candidate training
            alert                  str    — (rejected path only, when threshold reached)
                                            Human-readable recommendation to run a full
                                            benchmark. Key absent when no alert fires.
        """
        if self.current_model is None or self.reference_residuals is None:
            raise RuntimeError(
                "RetrainManager has not been initialized. Call initialize() first."
            )

        X, y = self.feature_fn(all_data)
        current_size = len(X)
        new_rows     = current_size - self.last_train_size

        # ── Gate 1: volume ────────────────────────────────────────────────────
        if not self.should_check_retrain(current_size):
            return {
                "retrained": False,
                "accepted":  False,
                "promoted":  False,
                "reason": (
                    f"Volume gate not met: {new_rows} new rows < "
                    f"min_new_rows={self.min_new_rows}. No action taken."
                ),
            }

        # ── Gate 2: drift ─────────────────────────────────────────────────────
        # Compute residuals of the CURRENT model on the new unseen rows only.
        X_recent = X.iloc[self.last_train_size:]
        y_recent = y.iloc[self.last_train_size:]
        recent_preds     = self.current_model.predict(X_recent)
        recent_residuals = y_recent.values - recent_preds

        drift_result = check_drift(self.reference_residuals, recent_residuals)

        if not drift_result["drift_detected"]:
            return {
                "retrained": False,
                "accepted":  False,
                "promoted":  False,
                "drift":     drift_result,
                "reason": (
                    f"Drift gate not met ({new_rows} new rows, "
                    f"but no drift signal). Retraining on a stable distribution "
                    f"would add compute cost without expected benefit."
                ),
            }

        # ── Both gates passed: apply sliding window, then train a candidate ─────
        #
        # The window is applied to all_data BEFORE the second feature-engineering
        # call so that time-based features (lags, rolling means) are computed on
        # the correct date range from the start — not truncated post-hoc from the
        # already-encoded feature matrix.
        #
        # Drift detection above deliberately uses the FULL (X, y) with
        # last_train_size indexing intact — the window must not disturb that
        # indexing.  The two roles are therefore kept separate:
        #   • (X, y)       → drift check only (full history, correct row index)
        #   • (X_win, y_win) → candidate training (windowed, most-recent regime)
        #
        # Known boundary effect: the first lag_max rows of X_win will have NaN
        # values where the lag window reaches before the cutoff date.  Those rows
        # are dropped inside feature_fn (same behaviour as at initial training).
        # For a 5-year window (~1825 daily rows) this is negligible (<0.4%).
        all_data_win   = self._apply_history_window(all_data)
        window_applied = len(all_data_win) < len(all_data)

        if window_applied:
            X_win, y_win = self.feature_fn(all_data_win)
        else:
            X_win, y_win = X, y   # reuse to avoid a redundant feature-engineering call

        X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X_win, y_win)

        X_tv = pd.concat([X_train, X_val], ignore_index=True)
        y_tv = pd.concat([y_train, y_val], ignore_index=True)

        candidate = self.model_class()
        candidate.fit(X_tv, y_tv)

        # ── Evaluate on the shared test set ────────────────────────────────────
        current_preds_test   = self.current_model.predict(X_test)
        candidate_preds_test = candidate.predict(X_test)

        old_rmse = float(np.sqrt(np.mean((y_test.values - current_preds_test)   ** 2)))
        new_rmse = float(np.sqrt(np.mean((y_test.values - candidate_preds_test) ** 2)))
        threshold_rmse = old_rmse * (1 + self.tolerance)
        accepted = new_rmse <= threshold_rmse

        if accepted:
            # ── No automatic promotion without human confirmation — safe-by-default ──
            #
            # A candidate that passes the acceptance gate is NOT promoted here.
            # Instead it is saved to pending_approvals/ and the decision is
            # deferred to a human reviewer via record_decision().
            # The current model continues serving predictions without interruption.
            #
            # This mirrors the selective verification pattern from PADSV (A-12):
            # human confirmation is required only at the highest-risk step
            # (replacing the production model), not at every intermediate gate.

            # Reset consecutive-rejection counter — the current architecture IS
            # capable of producing an acceptable candidate on this data.
            self.consecutive_rejections = 0
            write_rejection_counter(self._models_store_path, 0)

            pending_dir = self._models_store_path / APPROVAL_DIR_NAME

            # Enrich drift_result with FE row counts for training_period display
            drift_result["_n_train_val_rows"] = len(X_tv)
            drift_result["_total_fe_rows"]    = current_size

            approval = submit_for_approval(
                candidate               = candidate,
                old_rmse                = old_rmse,
                new_rmse                = new_rmse,
                drift_result            = drift_result,
                all_data                = all_data,
                pending_dir             = pending_dir,
                production_models_store = self._models_store_path,
            )
            reason = (
                f"Candidate passed acceptance gate (new_rmse={new_rmse:.2f} ≤ "
                f"old×(1+{self.tolerance})={threshold_rmse:.2f}) but is NOT "
                f"promoted automatically. Submitted for human approval — "
                f"ID: {approval.approval_id}. "
                f"Current model continues serving predictions without interruption."
            )
            return {
                "retrained":              True,
                "accepted":               True,    # acceptance gate passed
                "promoted":               False,   # no automatic promotion — awaits human decision
                "pending_approval":       approval,
                "old_rmse":               round(old_rmse, 4),
                "new_rmse":               round(new_rmse, 4),
                "drift":                  drift_result,
                "reason":                 reason,
                "consecutive_rejections": 0,
                "window_applied":         window_applied,
                "training_rows":          len(X_win),
            }

        else:
            # ── Rejection: increment consecutive-rejection counter ─────────────
            self.consecutive_rejections += 1
            write_rejection_counter(self._models_store_path, self.consecutive_rejections)

            reason = (
                f"Candidate rejected: new_rmse={new_rmse:.2f} > "
                f"old_rmse×(1+{self.tolerance})={threshold_rmse:.2f}. "
                f"Current model retained."
            )
            result: dict = {
                "retrained":              True,
                "accepted":               False,
                "promoted":               False,
                "old_rmse":               round(old_rmse, 4),
                "new_rmse":               round(new_rmse, 4),
                "drift":                  drift_result,
                "reason":                 reason,
                "consecutive_rejections": self.consecutive_rejections,
                "window_applied":         window_applied,
                "training_rows":          len(X_win),
            }

            # ── Alert policy: periodic modulo ────────────────────────────────
            #
            # Three options were considered:
            #
            #   (A) >= threshold  (every rejection from threshold onwards)
            #       Fires at 3, 4, 5, 6, 7 … → alert fatigue; operator stops
            #       reading after the 2nd or 3rd notification.
            #
            #   (B) == threshold  (once, then silence)
            #       Fires at 3 only. Risk: if the first alert is missed (e.g.
            #       logged to a file nobody reads that day), the operator has
            #       no further signal however many rejections accumulate.
            #
            #   (C) count % threshold == 0  (periodic, every N more rejections)
            #       Fires at 3, 6, 9, 12 … Chosen rationale:
            #         • The first alert is the actionable one — the operator
            #           should trigger a benchmark immediately.
            #         • If missed, a reminder every `rejection_alert_threshold`
            #           more rejections re-surfaces the signal without flooding.
            #         • Spacing equal to the threshold means silence grows
            #           proportionally to the sensitivity the caller configured:
            #           a strict threshold=3 re-alerts frequently; a loose
            #           threshold=10 leaves longer quiet periods.
            #
            # CHOSEN: option (C) — periodic modulo.
            if self.consecutive_rejections % self.rejection_alert_threshold == 0:
                alert = (
                    f"[ALERT] The current model has resisted "
                    f"{self.consecutive_rejections} consecutive retraining attempts. "
                    f"Continuing to retrain the same architecture in routine mode is "
                    f"unlikely to help. A full multi-model benchmark (see src/pipeline/) "
                    f"is recommended before the next retraining cycle."
                )
                logger.warning(alert)
                result["alert"] = alert

            return result

    # ------------------------------------------------------------------
    def apply_approved_candidate(
        self,
        approval: PendingApproval,
        all_data: pd.DataFrame,
    ) -> None:
        """
        Update in-memory manager state after an explicit approval decision.

        Must be called AFTER record_decision(..., True). In production, the
        manager would typically be re-instantiated from the versioned store on
        next startup; this method enables in-session promotion for demos and
        integration tests without restarting.

        Parameters
        ----------
        approval : PendingApproval
            The approved approval object (status must be "approved").
            Its candidate_model_path must still be accessible on disk.
        all_data : pd.DataFrame
            The complete raw dataset used to train the approved candidate
            (same data passed to attempt_retrain()). Used to recompute
            reference residuals after promotion.

        Raises
        ------
        ValueError        if approval.status != "approved".
        FileNotFoundError if the candidate pkl is no longer accessible.
        """
        if approval.status != "approved":
            raise ValueError(
                f"Cannot apply approval {approval.approval_id}: "
                f"status is '{approval.status}', expected 'approved'."
            )

        pkl_path = approval.candidate_model_path
        try:
            with open(pkl_path, "rb") as f:
                candidate = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Candidate pkl not found at {pkl_path}.\n"
                f"For in-session promotion, call apply_approved_candidate() "
                f"while the candidate pkl still exists under pending_approvals/.\n"
                f"In production, re-initialize the manager from the versioned "
                f"store instead (load_latest_version from model_versioning)."
            )

        X, y = self.feature_fn(all_data)
        X_train, y_train, X_val, y_val, _, _ = chronological_split(X, y)
        X_tv = pd.concat([X_train, X_val], ignore_index=True)
        y_tv = pd.concat([y_train, y_val], ignore_index=True)

        new_preds_tv             = candidate.predict(X_tv)
        self.reference_residuals = y_tv.values - new_preds_tv
        self.current_model       = candidate
        self.last_train_size     = len(X)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_date_col(self, df: pd.DataFrame) -> str | None:
        """
        Return the name of the first datetime-like column in df, or None.

        Detection order:
          1. Columns already typed datetime64 — no parsing needed.
          2. Object/string columns where ≥90% of values parse as dates with
             year ≥ 1970 — covers ISO-format strings from IoT loggers.
          3. Everything else is ignored (avoids mis-classifying numeric IDs
             or year-integer columns as date columns).
        """
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        for col in df.columns:
            if df[col].dtype not in (object, "string"):
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() >= 0.9 and parsed.min().year >= 1970:
                    return col
            except (TypeError, ValueError):
                pass

        return None

    def _apply_history_window(self, all_data: pd.DataFrame) -> pd.DataFrame:
        """
        Return all_data filtered to the most recent max_history_years years.

        DESIGN NOTE — theoretical choice, not empirically validated
        ----------------------------------------------------------
        5-year default (MAX_HISTORY_YEARS_DEFAULT): see the class-level
        constant docstring for the full rationale and revalidation checklist.
        Short version: ≥3 seasonal cycles for generalisation, ≤10 years to
        avoid incorporating an obsolete operational regime.

        Uniform weights: no time-decay. Cannot calibrate a decay constant
        without multi-year hold-out data. Revisit once ≥3 years of real
        C-1 data are available.

        Behaviour
        ---------
        • Windowing disabled (returns all_data unchanged) when:
            - max_history_years <= 0 or == inf
            - no datetime column found in all_data (emits RuntimeWarning)
            - total history span ≤ max_history_years (no data would be cut)
        • Returned DataFrame is reset_index(drop=True) to avoid index gaps
          in downstream chronological_split calls.

        Parameters
        ----------
        all_data : pd.DataFrame
            Raw accumulated dataset as passed to attempt_retrain().

        Returns
        -------
        pd.DataFrame  — windowed subset, or all_data unchanged.
        """
        if self.max_history_years <= 0 or self.max_history_years == float("inf"):
            return all_data

        date_col = self._date_col or self._detect_date_col(all_data)

        if date_col is None:
            warnings.warn(
                f"RetrainManager.max_history_years={self.max_history_years} is set "
                f"but no datetime column was detected in all_data "
                f"(columns: {list(all_data.columns)}). "
                f"Using the full history for candidate training. "
                f"Pass date_col='<your_date_column>' to suppress this warning.",
                RuntimeWarning,
                stacklevel=3,
            )
            return all_data

        dates   = pd.to_datetime(all_data[date_col], errors="coerce")
        max_dt  = dates.max()
        cutoff  = max_dt - pd.DateOffset(days=self.max_history_years * 365.25)
        mask    = dates >= cutoff

        if mask.all():
            return all_data   # history shorter than window — nothing to cut

        return all_data.loc[mask].reset_index(drop=True)
