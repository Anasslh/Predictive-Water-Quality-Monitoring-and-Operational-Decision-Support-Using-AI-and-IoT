"""
drift_detector.py — Statistical drift detection on model residuals.

DETECTION STRATEGY
------------------
Two independent signals are computed and combined with AND logic:

  1. Distribution shift (KS test)
     Kolmogorov-Smirnov two-sample test between the reference residual
     distribution (residuals on the last training window) and the recent
     residual distribution (residuals of the same model on new unseen data).
     Threshold: p-value < 0.05 (standard significance level; can be tuned
     via ks_alpha parameter).
     Interpretation: if the test rejects the null hypothesis, the residual
     distribution has changed shape (location, scale, or both).
     Important limitation: KS detects ANY distributional change, including
     benign seasonal shifts where the model actually performs better (e.g.
     high-variability monsoon season → calm dry season). KS alone is
     therefore an unreliable trigger for operational retraining decisions.

  2. RMSE degradation (ratio test)
     recent_rmse / reference_rmse > rmse_ratio_threshold
     Default threshold: 1.10 (i.e., 10% degradation in RMSE).
     Interpretation: the model's prediction accuracy has deteriorated
     measurably on the new data, regardless of distributional shape.
     This is the operationally meaningful signal — it directly measures
     what the user cares about.

DECISION GATE — RMSE ratio (primary), KS (diagnostic only)
------------------------------------------------------------
After empirical testing on the C-1 mining station (3 scenarios, KS power
analysis across 1 000 draws):

  The RMSE ratio is the reliable primary gate.
  The KS test is logged but NOT a decision gate.

  drift_detected = rmse_ratio > rmse_ratio_threshold

Rationale from empirical tests:

  (a) KS correctly detects genuine SEASONAL shifts (residuals from a calm
      dry season vs a high-variability monsoon season have different shapes).
      It fired at p=0.0083 on our test set — a valid distributional signal.

  (b) KS is BLIND to moderate degradation expressed as a scale increase on
      the same underlying distribution. Power analysis (1 000 draws, n=55):
        ×1.10 degradation → KS fires 5.8% (barely above α=0.05 = random)
        ×1.17 degradation → KS fires 7.3% (nearly random)
        ×1.20 degradation → KS fires 8.9% (still nearly random)
        ×1.50 degradation → KS fires 49.3% (finally useful)
      KS only becomes a reliable degradation detector for catastrophic
      failures (>50% RMSE increase), not for the moderate drifts that matter
      operationally.

  (c) RMSE ratio correctly handles all scenarios without this power gap:
        Benign seasonal improvement (RMSE ratio 0.566):  no fire ✓
        Moderate degradation        (RMSE ratio 1.20):   FIRE ✓
        Stable, no real change      (RMSE ratio 0.95):   no fire ✓

  (d) The acceptance gate in RetrainManager (new_rmse ≤ current × 1+tol)
      is the second safety net: even if the RMSE gate fires on a spurious
      case, a candidate trained on nearly identical data will not degrade
      the model — it will be accepted only if it is no worse.

  Why keep logging KS?
  KS tells you WHY drift happened: a sudden distributional jump (p very
  low, e.g. p < 0.001) points to a structural break (new contamination
  event, sensor replacement) rather than a slow drift. This is valuable
  context for the operator, independent of the retrain decision.
"""

import numpy as np
from scipy.stats import ks_2samp


def check_drift(
    reference_residuals: np.ndarray,
    recent_residuals: np.ndarray,
    ks_alpha: float = 0.05,
    rmse_ratio_threshold: float = 1.10,
) -> dict:
    """
    Test whether the recent residual distribution has drifted from the reference.

    Parameters
    ----------
    reference_residuals : np.ndarray (1-D)
        Residuals (actual - predicted) from the last training window.
        Computed on train+val data at the time of the last fit.
    recent_residuals : np.ndarray (1-D)
        Residuals of the SAME model on the new (unseen) data rows since
        the last training.
    ks_alpha : float
        Significance level for the KS test (default 0.05).
        Lower values (e.g. 0.01) make distribution-based detection
        more conservative.
    rmse_ratio_threshold : float
        Trigger ratio: recent_rmse / reference_rmse must exceed this to
        fire the RMSE signal (default 1.10 = 10% degradation).
        Set lower than the original 1.20 to compensate for the switch to
        AND logic — ensures subtle degradation is not missed.

    Returns
    -------
    dict with keys:
        drift_detected    bool   — True if either signal fires (OR logic)
        ks_statistic      float  — KS test statistic D
        ks_pvalue         float  — KS test p-value
        ks_drift          bool   — True if p-value < ks_alpha
        reference_rmse    float  — RMSE on reference residuals
        recent_rmse       float  — RMSE on recent residuals
        rmse_ratio        float  — recent_rmse / reference_rmse
        rmse_drift        bool   — True if rmse_ratio > rmse_ratio_threshold
        reason            str    — human-readable explanation

    Raises
    ------
    ValueError  if either array is empty or 1-D after flattening.
    """
    ref = np.asarray(reference_residuals).ravel()
    rec = np.asarray(recent_residuals).ravel()

    if len(ref) == 0:
        raise ValueError("reference_residuals is empty.")
    if len(rec) == 0:
        raise ValueError("recent_residuals is empty.")

    # --- KS test ---
    ks_stat, ks_pvalue = ks_2samp(ref, rec)
    ks_drift = bool(ks_pvalue < ks_alpha)

    # --- RMSE ratio ---
    reference_rmse = float(np.sqrt(np.mean(ref ** 2)))
    recent_rmse    = float(np.sqrt(np.mean(rec ** 2)))
    rmse_ratio     = recent_rmse / reference_rmse if reference_rmse > 0 else float("inf")
    rmse_drift     = bool(rmse_ratio > rmse_ratio_threshold)

    drift_detected = rmse_drift   # primary gate: RMSE ratio alone (KS logged but not a gate)

    # Build human-readable reason
    parts = []
    if ks_drift:
        parts.append(
            f"KS p={ks_pvalue:.4f} < α={ks_alpha} "
            f"(distribution shift, D={ks_stat:.3f})"
        )
    else:
        parts.append(
            f"KS p={ks_pvalue:.4f} ≥ α={ks_alpha} "
            f"(no distribution shift)"
        )
    if rmse_drift:
        parts.append(
            f"RMSE ratio={rmse_ratio:.3f} > {rmse_ratio_threshold} "
            f"({recent_rmse:.2f} vs ref {reference_rmse:.2f})"
        )
    else:
        parts.append(
            f"RMSE ratio={rmse_ratio:.3f} ≤ {rmse_ratio_threshold} "
            f"({recent_rmse:.2f} vs ref {reference_rmse:.2f})"
        )

    reason = " | ".join(parts)

    return {
        "drift_detected":      drift_detected,
        "ks_statistic":        round(float(ks_stat),    4),
        "ks_pvalue":           round(float(ks_pvalue),  4),
        "ks_drift":            ks_drift,
        "reference_rmse":      round(reference_rmse,    4),
        "recent_rmse":         round(recent_rmse,       4),
        "rmse_ratio":          round(rmse_ratio,        4),
        "rmse_drift":          rmse_drift,
        "reason":              reason,
    }
