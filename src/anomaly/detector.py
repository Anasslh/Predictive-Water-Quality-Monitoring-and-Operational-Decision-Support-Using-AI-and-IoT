"""
detector.py — Isolation Forest anomaly detector trained on prediction residuals.

Architecture (from literature review, A-12 / N-2):
  Phase 1 — Prediction : ECModelXGBoost predicts EC(t) from lag/rolling features.
  Phase 2 — Residual   : residual(t) = EC_actual(t) - EC_predicted(t).
  Phase 3 — Detection  : Isolation Forest on the 1-D residual stream.

Why residuals, not raw EC (N-2 finding):
  Raw-value Isolation Forest confounds normal high-EC periods with anomalies.
  Residuals isolate *unexpected* deviations from the prediction, making the
  detector blind to the EC level and sensitive only to model surprises.

Anomaly score [0, 1]:
  IsolationForest.score_samples() returns lower values for more anomalous
  points. We negate and min-max-normalize using the training distribution so
  that score ≈ 1 means "very anomalous" and score ≈ 0 means "normal", matching
  the A-12 PADSV convention (threshold = 0.5 by default).

Contamination:
  Set to 0.05 (5 %) — a conservative prior for this mining-zone dataset where
  genuine sensor faults and pollution spikes are expected to be rare.
  Affects the IsolationForest's internal offset but NOT our custom score
  normalization; the 0.5 threshold on the normalized score is independent.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

_DETECTOR_FILENAME = "anomaly_detector.pkl"


class ECAnomalyDetector:
    """
    Isolation Forest trained on EC prediction residuals.

    Usage
    -----
    # Build residuals from train+val (= "normal" reference period)
    residuals_tv = compute_residuals(model, X_tv, y_tv)
    detector = ECAnomalyDetector(contamination=0.05)
    detector.fit(residuals_tv)

    # Score new data
    residuals_test = compute_residuals(model, X_test, y_test)
    scores   = detector.score(residuals_test)    # float [0,1]
    is_anom  = detector.predict(residuals_test)  # bool array
    """

    def __init__(
        self,
        contamination: float = 0.05,
        threshold: float = 0.5,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        contamination : expected fraction of anomalies in the training data.
                        Keep low (0.02–0.10) — train+val should be "normal".
        threshold     : anomaly_score >= threshold → flagged as anomaly.
                        0.5 follows A-12 PADSV convention.
        random_state  : reproducibility seed for IsolationForest.
        """
        self.contamination = contamination
        self.threshold = threshold
        self._forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self._score_scaler = MinMaxScaler()
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, residuals: np.ndarray) -> "ECAnomalyDetector":
        """
        Fit the detector on the "normal" residual distribution (train+val).

        Parameters
        ----------
        residuals : 1-D array of shape (n,) — output of compute_residuals()
                    on the combined train+val set.
        """
        X = residuals.reshape(-1, 1)
        self._forest.fit(X)

        # Fit the score scaler on training anomaly scores so all future
        # scores are in [0, 1] relative to the training distribution.
        raw = -self._forest.score_samples(X)   # higher = more anomalous
        self._score_scaler.fit(raw.reshape(-1, 1))

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def score(self, residuals: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores in [0, 1] for each residual value.
        Score ≈ 1 → highly anomalous  |  Score ≈ 0 → normal.

        Parameters
        ----------
        residuals : 1-D array of shape (n,).

        Returns
        -------
        scores : np.ndarray of shape (n,), values in [0, 1].
        """
        self._check_fitted()
        raw = -self._forest.score_samples(residuals.reshape(-1, 1))
        scores = self._score_scaler.transform(raw.reshape(-1, 1)).ravel()
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    def predict(self, residuals: np.ndarray) -> np.ndarray:
        """
        Return a boolean anomaly flag for each residual value.
        True  → anomaly (score >= threshold)
        False → normal

        Parameters
        ----------
        residuals : 1-D array of shape (n,).

        Returns
        -------
        flags : np.ndarray of bool, shape (n,).
        """
        return self.score(residuals) >= self.threshold

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ECAnomalyDetector has not been fitted yet — call fit() first."
            )


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_anomaly_detector(
    detector: ECAnomalyDetector,
    models_store_path: str | Path,
) -> Path:
    """Serialize a fitted ECAnomalyDetector to <models_store_path>/anomaly_detector.pkl."""
    out_path = Path(models_store_path) / _DETECTOR_FILENAME
    with open(out_path, "wb") as f:
        pickle.dump(detector, f)
    return out_path


def load_anomaly_detector(
    models_store_path: str | Path,
) -> "ECAnomalyDetector | None":
    """Load a fitted ECAnomalyDetector from <models_store_path>/anomaly_detector.pkl.

    Returns None if the file does not exist (detector not yet trained for this parameter).
    """
    p = Path(models_store_path) / _DETECTOR_FILENAME
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
