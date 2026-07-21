"""
approval.py — Human-in-the-loop approval gate for candidate model promotion.

DESIGN RATIONALE
----------------
The automatic pipeline gates promotion on two checks (drift detection +
RMSE acceptance). Both are necessary conditions but not sufficient for
unattended deployment. Replacing a production model is the highest-risk
action in the pipeline — it affects all downstream predictions until the
next approved replacement.

Human confirmation at this final step costs little (a one-line decision
per candidate) and prevents edge cases that automatic gates can miss:
  - Legitimate RMSE gain but suspicious data provenance
  - Threshold-boundary passes that represent statistical noise, not drift
  - Seasonal patterns that look like drift but should not trigger retraining

CURRENT IMPLEMENTATION — SIMULATION
-------------------------------------
The human interface is SIMULATED: the decision is recorded by calling
record_decision() directly with True/False from the caller's code.

In a real deployment, the JSON files created by submit_for_approval()
would be surfaced through a dashboard, CLI utility, or notification
(Slack/email), and record_decision() would be called when the reviewer
acts. The JSON contract defined here is stable — the UI layer is
entirely decoupled from this module.

SAFE-BY-DEFAULT GUARANTEE
--------------------------
A candidate model is NEVER copied to the production versioned store
without an explicit True from record_decision(). Rejecting a candidate
removes its pkl immediately; no corrupted state is left in models_store/.
The current serving model is never touched by any function in this module.
"""

from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base import ParameterModel
from src.retraining.model_versioning import (
    save_model_version,
    write_current_model_pointer,
)


# Directory name created inside models_store_path by the manager.
APPROVAL_DIR_NAME = "pending_approvals"


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class PendingApproval:
    """
    Complete snapshot of a candidate model waiting for human review.

    Serialized as JSON to {pending_dir}/{approval_id}.json.
    The candidate pkl lives at candidate_model_path.

    Status lifecycle
    ----------------
    "pending_approval"  →  "approved"  (record_decision called with True)
                        →  "rejected"  (record_decision called with False)

    Fields
    ------
    approval_id : str
        Unique ID, format: {parameter_name}_{model_version}_{YYYYMMDD_HHMMSS}.
    current_rmse : float
        RMSE of the current production model on the shared test set.
    candidate_rmse : float
        RMSE of the candidate on the same test set.
    improvement_pct : float
        (current_rmse - candidate_rmse) / current_rmse * 100.
        Positive = candidate is better.
    drift_reason : str
        Human-readable summary verbatim from check_drift()["reason"].
    rmse_ratio : float
        recent_rmse / reference_rmse at drift-check time. > threshold → drift.
    ks_pvalue : float
        KS two-sample p-value (diagnostic only — not the decision gate).
    ks_drift : bool
        Whether KS fired at drift-check time.
    training_period : dict
        {"n_train_val_rows": int, "total_fe_rows": int,
         "data_start": str, "data_end": str}
    candidate_model_path : str
        Absolute path to the candidate pkl saved under pending_dir.
        Deleted on rejection; kept on approval (also saved to versioned store).
    production_models_store : str
        Absolute path where save_model_version() writes on approval.
    timestamp_created : str   ISO 8601 UTC
    status : str              "pending_approval" | "approved" | "rejected"
    reviewer_note : str       Free text written by the reviewer on decision.
    timestamp_decided : str   ISO 8601 UTC, empty string while still pending.
    """
    approval_id:             str
    parameter_name:          str
    model_version:           str
    current_rmse:            float
    candidate_rmse:          float
    improvement_pct:         float
    drift_reason:            str
    rmse_ratio:              float
    ks_pvalue:               float
    ks_drift:                bool
    training_period:         dict
    candidate_model_path:    str
    production_models_store: str
    timestamp_created:       str
    status:                  str = "pending_approval"
    reviewer_note:           str = ""
    timestamp_decided:       str = ""


# ── Public API ─────────────────────────────────────────────────────────────────

def submit_for_approval(
    candidate: ParameterModel,
    old_rmse: float,
    new_rmse: float,
    drift_result: dict,
    all_data: Any,         # pd.DataFrame — typed as Any to avoid hard import
    pending_dir: str | Path,
    production_models_store: str | Path,
) -> PendingApproval:
    """
    Save a candidate model and its metadata for human review.

    This function does NOT touch the current production model. It:
      1. Serializes the candidate to a pkl under pending_dir.
      2. Builds a PendingApproval with all relevant metrics.
      3. Saves the approval metadata as a JSON file under pending_dir.

    Parameters
    ----------
    candidate : ParameterModel
        The trained candidate model. Must be fitted (_is_fitted = True).
    old_rmse : float
        RMSE of the current production model on the shared test set.
    new_rmse : float
        RMSE of the candidate on the same test set.
    drift_result : dict
        Output of check_drift() — must contain keys: "drift_reason",
        "rmse_ratio", "ks_pvalue", "ks_drift".
    all_data : pd.DataFrame
        Raw accumulated data used to train the candidate. Used only to
        extract the date range for the training_period field.
    pending_dir : str | Path
        Directory where the pkl and JSON are written. Created if absent.
    production_models_store : str | Path
        Production store path. Stored in the approval for use at promotion.

    Returns
    -------
    PendingApproval
    """
    pending_dir = Path(pending_dir)
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Include a 6-char UUID suffix to guarantee uniqueness even when two
    # submissions occur within the same second (e.g. demo scripts, tests).
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    approval_id = f"{candidate.parameter_name}_{candidate.model_version}_{ts}_{uid}"

    # Save candidate pkl
    pkl_path = pending_dir / f"{approval_id}_candidate.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(candidate, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Extract date range from raw data if available
    data_start = "unknown"
    data_end   = "unknown"
    try:
        if hasattr(all_data, "columns") and "Date" in all_data.columns:
            data_start = str(all_data["Date"].min().date())
            data_end   = str(all_data["Date"].max().date())
    except Exception:
        pass

    # Determine train+val size via approval's drift context
    # (total_fe_rows not directly available here — passed via training_period
    #  if the caller enriches it; we use the drift window as a proxy)
    n_fe_rows_tv  = drift_result.get("_n_train_val_rows", "unknown")
    n_fe_rows_all = drift_result.get("_total_fe_rows",    "unknown")

    improvement_pct = (
        (old_rmse - new_rmse) / old_rmse * 100.0
        if old_rmse > 0 else 0.0
    )

    approval = PendingApproval(
        approval_id             = approval_id,
        parameter_name          = candidate.parameter_name,
        model_version           = candidate.model_version,
        current_rmse            = round(old_rmse, 4),
        candidate_rmse          = round(new_rmse, 4),
        improvement_pct         = round(improvement_pct, 2),
        drift_reason            = drift_result.get("reason", ""),
        rmse_ratio              = round(float(drift_result.get("rmse_ratio", 0.0)), 4),
        ks_pvalue               = round(float(drift_result.get("ks_pvalue",  1.0)), 4),
        ks_drift                = bool(drift_result.get("ks_drift", False)),
        training_period         = {
            "n_train_val_rows": n_fe_rows_tv,
            "total_fe_rows":    n_fe_rows_all,
            "data_start":       data_start,
            "data_end":         data_end,
        },
        candidate_model_path    = str(pkl_path.resolve()),
        production_models_store = str(Path(production_models_store).resolve()),
        timestamp_created       = datetime.now(timezone.utc).isoformat(),
    )

    _save_approval_json(approval, pending_dir)
    return approval


def record_decision(
    approval_id: str,
    decision: bool,
    pending_dir: str | Path,
    reviewer_note: str = "",
    max_versions: int = 3,
) -> dict:
    """
    Record a human decision on a pending candidate.

    SIMULATED INTERFACE — in production, this would be called by a dashboard
    or CLI tool when a reviewer acts. Calling it with True/False directly in
    code simulates the human interaction for demonstration purposes.

    Parameters
    ----------
    approval_id : str
        ID of the approval to decide on (from PendingApproval.approval_id).
    decision : bool
        True = approve and promote the candidate to production versioned store.
        False = reject and delete the candidate pkl.
    pending_dir : str | Path
        Same directory passed to submit_for_approval().
    reviewer_note : str
        Optional free-text note recorded in the approval JSON.
    max_versions : int
        Passed to save_model_version() on approval (default 3).

    Returns
    -------
    dict with keys:
        promoted   bool   — True if candidate was saved to production store.
        saved_path str    — Path of the new versioned file (if promoted).
        approval   PendingApproval — Updated approval object.

    Raises
    ------
    FileNotFoundError  if the approval JSON does not exist.
    ValueError         if the approval is not in "pending_approval" status.
    """
    pending_dir  = Path(pending_dir)
    json_path    = pending_dir / f"{approval_id}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Approval JSON not found: {json_path}\n"
            f"Available IDs: {[p.stem for p in pending_dir.glob('*.json')]}"
        )

    approval = _load_approval_json(json_path)

    if approval.status != "pending_approval":
        raise ValueError(
            f"Approval {approval_id} already decided: status='{approval.status}'."
        )

    ts_decided = datetime.now(timezone.utc).isoformat()

    if decision:
        # ── Promote: load candidate pkl → save to production versioned store ──
        pkl_path = Path(approval.candidate_model_path)
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Candidate pkl not found: {pkl_path}\n"
                f"It may have been deleted externally."
            )
        with open(pkl_path, "rb") as f:
            candidate = pickle.load(f)

        saved_path = save_model_version(
            candidate,
            models_store_path=approval.production_models_store,
            max_versions=max_versions,
        )

        # Write the restart-safe pointer so the next process start finds the
        # approved model without needing to parse approval JSONs.
        write_current_model_pointer(
            models_store_path = approval.production_models_store,
            model_pkl_path    = saved_path,
            approval_id       = approval.approval_id,
            parameter_name    = approval.parameter_name,
            model_version     = approval.model_version,
        )

        approval.status            = "approved"
        approval.reviewer_note     = reviewer_note
        approval.timestamp_decided = ts_decided
        _save_approval_json(approval, pending_dir)

        return {
            "promoted":   True,
            "saved_path": str(saved_path),
            "approval":   approval,
        }

    else:
        # ── Reject: delete candidate pkl, keep JSON for audit ─────────────────
        pkl_path = Path(approval.candidate_model_path)
        if pkl_path.exists():
            pkl_path.unlink()

        approval.status            = "rejected"
        approval.reviewer_note     = reviewer_note
        approval.timestamp_decided = ts_decided
        _save_approval_json(approval, pending_dir)

        return {
            "promoted":   False,
            "saved_path": None,
            "approval":   approval,
        }


def list_pending_approvals(pending_dir: str | Path) -> list[PendingApproval]:
    """
    Return all approvals currently waiting for a human decision.

    Parameters
    ----------
    pending_dir : str | Path
        Directory passed to submit_for_approval().

    Returns
    -------
    list[PendingApproval]  — sorted by timestamp_created, oldest first.
    """
    pending_dir = Path(pending_dir)
    if not pending_dir.exists():
        return []

    approvals = []
    for json_path in sorted(pending_dir.glob("*.json")):
        try:
            a = _load_approval_json(json_path)
            if a.status == "pending_approval":
                approvals.append(a)
        except Exception:
            pass

    return sorted(approvals, key=lambda a: a.timestamp_created)


def format_report(approval: PendingApproval) -> str:
    """
    Format a PendingApproval as a human-readable review report.

    This is what an operator would see in a dashboard or CLI notification.
    """
    SEP  = "─" * 58
    SEP2 = "═" * 58

    status_icon = {
        "pending_approval": "⏳ pending_approval",
        "approved":         "✅ approved",
        "rejected":         "❌ rejected",
    }.get(approval.status, approval.status)

    direction = (
        f"+{approval.improvement_pct:.1f}%  ← candidate is better"
        if approval.improvement_pct >= 0
        else f"{approval.improvement_pct:.1f}%  ← candidate is worse"
    )

    tp = approval.training_period
    n_tv  = tp.get("n_train_val_rows", "?")
    n_all = tp.get("total_fe_rows",    "?")
    d0    = tp.get("data_start", "?")
    d1    = tp.get("data_end",   "?")

    decided_line = ""
    if approval.timestamp_decided:
        decided_line = f"Decided   : {approval.timestamp_decided}\n"
    note_line = ""
    if approval.reviewer_note:
        note_line = f"Reviewer  : {approval.reviewer_note}\n"

    action_block = (
        f"  record_decision(\"{approval.approval_id}\", True,  pending_dir, reviewer_note=\"…\")\n"
        f"  record_decision(\"{approval.approval_id}\", False, pending_dir, reviewer_note=\"…\")"
        if approval.status == "pending_approval" else ""
    )

    lines = [
        SEP2,
        f"  PENDING APPROVAL — {approval.parameter_name} / {approval.model_version}",
        SEP2,
        f"  ID       : {approval.approval_id}",
        f"  Status   : {status_icon}",
        f"  Created  : {approval.timestamp_created}",
    ]
    if decided_line:
        lines.append(f"  {decided_line.rstrip()}")
    if note_line:
        lines.append(f"  {note_line.rstrip()}")
    lines += [
        SEP,
        "  PERFORMANCE  (shared test set)",
        f"  Current model RMSE  : {approval.current_rmse:.2f} µS/cm",
        f"  Candidate RMSE      : {approval.candidate_rmse:.2f} µS/cm",
        f"  Improvement         : {direction}",
        SEP,
        "  DRIFT DETECTION CONTEXT",
        f"  RMSE ratio (trigger): {approval.rmse_ratio:.4f}  (threshold > 1.10)",
        f"  KS p-value  (diag)  : {approval.ks_pvalue:.4f}  "
        f"({'distribution shift detected' if approval.ks_drift else 'no distribution shift'})",
        f"  Drift reason        : {approval.drift_reason}",
        SEP,
        "  TRAINING DATA",
        f"  Train+val FE rows   : {n_tv}",
        f"  Total FE rows       : {n_all}",
        f"  Date range          : {d0}  →  {d1}",
        SEP,
        "  CANDIDATE FILE",
        f"  {approval.candidate_model_path}",
    ]
    if action_block:
        lines += [SEP, "  TO ACT:", action_block]
    lines.append(SEP2)

    return "\n".join(lines)


# ── Private helpers ────────────────────────────────────────────────────────────

def _save_approval_json(approval: PendingApproval, pending_dir: Path) -> None:
    json_path = pending_dir / f"{approval.approval_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(approval), f, indent=2, ensure_ascii=False)


def _load_approval_json(json_path: Path) -> PendingApproval:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return PendingApproval(**data)
