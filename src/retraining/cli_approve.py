"""
cli_approve.py — Interactive CLI for reviewing pending model-promotion approvals.

Usage
-----
  # List all pending approvals (all parameters)
  python -m src.retraining.cli_approve

  # Full detail + interactive y/n for one approval
  python -m src.retraining.cli_approve <approval_id>

  # List approvals waiting more than 14 days (no automatic action)
  python -m src.retraining.cli_approve --reject-all-stale

  # Same with a custom threshold
  python -m src.retraining.cli_approve --reject-all-stale --stale-days 7

  # Override the default models_store/ directory
  python -m src.retraining.cli_approve --models-store /path/to/store

Design constraints
------------------
- Generic: discovers parameters by scanning models_store/<param>/pending_approvals/.
- Never modifies approval.py — only calls its public API:
    list_pending_approvals(), record_decision(), format_report().
- No automatic mass-action: --reject-all-stale lists stale items but does not
  reject them. The user must go through the per-ID interactive flow.
- input_fn and record_fn are injectable so the whole CLI is unit-testable
  without touching sys.stdin or the filesystem promotion path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

# ── Project-root bootstrap (works as module -m or as a direct script) ──────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.retraining.approval import (
    list_pending_approvals,
    record_decision,
    format_report,
    PendingApproval,
)

_DEFAULT_MODELS_STORE = _ROOT / "models_store"
_APPROVAL_DIR_NAME    = "pending_approvals"
logger = logging.getLogger(__name__)


# ── Discovery helpers ──────────────────────────────────────────────────────────

def _all_pending_dirs(models_store: Path) -> list[Path]:
    """
    Yield every pending_approvals/ sub-directory found under models_store/<param>/.
    """
    if not models_store.is_dir():
        return []
    return sorted(
        d / _APPROVAL_DIR_NAME
        for d in models_store.iterdir()
        if d.is_dir() and (d / _APPROVAL_DIR_NAME).is_dir()
    )


def _list_all_pending(models_store: Path) -> list[tuple[PendingApproval, Path]]:
    """
    Return all pending approvals across every parameter, with their pending_dir.

    Returns list of (PendingApproval, pending_dir_path), sorted oldest-first.
    """
    results: list[tuple[PendingApproval, Path]] = []
    for pd_dir in _all_pending_dirs(models_store):
        for approval in list_pending_approvals(pd_dir):
            results.append((approval, pd_dir))
    results.sort(key=lambda t: t[0].timestamp_created)
    return results


def _find_by_id(
    approval_id: str,
    models_store: Path,
) -> tuple[PendingApproval, Path] | None:
    """
    Locate a specific approval by its ID, scanning all parameters.

    Returns (PendingApproval, pending_dir) or None if not found.
    """
    for pd_dir in _all_pending_dirs(models_store):
        for approval in list_pending_approvals(pd_dir):
            if approval.approval_id == approval_id:
                return approval, pd_dir
    return None


# ── Display helpers ────────────────────────────────────────────────────────────

def _age_label(ts_iso: str) -> str:
    """Return a human-readable age string for a ISO-8601 timestamp."""
    try:
        dt = datetime.fromisoformat(ts_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        days  = delta.days
        if days == 0:
            return "today"
        if days == 1:
            return "1 day ago"
        return f"{days} days ago"
    except ValueError:
        return ts_iso


def _format_listing(pairs: list[tuple[PendingApproval, Path]]) -> str:
    """
    Compact multi-line listing suitable for scanning quickly.

    Example output:
        ════════════════════════════════════════════
          2 pending approval(s)
        ════════════════════════════════════════════

          #1  EC / xgb_ec_v1_final
              ID        : EC_xgb_ec_v1_final_20260720_143022_a1b2c3
              Submitted : 2026-07-20  (1 day ago)
              RMSE      : 53.48 → 51.20  (+4.3% improvement)
              Trigger   : RMSE ratio 1.12 > threshold 1.10

          #2  pH / xgb_ph_v1_final
              ...
        ────────────────────────────────────────────
          To review:  python -m src.retraining.cli_approve <id>
    """
    SEP  = "─" * 58
    SEP2 = "═" * 58

    if not pairs:
        return f"\n  {SEP}\n  No pending approvals.\n  {SEP}\n"

    lines = [
        "",
        SEP2,
        f"  {len(pairs)} pending approval(s)",
        SEP2,
    ]

    for idx, (a, _) in enumerate(pairs, start=1):
        direction = (
            f"+{a.improvement_pct:.1f}%  (candidate better)"
            if a.improvement_pct >= 0
            else f"{a.improvement_pct:.1f}%  (candidate worse)"
        )
        lines += [
            "",
            f"  #{idx}  {a.parameter_name} / {a.model_version}",
            f"      ID        : {a.approval_id}",
            f"      Submitted : {a.timestamp_created[:10]}  ({_age_label(a.timestamp_created)})",
            f"      RMSE      : {a.current_rmse:.4f}  →  {a.candidate_rmse:.4f}  ({direction})",
            f"      Trigger   : {a.drift_reason}",
        ]

    lines += [
        "",
        SEP,
        "  To review:   python -m src.retraining.cli_approve <id>",
        "  Stale check: python -m src.retraining.cli_approve --reject-all-stale",
        "",
    ]
    return "\n".join(lines)


# ── Command implementations ────────────────────────────────────────────────────

def _cmd_list(
    models_store: Path,
    output_fn: Callable[[str], None] = print,
) -> None:
    """List all pending approvals (no interaction)."""
    pairs = _list_all_pending(models_store)
    output_fn(_format_listing(pairs))


def _cmd_detail(
    approval_id: str,
    models_store: Path,
    input_fn:  Callable[[str], str]                       = input,
    record_fn: Callable[..., dict]                        = record_decision,
    output_fn: Callable[[str], None]                      = print,
) -> None:
    """
    Show full detail for one approval, then ask for interactive y/n decision.

    Calls record_fn only after an explicit 'y' from the user.
    'n' / empty / anything else → abort, no action taken.

    Parameters
    ----------
    approval_id : ID to look up (scanned across all parameters).
    models_store: Root of the models_store/ tree.
    input_fn    : Callable that prompts and returns a string (injectable for tests).
    record_fn   : Callable matching record_decision() signature (injectable for tests).
    output_fn   : Callable for output lines (injectable for tests).
    """
    found = _find_by_id(approval_id, models_store)
    if found is None:
        output_fn(f"\n  [ERROR] No pending approval found with ID: {approval_id}\n")
        return

    approval, pending_dir = found

    output_fn(format_report(approval))
    output_fn("")

    answer = input_fn("  Decision — approve? [y/N] : ").strip().lower()

    if answer != "y":
        output_fn("  Aborted — no action taken.\n")
        return

    note = input_fn("  Reviewer note (optional, press Enter to skip): ").strip()

    output_fn("")
    output_fn(f"  Approving {approval_id} …")
    result = record_fn(
        approval_id   = approval_id,
        decision      = True,
        pending_dir   = pending_dir,
        reviewer_note = note,
    )
    if result.get("promoted"):
        output_fn(f"  Promoted → {result['saved_path']}")
    output_fn("  Done.\n")


def _cmd_stale(
    models_store: Path,
    stale_days: int                     = 14,
    output_fn:  Callable[[str], None]   = print,
) -> list[tuple[PendingApproval, Path]]:
    """
    List approvals pending for more than stale_days. No automatic action.

    Returns the stale pairs so the caller (or test) can inspect them.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
    pairs  = _list_all_pending(models_store)

    stale: list[tuple[PendingApproval, Path]] = []
    for a, pd_dir in pairs:
        try:
            dt = datetime.fromisoformat(a.timestamp_created)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                stale.append((a, pd_dir))
        except ValueError as exc:
            logger.warning(
                "_cmd_stale: malformed timestamp in approval '%s', skipping stale check for this entry: %s",
                a.approval_id, exc,
            )

    SEP  = "─" * 58
    SEP2 = "═" * 58

    if not stale:
        output_fn(f"\n  {SEP}")
        output_fn(f"  No approvals pending for more than {stale_days} days.")
        output_fn(f"  {SEP}\n")
        return stale

    output_fn("")
    output_fn(SEP2)
    output_fn(f"  {len(stale)} stale approval(s)  (waiting > {stale_days} days)")
    output_fn(SEP2)

    for a, _ in stale:
        output_fn(
            f"  {a.approval_id}"
            f"  |  {a.parameter_name}"
            f"  |  submitted {a.timestamp_created[:10]}"
            f"  ({_age_label(a.timestamp_created)})"
        )

    output_fn("")
    output_fn(SEP)
    output_fn("  These approvals have NOT been rejected automatically.")
    output_fn("  To act on one:  python -m src.retraining.cli_approve <id>")
    output_fn("")

    return stale


# ── Entry point ────────────────────────────────────────────────────────────────

def main(
    argv: list[str] | None                = None,
    models_store: Path | str | None       = None,
    input_fn:  Callable[[str], str]       = input,
    record_fn: Callable[..., dict]        = record_decision,
    output_fn: Callable[[str], None]      = print,
) -> None:
    """
    CLI entry point.  All I/O channels are injectable for tests.

    Parameters
    ----------
    argv         : Argument list (default sys.argv[1:]).
    models_store : Override the models_store root path.
    input_fn     : Callable for interactive prompts (injectable).
    record_fn    : Callable matching record_decision() (injectable).
    output_fn    : Callable for output (injectable).
    """
    parser = argparse.ArgumentParser(
        prog        = "python -m src.retraining.cli_approve",
        description = "Review and approve/reject pending model candidates.",
        add_help    = True,
    )
    parser.add_argument(
        "approval_id",
        nargs   = "?",
        default = None,
        help    = "Approval ID to review interactively. Omit to list all.",
    )
    parser.add_argument(
        "--reject-all-stale",
        action  = "store_true",
        help    = "List approvals that have been waiting more than --stale-days days (no auto-action).",
    )
    parser.add_argument(
        "--stale-days",
        type    = int,
        default = 14,
        metavar = "N",
        help    = "Age threshold in days for --reject-all-stale (default 14).",
    )
    parser.add_argument(
        "--models-store",
        type    = Path,
        default = None,
        metavar = "PATH",
        help    = f"Path to models_store/ root (default: {_DEFAULT_MODELS_STORE}).",
    )

    args  = parser.parse_args(argv)
    store = Path(models_store) if models_store else (args.models_store or _DEFAULT_MODELS_STORE)

    if args.reject_all_stale:
        _cmd_stale(store, stale_days=args.stale_days, output_fn=output_fn)
    elif args.approval_id:
        _cmd_detail(
            args.approval_id,
            models_store = store,
            input_fn     = input_fn,
            record_fn    = record_fn,
            output_fn    = output_fn,
        )
    else:
        _cmd_list(store, output_fn=output_fn)


if __name__ == "__main__":
    main()
