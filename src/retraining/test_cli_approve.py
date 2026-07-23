"""
test_cli_approve.py — Tests for src/retraining/cli_approve.py

Simulates two pending approvals (EC and pH) written to a temp directory,
then tests:
  1. Listing shows both approvals
  2. Approving (input="y") triggers record_fn with decision=True
  3. Rejecting (input="n") does NOT trigger record_fn
  4. Stale listing shows only approvals older than the threshold

record_fn is injected as a mock so no real pkl loading or file promotion
happens.  The JSON approval files are written directly (no ParameterModel
training required).

Run:
    python -m src.retraining.test_cli_approve
    # or
    python src/retraining/test_cli_approve.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.retraining.approval import PendingApproval
from src.retraining.cli_approve import (
    _cmd_detail,
    _cmd_list,
    _cmd_stale,
    _list_all_pending,
    main,
)


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _write_approval(
    pending_dir: Path,
    parameter_name: str,
    approval_id: str,
    days_ago: int = 1,
    current_rmse: float = 53.48,
    candidate_rmse: float = 50.21,
) -> PendingApproval:
    """
    Write a fake PendingApproval JSON directly to pending_dir.

    Does NOT create a candidate pkl — listing and "n" tests don't need it;
    "y" tests inject a mock record_fn that skips pkl loading.
    """
    pending_dir.mkdir(parents=True, exist_ok=True)
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    improvement = (current_rmse - candidate_rmse) / current_rmse * 100.0

    approval = PendingApproval(
        approval_id             = approval_id,
        parameter_name          = parameter_name,
        model_version           = f"xgb_{parameter_name.lower()}_v1",
        current_rmse            = round(current_rmse,  4),
        candidate_rmse          = round(candidate_rmse, 4),
        improvement_pct         = round(improvement,   2),
        drift_reason            = "RMSE ratio 1.12 exceeded threshold 1.10",
        rmse_ratio              = 1.12,
        ks_pvalue               = 0.03,
        ks_drift                = True,
        training_period         = {
            "n_train_val_rows": 310,
            "total_fe_rows":    303,
            "data_start":       "2025-02-01",
            "data_end":         "2026-01-10",
        },
        candidate_model_path    = str(pending_dir / f"{approval_id}_candidate.pkl"),
        production_models_store = str(pending_dir.parent),
        timestamp_created       = ts,
        status                  = "pending_approval",
        reviewer_note           = "",
        timestamp_decided       = "",
    )

    json_path = pending_dir / f"{approval_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(approval), f, indent=2)

    return approval


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_list_shows_both_parameters():
    """
    Two pending approvals (EC + pH) must both appear in the listing output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir)

        ec_id = "EC_xgb_ec_v1_20260720_143022_aabbcc"
        ph_id = "pH_xgb_ph_v1_20260718_093011_ddeeff"

        _write_approval(store / "ec" / "pending_approvals", "EC", ec_id,
                        current_rmse=53.48, candidate_rmse=50.21)
        _write_approval(store / "ph" / "pending_approvals", "pH", ph_id,
                        current_rmse=0.854, candidate_rmse=0.802)

        output_lines: list[str] = []
        _cmd_list(store, output_fn=output_lines.append)
        output_text = "\n".join(output_lines)

        assert ec_id in output_text, f"EC approval ID not found in listing:\n{output_text}"
        assert ph_id in output_text, f"pH approval ID not found in listing:\n{output_text}"
        assert "2 pending approval(s)" in output_text

        # Both parameter names visible
        assert "EC" in output_text
        assert "pH" in output_text

        # RMSE values visible
        assert "53.4800" in output_text or "53.48" in output_text
        assert "0.8540"  in output_text or "0.854" in output_text

    print("PASS  test_list_shows_both_parameters  (EC + pH listed, RMSE visible)")


def test_approve_y_calls_record_fn():
    """
    Input "y" + "" (empty note) must call record_fn with decision=True
    and the correct approval_id.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store  = Path(tmpdir)
        ec_id  = "EC_xgb_ec_v1_20260720_143022_aabbcc"
        pd_dir = store / "ec" / "pending_approvals"
        _write_approval(pd_dir, "EC", ec_id)

        calls: list[dict] = []

        def mock_record(approval_id, decision, pending_dir, reviewer_note="", **kw):
            calls.append({
                "approval_id":   approval_id,
                "decision":      decision,
                "reviewer_note": reviewer_note,
            })
            return {"promoted": True, "saved_path": "/fake/path/model.pkl"}

        # input_fn returns "y" first (confirm), then "" (empty note)
        responses = iter(["y", ""])
        def fake_input(_prompt=""):
            return next(responses)

        output_lines: list[str] = []
        _cmd_detail(
            ec_id,
            models_store = store,
            input_fn     = fake_input,
            record_fn    = mock_record,
            output_fn    = output_lines.append,
        )

        assert len(calls) == 1, f"record_fn should be called exactly once, got {len(calls)}"
        assert calls[0]["decision"]    is True
        assert calls[0]["approval_id"] == ec_id
        assert "Promoted" in "\n".join(output_lines)

    print("PASS  test_approve_y_calls_record_fn  (record_fn called with decision=True)")


def test_reject_n_does_not_call_record_fn():
    """
    Input "n" must abort without calling record_fn at all.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store  = Path(tmpdir)
        ec_id  = "EC_xgb_ec_v1_20260720_143022_aabbcc"
        pd_dir = store / "ec" / "pending_approvals"
        _write_approval(pd_dir, "EC", ec_id)

        calls: list[dict] = []

        def mock_record(*args, **kw):
            calls.append(args)
            return {}

        responses = iter(["n"])
        def fake_input(_prompt=""):
            return next(responses)

        output_lines: list[str] = []
        _cmd_detail(
            ec_id,
            models_store = store,
            input_fn     = fake_input,
            record_fn    = mock_record,
            output_fn    = output_lines.append,
        )

        assert len(calls) == 0, f"record_fn must NOT be called on 'n', got {len(calls)} call(s)"
        assert "Aborted" in "\n".join(output_lines)

    print("PASS  test_reject_n_does_not_call_record_fn  (no action on 'n')")


def test_stale_listing_filters_by_age():
    """
    --reject-all-stale with stale_days=14 should show an approval created
    20 days ago and NOT show one created 2 days ago.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir)

        old_id  = "EC_xgb_ec_v1_20260701_120000_old111"
        new_id  = "pH_xgb_ph_v1_20260720_100000_new222"

        _write_approval(store / "ec" / "pending_approvals", "EC", old_id, days_ago=20)
        _write_approval(store / "ph" / "pending_approvals", "pH", new_id, days_ago=2)

        output_lines: list[str] = []
        stale = _cmd_stale(store, stale_days=14, output_fn=output_lines.append)
        output_text = "\n".join(output_lines)

        stale_ids = [a.approval_id for a, _ in stale]

        assert old_id in stale_ids, f"20-day-old approval should appear as stale"
        assert new_id not in stale_ids, f"2-day-old approval must NOT appear as stale"
        assert old_id in output_text
        assert new_id not in output_text
        assert "1 stale approval(s)" in output_text
        assert "NOT been rejected automatically" in output_text

    print("PASS  test_stale_listing_filters_by_age  (old shown, recent hidden)")


def test_unknown_id_shows_error():
    """Passing a non-existent ID prints an error without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_lines: list[str] = []
        _cmd_detail(
            "NONEXISTENT_ID_xyz",
            models_store = Path(tmpdir),
            input_fn     = lambda _: "y",
            output_fn    = output_lines.append,
        )
        output_text = "\n".join(output_lines)
        assert "ERROR" in output_text or "not found" in output_text.lower()

    print("PASS  test_unknown_id_shows_error  (graceful error on bad ID)")


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_list_shows_both_parameters,
        test_approve_y_calls_record_fn,
        test_reject_n_does_not_call_record_fn,
        test_stale_listing_filters_by_age,
        test_unknown_id_shows_error,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            print(f"FAIL  {t.__name__}:  {exc}")
            failed += 1
        except Exception as exc:
            import traceback
            print(f"ERROR {t.__name__}:  {exc}")
            traceback.print_exc()
            failed += 1

    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"RESULT: {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
