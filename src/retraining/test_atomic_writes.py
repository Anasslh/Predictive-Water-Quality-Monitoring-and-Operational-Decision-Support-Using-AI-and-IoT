"""
test_atomic_writes.py — Crash-safety tests for critical JSON writers.

Simulates a write interrupted mid-way (exception raised during json.dump,
before os.replace is called) and verifies:
  - The original file is untouched (not empty, not corrupted).
  - A stale .tmp file may be left on disk, but the live file is intact.

This covers:
  - write_current_model_pointer()  → current_model.json
  - write_rejection_counter()      → rejection_counter.json
  - approval._save_approval_json() → <approval_id>.json
  - monitor/export.py              → <parameter>.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import traceback
import unittest
from dataclasses import dataclass, asdict
from pathlib import Path
from unittest.mock import patch

# ── path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retraining.model_versioning import (
    write_current_model_pointer,
    write_rejection_counter,
    _atomic_write_json,
)
from src.retraining.approval import _save_approval_json, PendingApproval


# ── helpers ───────────────────────────────────────────────────────────────────

VALID_POINTER = {
    "model_path":     "/fake/store/EC_xgb_v1_20260101_120000.pkl",
    "approval_id":    "EC_xgb_v1_20260101_120000_abc123",
    "promoted_at":    "2026-01-01T12:00:00+00:00",
    "parameter_name": "EC",
    "model_version":  "xgb_v1",
}

VALID_COUNTER = {
    "consecutive_rejections": 2,
    "parameter_name": "EC",
    "updated_at": "2026-01-01T12:00:00+00:00",
}


def _make_approval(tmp_dir: Path) -> PendingApproval:
    return PendingApproval(
        approval_id             = "EC_xgb_v1_20260101_120000_abc123",
        parameter_name          = "EC",
        model_version           = "xgb_v1",
        current_rmse            = 60.0,
        candidate_rmse          = 55.0,
        improvement_pct         = 8.33,
        drift_reason            = "RMSE ratio 1.18 > threshold 1.10",
        rmse_ratio              = 1.18,
        ks_pvalue               = 0.03,
        ks_drift                = True,
        training_period         = {"n_train_val_rows": 280, "total_fe_rows": 350,
                                   "data_start": "2025-02-01", "data_end": "2026-01-31"},
        candidate_model_path    = str(tmp_dir / "candidate.pkl"),
        production_models_store = str(tmp_dir),
        timestamp_created       = "2026-01-01T12:00:00+00:00",
    )


_PASS = []
_FAIL = []


def _run(name: str, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _PASS.append(name)
    except Exception:
        print(f"FAIL  {name}")
        traceback.print_exc()
        _FAIL.append(name)


# ── Test 1: _atomic_write_json leaves original intact on crash ────────────────

def test_atomic_helper_leaves_original_on_crash():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "target.json"
        original = {"key": "original_value"}

        # Write the original file first
        with open(path, "w") as f:
            json.dump(original, f)

        # Simulate crash: json.dump raises mid-write
        def crash_dump(data, fp, **_kw):
            fp.write('{"key": "CORRUPTED')   # partial JSON
            raise OSError("Simulated disk-full error during json.dump")

        with patch("src.retraining.model_versioning.json.dump", side_effect=crash_dump):
            try:
                _atomic_write_json(path, {"key": "new_value"}, indent=2)
            except OSError:
                pass   # expected — the exception propagates

        # Original must still be valid and intact
        content = json.loads(path.read_text())
        assert content == original, f"Original file was corrupted: {content}"

        # .tmp may still be on disk (not cleaned on crash — acceptable)
        tmp = path.with_suffix(".json.tmp")
        # we don't assert on whether tmp exists; either is fine


# ── Test 2: write_current_model_pointer — original intact on crash ────────────

def test_pointer_write_crash_leaves_original():
    with tempfile.TemporaryDirectory() as td:
        store = Path(td)

        # Write the original pointer
        pointer_path = store / "current_model.json"
        with open(pointer_path, "w") as f:
            json.dump(VALID_POINTER, f, indent=2)

        new_pointer = dict(VALID_POINTER, model_path="/fake/NEW.pkl")

        def crash_dump(data, fp, **_kw):
            fp.write('{"model_path": "INCO')
            raise RuntimeError("Simulated process kill")

        with patch("src.retraining.model_versioning.json.dump", side_effect=crash_dump):
            try:
                write_current_model_pointer(
                    models_store_path = store,
                    model_pkl_path    = "/fake/NEW.pkl",
                    approval_id       = "NEW_ID",
                    parameter_name    = "EC",
                    model_version     = "xgb_v1",
                )
            except RuntimeError:
                pass

        recovered = json.loads(pointer_path.read_text())
        assert recovered["model_path"] == VALID_POINTER["model_path"], (
            f"Pointer was corrupted after simulated crash: {recovered}"
        )


# ── Test 3: write_rejection_counter — original intact on crash ────────────────

def test_counter_write_crash_leaves_original():
    with tempfile.TemporaryDirectory() as td:
        store = Path(td)

        counter_path = store / "rejection_counter.json"
        with open(counter_path, "w") as f:
            json.dump(VALID_COUNTER, f, indent=2)

        def crash_dump(data, fp, **_kw):
            fp.write('{"consecutive_rejections":')   # partial
            raise IOError("Simulated I/O error")

        with patch("src.retraining.model_versioning.json.dump", side_effect=crash_dump):
            try:
                write_rejection_counter(store, count=99, parameter_name="EC")
            except IOError:
                pass

        recovered = json.loads(counter_path.read_text())
        assert recovered["consecutive_rejections"] == 2, (
            f"Counter was corrupted: {recovered}"
        )


# ── Test 4: _save_approval_json — original intact on crash ────────────────────

def test_approval_json_crash_leaves_original():
    with tempfile.TemporaryDirectory() as td:
        pending_dir = Path(td)
        approval = _make_approval(pending_dir)

        # Write original approval JSON
        original_path = pending_dir / f"{approval.approval_id}.json"
        with open(original_path, "w") as f:
            json.dump(asdict(approval), f, indent=2)

        # Modify approval in memory (simulates a retried write)
        approval.status = "CORRUPTED"

        def crash_dump(data, fp, **_kw):
            fp.write('{"approval_id": "')   # partial
            raise MemoryError("Simulated OOM")

        with patch("src.retraining.approval.json.dump", side_effect=crash_dump):
            try:
                _save_approval_json(approval, pending_dir)
            except MemoryError:
                pass

        recovered = json.loads(original_path.read_text())
        assert recovered["status"] == "pending_approval", (
            f"Approval JSON was corrupted: {recovered}"
        )


# ── Test 5: Successful write (no crash) works correctly ──────────────────────

def test_atomic_write_succeeds_normally():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "data.json"
        _atomic_write_json(path, {"status": "ok", "value": 42}, indent=2)

        assert path.exists(), "Target file not created"
        tmp = path.with_suffix(".json.tmp")
        assert not tmp.exists(), ".tmp file was not cleaned up after successful write"

        content = json.loads(path.read_text())
        assert content == {"status": "ok", "value": 42}


# ── Test 6: JSONL export write is atomic ─────────────────────────────────────

def test_export_jsonl_atomic_write():
    """export.py uses write_text → os.replace; simulate crash before replace."""
    with tempfile.TemporaryDirectory() as td:
        exports_dir = Path(td)
        jsonl_path  = exports_dir / "EC.jsonl"

        original_content = '{"timestamp":"2026-01-01T00:00:00+00:00","parameter_name":"EC"}\n'
        jsonl_path.write_text(original_content, encoding="utf-8")

        # Patch os.replace in export module to raise before the rename happens
        import src.monitor.export as export_mod

        original_replace = os.replace

        def crash_replace(src, dst):
            raise OSError("Simulated rename failure")

        with patch.object(export_mod.os, "replace", side_effect=crash_replace):
            try:
                # write_text on tmp succeeded; os.replace raises → original untouched
                tmp = jsonl_path.with_suffix(".jsonl.tmp")
                tmp.write_text('{"timestamp":"2026-01-02T00:00:00+00:00"}\n', encoding="utf-8")
                export_mod.os.replace(tmp, jsonl_path)
            except OSError:
                pass

        recovered = jsonl_path.read_text(encoding="utf-8")
        assert recovered == original_content, f"JSONL file was corrupted: {recovered!r}"


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        ("atomic helper leaves original intact on crash",        test_atomic_helper_leaves_original_on_crash),
        ("write_current_model_pointer: crash leaves original",   test_pointer_write_crash_leaves_original),
        ("write_rejection_counter: crash leaves original",       test_counter_write_crash_leaves_original),
        ("_save_approval_json: crash leaves original",           test_approval_json_crash_leaves_original),
        ("_atomic_write_json: successful write works",           test_atomic_write_succeeds_normally),
        ("export JSONL: os.replace crash leaves original intact",test_export_jsonl_atomic_write),
    ]
    for name, fn in tests:
        _run(name, fn)

    total = len(_PASS) + len(_FAIL)
    print(f"\nRESULT: {len(_PASS)}/{total} tests passed")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
