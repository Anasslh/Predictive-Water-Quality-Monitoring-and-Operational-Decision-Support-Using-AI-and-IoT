"""
test_status_pending.py — Verify that 'run.py status' correctly surfaces
parameters whose benchmark is done but whose model has not yet been frozen.

Scenario tested
---------------
After 'onboard --all' with no subsequent 'review', models_store/<param>/
contains a pending_benchmark_reports/*.pkl entry but NO current_model.json.
'status' must:
  1. NOT print "No models found" or suggest 'onboard --parameter'.
  2. Print "AWAITING REVIEW" with the parameter count.
  3. List each pending parameter by name.
  4. Show the exact 'python run.py review <param>' command for each.
  5. Handle a mix: one frozen param + one pending → show both sections.
  6. Return cleanly (no sys.exit) when only pending params exist.

Run:  python src/pipeline/test_status_pending.py
      (also picked up automatically by run_tests.sh)
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Test helpers ───────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    symbol = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{symbol}  {label}{suffix}")
    _results.append((label, condition))


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _plant_pending_report(store_root: Path, param: str) -> None:
    """Create minimal pending_benchmark_reports/*.pkl + *.json to simulate onboard --all."""
    import json as _json
    pdir = store_root / param.lower() / "pending_benchmark_reports"
    pdir.mkdir(parents=True, exist_ok=True)
    fake_payload = {"param": param, "stub": True}
    ts = "20260101_000000"
    (pdir / f"{ts}.pkl").write_bytes(pickle.dumps(fake_payload))
    summary = {"parameter_name": param, "saved_at": "2026-01-01T00:00:00+00:00"}
    (pdir / f"{ts}.json").write_text(_json.dumps(summary), encoding="utf-8")


def _plant_frozen_model(store_root: Path, param: str) -> None:
    """Create a minimal current_model.json to simulate a frozen model."""
    import json
    mdir = store_root / param.lower()
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "current_model.json").write_text(
        json.dumps({"model_id": f"rf_{param}_stub", "variant": "raw"}),
        encoding="utf-8",
    )


def _run_status(store_root: Path, parameter: str | None = None) -> str:
    """Invoke _cmd_status and capture its stdout."""
    from run import _cmd_status

    ns = argparse.Namespace(
        models_store = str(store_root),
        parameter    = parameter,
        system_config  = str(ROOT / "config" / "system_config.json"),
        sensors_config = str(ROOT / "config" / "sensors_config.json"),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            _cmd_status(ns)
        except SystemExit:
            pass
    return buf.getvalue()


# ── Test 1: only pending params → AWAITING REVIEW shown, no "No models found" ─

def test_only_pending_shows_awaiting_review() -> None:
    with tempfile.TemporaryDirectory(prefix="test_status_pend_") as tmp:
        store = Path(tmp)
        _plant_pending_report(store, "EC")
        _plant_pending_report(store, "pH")

        out = _run_status(store)

    check(
        "AWAITING REVIEW header present",
        "AWAITING REVIEW" in out,
        repr(out[:200]),
    )
    check(
        "parameter count shows 2",
        "2 parameter(s)" in out,
        repr(out[:200]),
    )
    check(
        "EC listed in output",
        "EC" in out,
    )
    check(
        "pH listed in output",
        "pH" in out,
    )
    check(
        "review command for EC present",
        "python run.py review EC" in out or "python run.py review ec" in out,
        repr(out),
    )
    check(
        "review command for pH present",
        "python run.py review pH" in out or "python run.py review ph" in out,
        repr(out),
    )
    check(
        "'No models found' NOT printed",
        "No models found" not in out,
    )
    check(
        "'onboard --parameter' NOT suggested",
        "onboard --parameter" not in out,
    )


# ── Test 2: empty store → old "No models found" message still shown ────────────

def test_empty_store_shows_no_models() -> None:
    with tempfile.TemporaryDirectory(prefix="test_status_empty_") as tmp:
        store = Path(tmp)
        out = _run_status(store)

    check(
        "No models found message shown for empty store",
        "No models found" in out,
        repr(out[:200]),
    )
    check(
        "onboard suggestion present for empty store",
        "onboard" in out,
    )


# ── Test 3: mixed — one frozen + one pending → both sections shown ─────────────

def test_mixed_frozen_and_pending() -> None:
    with tempfile.TemporaryDirectory(prefix="test_status_mix_") as tmp:
        store = Path(tmp)
        _plant_frozen_model(store, "EC")
        _plant_pending_report(store, "pH")

        out = _run_status(store)

    check(
        "SYSTEM STATUS section present (frozen param)",
        "SYSTEM STATUS" in out,
        repr(out[:300]),
    )
    check(
        "AWAITING REVIEW section present (pending param)",
        "AWAITING REVIEW" in out,
        repr(out[:300]),
    )
    check(
        "review command for pH shown in mixed output",
        "python run.py review pH" in out or "python run.py review ph" in out,
        repr(out),
    )


# ── Test 4: --parameter for a pending param → AWAITING REVIEW shown ───────────

def test_single_parameter_pending() -> None:
    with tempfile.TemporaryDirectory(prefix="test_status_single_") as tmp:
        store = Path(tmp)
        _plant_pending_report(store, "Turbidity")

        out = _run_status(store, parameter="Turbidity")

    check(
        "AWAITING REVIEW shown for single pending param",
        "AWAITING REVIEW" in out,
        repr(out[:200]),
    )
    check(
        "review command for Turbidity shown",
        "python run.py review Turbidity" in out
        or "python run.py review turbidity" in out,
        repr(out),
    )


# ── Test 5: frozen param has NO pending section ────────────────────────────────

def test_frozen_only_no_pending_section() -> None:
    with tempfile.TemporaryDirectory(prefix="test_status_frz_") as tmp:
        store = Path(tmp)
        _plant_frozen_model(store, "EC")

        out = _run_status(store)

    check(
        "AWAITING REVIEW NOT shown when all params are frozen",
        "AWAITING REVIEW" not in out,
        repr(out[:300]),
    )
    check(
        "SYSTEM STATUS shown when param is frozen",
        "SYSTEM STATUS" in out,
        repr(out[:300]),
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64
    print(f"\n{SEP2}")
    print("  status pending-review — end-to-end tests")
    print(SEP2)

    test_only_pending_shows_awaiting_review()
    print(SEP)
    test_empty_store_shows_no_models()
    print(SEP)
    test_mixed_frozen_and_pending()
    print(SEP)
    test_single_parameter_pending()
    print(SEP)
    test_frozen_only_no_pending_section()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(SEP)
    print(f"  {passed}/{total} passed.")
    print(SEP2)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
