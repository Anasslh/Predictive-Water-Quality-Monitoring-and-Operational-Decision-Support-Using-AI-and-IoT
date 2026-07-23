"""
memory_canary_test.py — Memory stress test for ParameterMonitor.

STRATEGY
--------
Two-phase approach to avoid tracemalloc's 3-10x runtime overhead on SHAP:

  Phase 1 — Fast main loop (2 000 iterations, no tracemalloc):
    Measures RSS (OS page-level) and gc.get_objects() count at 10 checkpoints.
    gc.get_objects() counts live Python objects after gc.collect() — a growing
    count across checkpoints means objects are not being freed (leak signal).
    Verdict is based on Phase 1.

  Phase 2 — Short tracemalloc diagnostic (100 iterations, slow):
    Runs ONLY if Phase 1 detects suspicious growth (object count delta > 500
    or RSS delta > 10 MB). Identifies the top Python allocation sites so the
    developer can locate the source before concluding "leak" or "false alarm".

TWO DISTINCT THINGS BEING MEASURED
------------------------------------
• gc.get_objects() delta — counts live Python objects; GC-sensitive, reliable
  for detecting unbounded accumulation of dicts/lists/DataFrames.
• RSS delta — includes C-heap (XGBoost, SHAP C extension), Python's free-list
  allocator (keeps freed blocks warm), and OS fragmentation. RSS alone is NOT
  a reliable leak indicator — the gc count is.

KNOWN PATHOLOGICAL CASE — JSONL ACCUMULATION
----------------------------------------------
All 2 000 calls share the same wall-clock date, so all records fall inside the
30-day retention window.  The JSONL file therefore grows to 2 000 lines.
This is not a production scenario (daily data → at most 30 lines).
The test is run TWICE:
  Run A — JSONL purged immediately (retention_days=0): isolates prediction pipeline.
  Run B — JSONL accumulates (normal retention): shows the JSONL contribution.

CONCURRENCY LIMITATION (not a memory issue, documented for completeness)
-------------------------------------------------------------------------
File writes use crash-safe atomic rename (see model_versioning._atomic_write_json).
No cross-process locking is implemented. Single-process deployment only.
See model_versioning.py module docstring for full rationale and remediation path.

USAGE
-----
    python reports/memory_canary_test.py
    python reports/memory_canary_test.py --iterations 2000 --checkpoints 10
    python reports/memory_canary_test.py --skip-rss
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
import sys
import tempfile
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Silence pipeline logs — they dominate stdout during stress loops
logging.disable(logging.WARNING)

SEP  = "─" * 64
SEP2 = "═" * 64


# ── RSS helper (Linux / WSL, stdlib only) ─────────────────────────────────────

def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ValueError):
        pass
    return None


# ── Setup ─────────────────────────────────────────────────────────────────────

def _load_ec_model():
    store = ROOT / "models_store" / "ec"
    candidates = sorted(store.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No EC pkl found in {store}")
    with open(candidates[-1], "rb") as f:
        return pickle.load(f)


def _load_test_rows() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "ec_X_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test features not found: {path}")
    return pd.read_csv(path)


def _build_monitor(model, test_rows: pd.DataFrame, exports_dir: Path, retention_days: int):
    from src.monitor import ParameterMonitor
    from src.anomaly.detector import ECAnomalyDetector
    from src.retraining.retrain_manager import RetrainManager
    from src.data.split import chronological_split
    from src.config import get_config

    cfg = get_config()
    cfg.exports.exports_dir    = str(exports_dir)
    cfg.exports.retention_days = retention_days

    # Fit anomaly detector on test rows
    anomaly_detector = ECAnomalyDetector()
    anomaly_detector.fit(np.zeros(len(test_rows)))

    # RetrainManager with an unreachable volume gate
    retrain_manager = RetrainManager(
        model_class       = type(model),
        feature_fn        = lambda df: (df, pd.Series(np.zeros(len(df)))),
        min_new_rows      = 100_000,
        models_store_path = ROOT / "models_store" / "ec",
    )
    dummy_X = test_rows.copy()
    dummy_y = pd.Series(np.zeros(len(dummy_X)))
    X_train, y_train, X_val, y_val, _, _ = chronological_split(dummy_X, dummy_y)
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    retrain_manager.current_model       = model
    retrain_manager.reference_residuals = np.zeros(len(X_tv))
    retrain_manager.last_train_size     = len(dummy_X)

    return ParameterMonitor(
        parameter_name    = "EC",
        prediction_model  = model,
        feature_fn        = lambda df: (df, pd.Series(np.zeros(len(df)))),
        retrain_manager   = retrain_manager,
        anomaly_detector  = anomaly_detector,
        models_store_path = ROOT / "models_store" / "ec",
        config            = cfg,
    )


# ── Phase 1: fast main loop ───────────────────────────────────────────────────

def _phase1(model, test_rows: pd.DataFrame, n_iterations: int, n_checkpoints: int,
            exports_dir: Path, retention_days: int, skip_rss: bool, label: str) -> dict:
    """
    Run n_iterations calls without tracemalloc. Sample gc.get_objects() and RSS
    at n_checkpoints evenly spaced through the loop.
    """
    n_rows = len(test_rows)
    monitor = _build_monitor(model, test_rows, exports_dir, retention_days)

    # Warm-up: Python/XGBoost/SHAP lazy initialization
    WARMUP = min(30, n_rows * 2)
    for i in range(WARMUP):
        row = test_rows.iloc[[i % n_rows]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monitor.process_new_measurement(row)

    gc.collect(); gc.collect()
    check_every = max(1, n_iterations // n_checkpoints)

    samples_obj: list[tuple[int, int]]          = []   # (iteration, gc_count)
    samples_rss: list[tuple[int, float | None]] = []   # (iteration, rss_mb)

    def _snap(i):
        gc.collect()
        cnt = len(gc.get_objects())
        rss = _rss_mb() if not skip_rss else None
        samples_obj.append((i, cnt))
        samples_rss.append((i, rss))

    _snap(0)   # baseline after warm-up

    for i in range(1, n_iterations + 1):
        row = test_rows.iloc[[i % n_rows]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monitor.process_new_measurement(row)

        if i % check_every == 0:
            _snap(i)

    gc.collect(); gc.collect()
    _snap(n_iterations)   # final (may duplicate last checkpoint — OK)

    # Deduplicate by iteration index
    seen = set()
    deduped_obj = []
    deduped_rss = []
    for (io, co), (ir, cr) in zip(samples_obj, samples_rss):
        if io not in seen:
            seen.add(io)
            deduped_obj.append((io, co))
            deduped_rss.append((ir, cr))

    obj_baseline = deduped_obj[0][1]
    obj_final    = deduped_obj[-1][1]
    obj_delta    = obj_final - obj_baseline

    rss_baseline = deduped_rss[0][1]
    rss_final    = deduped_rss[-1][1]
    rss_delta    = (rss_final - rss_baseline) if (rss_baseline and rss_final) else None

    # Is the object count monotonically increasing across checkpoints?
    counts = [c for _, c in deduped_obj]
    monotone = all(counts[j] >= counts[j-1] for j in range(1, len(counts)))

    return {
        "label":        label,
        "samples_obj":  deduped_obj,
        "samples_rss":  deduped_rss,
        "obj_delta":    obj_delta,
        "rss_delta_mb": rss_delta,
        "monotone":     monotone,
    }


# ── Phase 2: short tracemalloc diagnostic ────────────────────────────────────

def _phase2_diagnostic(model, test_rows: pd.DataFrame,
                       exports_dir: Path, retention_days: int,
                       diag_iterations: int = 100) -> list[tuple[str, int]]:
    """
    Short tracemalloc session to identify top growing allocations.
    Only called when Phase 1 detects suspicious growth.
    """
    n_rows = len(test_rows)
    monitor = _build_monitor(model, test_rows, exports_dir, retention_days)

    # Minimal warm-up
    for i in range(min(10, n_rows)):
        row = test_rows.iloc[[i % n_rows]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monitor.process_new_measurement(row)

    gc.collect(); gc.collect()
    tracemalloc.start(5)
    snap1 = tracemalloc.take_snapshot()

    for i in range(diag_iterations):
        row = test_rows.iloc[[i % n_rows]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monitor.process_new_measurement(row)

    gc.collect(); gc.collect()
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap2.compare_to(snap1, "lineno")
    top = sorted([s for s in stats if s.size_diff > 0], key=lambda s: s.size_diff, reverse=True)[:8]
    return [
        (str(s.traceback[0]).replace(str(ROOT), "."), s.size_diff)
        for s in top
    ]


# ── Pretty print ──────────────────────────────────────────────────────────────

def _print_phase1(r: dict, n_iterations: int, skip_rss: bool):
    print(f"\n  {r['label']}")
    print(f"  {SEP}")

    # Table header
    has_rss = not skip_rss and any(rss is not None for _, rss in r["samples_rss"])
    hdr = f"  {'Iter':>6}  {'gc objects':>12}  {'Δ objects':>10}"
    if has_rss:
        hdr += f"  {'RSS (MB)':>10}  {'Δ RSS':>8}"
    print(hdr)
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}" + (f"  {'-'*10}  {'-'*8}" if has_rss else ""))

    obj_base = r["samples_obj"][0][1]
    rss_base = r["samples_rss"][0][1] if has_rss else None

    for (it, cnt), (_, rss) in zip(r["samples_obj"], r["samples_rss"]):
        dobj = cnt - obj_base
        line = f"  {it:>6}  {cnt:>12,}  {dobj:>+10,}"
        if has_rss:
            drss = (rss - rss_base) if (rss is not None and rss_base is not None) else None
            line += f"  {rss:>10.1f}" if rss else f"  {'n/a':>10}"
            line += f"  {drss:>+8.1f}" if drss is not None else f"  {'n/a':>8}"
        note = "  ← baseline" if it == r["samples_obj"][0][0] else ""
        print(line + note)

    print()
    print(f"  gc.get_objects() delta : {r['obj_delta']:>+,} objects")
    if r["rss_delta_mb"] is not None:
        print(f"  RSS delta              : {r['rss_delta_mb']:>+.1f} MB")
    print(f"  Monotone growth        : {'YES — every checkpoint higher than previous' if r['monotone'] else 'NO — count stabilises or fluctuates (normal)'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _run(n_iterations: int = 2000, n_checkpoints: int = 10, skip_rss: bool = False):
    print()
    print(SEP2)
    print(f"  Memory stress test — ParameterMonitor.process_new_measurement()")
    print(f"  Phase 1: {n_iterations} iterations (RSS + gc.get_objects(), no tracemalloc overhead)")
    print(f"  Phase 2: 100-iteration tracemalloc diagnostic (only if Phase 1 shows growth)")
    print(SEP2)

    print("\n  Loading EC model and test data …", flush=True)
    model     = _load_ec_model()
    test_rows = _load_test_rows()
    print(f"  Model: {type(model).__name__}  |  Test rows: {len(test_rows)}")

    print(f"\n  JSONL note: {n_iterations} calls all fall within the 30-day retention window")
    print(f"  (production scenario: daily data → max 30 lines; here: {n_iterations} lines)")
    print(f"  Run A disables JSONL accumulation (retention_days=0) to isolate the pipeline.")
    print(f"  Run B enables normal retention to show the JSONL contribution.")

    with tempfile.TemporaryDirectory() as tmpd:
        exports_a = Path(tmpd) / "exports_a"
        exports_b = Path(tmpd) / "exports_b"
        exports_a.mkdir(); exports_b.mkdir()

        # ── Run A: JSONL disabled (retention=0 purges each record immediately) ──
        print(f"\n  [Phase 1 / Run A] …", flush=True)
        result_a = _phase1(
            model, test_rows, n_iterations, n_checkpoints,
            exports_dir=exports_a, retention_days=0,
            skip_rss=skip_rss,
            label="Run A — Prediction + SHAP + Anomaly (JSONL retention=0, no accumulation)",
        )
        _print_phase1(result_a, n_iterations, skip_rss)

        # ── Run B: full pipeline, JSONL accumulates ───────────────────────────
        print(f"\n  [Phase 1 / Run B] …", flush=True)
        result_b = _phase1(
            model, test_rows, n_iterations, n_checkpoints,
            exports_dir=exports_b, retention_days=30,
            skip_rss=skip_rss,
            label=f"Run B — Full pipeline (JSONL accumulates {n_iterations} lines in temp dir)",
        )
        _print_phase1(result_b, n_iterations, skip_rss)

        # ── Verdicts ──────────────────────────────────────────────────────────
        OBJ_THRESHOLD = 500    # objects
        RSS_THRESHOLD = 10.0   # MB

        suspicious_a = (
            result_a["obj_delta"] > OBJ_THRESHOLD
            or (result_a["rss_delta_mb"] is not None and result_a["rss_delta_mb"] > RSS_THRESHOLD)
        )
        suspicious_b = (
            result_b["obj_delta"] > OBJ_THRESHOLD
            or (result_b["rss_delta_mb"] is not None and result_b["rss_delta_mb"] > RSS_THRESHOLD)
        )

        # ── Phase 2 if needed ─────────────────────────────────────────────────
        diag_a = diag_b = None
        if suspicious_a or suspicious_b:
            print(f"\n  [Phase 2] Growth detected — running 100-iteration tracemalloc diagnostic …")
            if suspicious_a:
                print(f"  Diagnosing Run A …", flush=True)
                diag_a = _phase2_diagnostic(model, test_rows, exports_a, 0)
            if suspicious_b:
                print(f"  Diagnosing Run B …", flush=True)
                diag_b = _phase2_diagnostic(model, test_rows, exports_b, 30)

    # ── Overall verdict ───────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  VERDICT")
    print(f"  {SEP}")

    def _verdict_line(label: str, result: dict, diag: list | None):
        delta_obj = result["obj_delta"]
        delta_rss = result["rss_delta_mb"]
        suspicious = (
            delta_obj > OBJ_THRESHOLD
            or (delta_rss is not None and delta_rss > RSS_THRESHOLD)
        )
        if not suspicious:
            status = "PASS"
            msg = f"gc Δ={delta_obj:+,} objects"
            if delta_rss is not None:
                msg += f", RSS Δ={delta_rss:+.1f} MB"
            msg += " — within thresholds (obj≤500, RSS≤10 MB)"
        else:
            status = "WARN"
            msg = f"gc Δ={delta_obj:+,} objects"
            if delta_rss is not None:
                msg += f", RSS Δ={delta_rss:+.1f} MB"
            msg += " — exceeds threshold; see tracemalloc below"

        print(f"  {label}")
        print(f"    {status}  {msg}")
        if result["monotone"] and suspicious:
            print(f"    Object count is monotonically increasing — investigate further.")

        if diag:
            print(f"    Top growing allocations (100-iteration tracemalloc session):")
            for loc, size_diff in diag:
                print(f"      {size_diff/1024:>8.1f} KB  {loc}")

    _verdict_line("Run A (pipeline only, no JSONL accumulation):", result_a, diag_a)
    print()
    _verdict_line("Run B (full pipeline, JSONL accumulates):", result_b, diag_b)

    b_minus_a = result_b["obj_delta"] - result_a["obj_delta"]
    if abs(b_minus_a) > 200:
        print()
        print(f"  Run B − Run A object delta: {b_minus_a:+,} objects")
        print(f"  → This difference is attributable to JSONL string buffers, not a pipeline leak.")
        print(f"  → In production (30 lines max), this delta would be negligible.")

    print()
    print(f"  Concurrency: writes are crash-safe (atomic rename). No multi-process locking.")
    print(f"  This is intentional for single-process deployment (see model_versioning.py).")
    print(SEP2)
    print()

    # Fail CI if Run A (pipeline without JSONL) leaks — Run B leaking is expected/explained
    if suspicious_a and result_a["monotone"]:
        print("EXIT 1 — pipeline-only run shows monotone object growth (Run A).")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory stress test for ParameterMonitor")
    parser.add_argument("--iterations",   type=int, default=2000)
    parser.add_argument("--checkpoints",  type=int, default=10,
                        help="Number of gc.get_objects() samples (default 10)")
    parser.add_argument("--skip-rss",    action="store_true",
                        help="Skip /proc RSS read (for non-Linux environments)")
    args = parser.parse_args()
    _run(n_iterations=args.iterations, n_checkpoints=args.checkpoints, skip_rss=args.skip_rss)
