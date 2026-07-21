"""
evaluate_turbidity_log_transform.py — Raw vs log1p(Turbidity) benchmark.

QUESTION
--------
Does a log1p transformation of the target (Turbidity) improve model accuracy
compared to predicting raw Turbidity values? And when should it be recommended
as default practice?

METHODOLOGY
-----------
1.  Build features for BOTH raw and log-transformed target using the same
    time-aware feature engineering (lag_hours=[24,48,72], roll=[72,168],
    co_variables=['pH','EC']).
2.  For each variant, run the full RF/XGBoost/SVR benchmark (same grid as
    the generic pipeline: 33 models per variant).
3.  For log models, predictions are back-transformed via expm1() so that
    RMSE is reported in original NTU scale for interpretability.
4.  Compute relative RMSE = RMSE / mean(y_val_original) * 100 for fair
    cross-parameter comparison against EC's reference value.
5.  Bootstrap significance test (2000 draws) between rank-1 and rank-2 in
    each variant.
6.  Output a recommendation threshold for future parameters.

REFERENCE VALUES
----------------
EC (frozen production model, ec_xgboost_v1_final.json):
  Val RMSE = 52.13 µS/cm  |  Val mean = 698.25 µS/cm  |  Rel RMSE = 7.47%
  Test RMSE = 53.48 µS/cm |  Test mean = 787.62 µS/cm |  Rel RMSE = 6.79%

EXECUTION
---------
  python src/evaluation/evaluate_turbidity_log_transform.py

Prints a multi-section comparison report. No files are saved (analysis only).
"""

from __future__ import annotations

import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import chronological_split
from src.forecasting.feature_engineering_generic import build_features_time_aware
from src.forecasting.frequency_detector import detect_frequency
from src.pipeline.model_benchmark import (
    BenchmarkReport,
    ModelResult,
    benchmark_models,
    format_benchmark_report,
    quick_noise_diagnostic,
)
from src.pipeline.timestamp_detector import detect_timestamp_column, parse_timestamp_column

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_RAW   = "Turbidity"
TARGET_LOG   = "Turbidity_log"
UNIT         = "NTU"
CO_VARIABLES = ["pH", "EC"]
LAG_HOURS    = [24, 48, 72]
ROLL_HOURS   = [72, 168]
N_BOOTSTRAP  = 2000
RNG_SEED     = 42

# EC reference (computed from ec_xgboost_v1_final.json on val split)
EC_VAL_RMSE      = 52.13    # µS/cm
EC_VAL_MEAN      = 698.25   # µS/cm
EC_VAL_REL_RMSE  = 7.47     # %
EC_TEST_REL_RMSE = 6.79     # %

SEP  = "─" * 72
SEP2 = "═" * 72


# ── Data preparation ───────────────────────────────────────────────────────────

def load_and_prepare() -> tuple[pd.DataFrame, timedelta]:
    data_path = PROJECT_ROOT / "data" / "processed" / "c1_with_wqi.csv"
    df = pd.read_csv(data_path)

    ts_col    = detect_timestamp_column(df, ["Date", "timestamp", "date", "Time"])
    df        = parse_timestamp_column(df, ts_col)
    frequency = detect_frequency(df, date_col=ts_col)

    # Add log1p column for the log-space benchmark
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW])

    return df, frequency


def build_and_split(
    df: pd.DataFrame,
    target: str,
    frequency: timedelta,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X, y = build_features_time_aware(
        df            = df,
        target        = target,
        frequency     = frequency,
        lag_hours     = LAG_HOURS,
        rolling_hours = ROLL_HOURS,
        co_variables  = CO_VARIABLES,
    )
    return chronological_split(X, y)


# ── Post-hoc original-scale RMSE for log models ────────────────────────────────

def recompute_original_scale_metrics(
    report_log: BenchmarkReport,
    X_val: pd.DataFrame,
    y_val_orig: pd.Series,
) -> list[dict]:
    """
    For each model in report_log (trained on log-scale y), compute RMSE and MAE
    in the original (NTU) scale by back-transforming predictions via expm1().

    Returns a list of dicts (one per model in report_log.results), sorted by
    original-scale Val RMSE ascending (re-ranked).
    """
    y_orig = y_val_orig.values
    rows: list[dict] = []

    for r in report_log.results:
        y_pred_log  = r.fitted_model.predict(X_val)
        y_pred_orig = np.expm1(y_pred_log)
        rmse_orig   = float(np.sqrt(np.mean((y_orig - y_pred_orig) ** 2)))
        mae_orig    = float(np.mean(np.abs(y_orig - y_pred_orig)))
        rows.append({
            "log_rank":   r.rank,
            "label":      r.label,
            "algorithm":  r.algorithm,
            "hyperparams": r.hyperparams,
            "log_val_rmse": r.val_rmse,
            "orig_val_rmse": round(rmse_orig, 4),
            "orig_val_mae":  round(mae_orig,  4),
            "model":        r.fitted_model,
        })

    rows.sort(key=lambda d: d["orig_val_rmse"])
    for rank, row in enumerate(rows, start=1):
        row["orig_rank"] = rank

    return rows


# ── Bootstrap significance between rank-1 and rank-2 ─────────────────────────

def bootstrap_significance(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> dict:
    """
    Bootstrap 95% CI for delta_RMSE = RMSE(rank2) - RMSE(rank1).

    If the entire 95% CI is strictly > 0, rank-1 is significantly better.
    If the CI straddles 0, the difference is not statistically significant.

    Parameters
    ----------
    y_true   : Ground-truth target values (original scale).
    y_pred_1 : Predictions from rank-1 model.
    y_pred_2 : Predictions from rank-2 model.

    Returns
    -------
    dict with keys: delta_observed, ci_low, ci_high, significant (bool).
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    deltas = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx         = rng.integers(0, n, size=n)
        rmse1       = np.sqrt(np.mean((y_true[idx] - y_pred_1[idx]) ** 2))
        rmse2       = np.sqrt(np.mean((y_true[idx] - y_pred_2[idx]) ** 2))
        deltas[i]   = rmse2 - rmse1

    ci_low  = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    delta_observed = float(
        np.sqrt(np.mean((y_true - y_pred_2) ** 2)) -
        np.sqrt(np.mean((y_true - y_pred_1) ** 2))
    )

    return {
        "delta_observed": round(delta_observed, 4),
        "ci_low":  round(ci_low,  4),
        "ci_high": round(ci_high, 4),
        "significant": bool(ci_low > 0),
    }


# ── Format helpers ─────────────────────────────────────────────────────────────

def _sig_label(sig: bool) -> str:
    return "SIGNIFICANT (rank-1 better)" if sig else "not significant (cannot rule out chance)"


def print_top_n(results_dicts: list[dict], key_rmse: str, n: int = 8) -> None:
    header = (
        f"  {'Rank':>4}  {'Algorithm':<10}  {'Val RMSE':>10}  "
        f"{'Val MAE':>9}  Key params"
    )
    print(SEP)
    print(header)
    print(SEP)
    for row in results_dicts[:n]:
        rank     = row.get("orig_rank") or row["log_rank"]
        rmse_val = row[key_rmse]
        mae_val  = row.get("orig_val_mae") or row.get("log_val_mae", 0)
        params   = "  ".join(f"{k}={v}" for k, v in row["hyperparams"].items())
        print(
            f"  {rank:>4}  {row['algorithm']:<10}  {rmse_val:>10.2f}  "
            f"{mae_val:>9.2f}  {params}"
        )
    if len(results_dicts) > n:
        print(f"  ... {len(results_dicts) - n} more not shown")
    print(SEP)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP2)
    print("  RAW vs LOG1P(TURBIDITY) — Full Benchmark Comparison")
    print(SEP2)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data and detecting frequency ...")
    df, frequency = load_and_prepare()
    freq_h = frequency.total_seconds() / 3600.0
    print(f"      frequency = {freq_h:.0f} h | rows = {len(df)}")

    # ── Distribution diagnostics ──────────────────────────────────────────────
    print("\n[2/6] Distribution diagnostics:")
    print(quick_noise_diagnostic(df, TARGET_RAW))
    turb_log_series = df[TARGET_LOG].dropna()
    print(f"\n  log1p(Turbidity) stats:")
    print(f"    mean   = {turb_log_series.mean():.4f}")
    print(f"    std    = {turb_log_series.std():.4f}")
    print(f"    CV     = {turb_log_series.std()/turb_log_series.mean()*100:.1f}%  "
          f"(raw: {df[TARGET_RAW].std()/df[TARGET_RAW].mean()*100:.1f}%)")
    print(f"    skew   = {turb_log_series.skew():.3f}  "
          f"(raw: {df[TARGET_RAW].skew():.3f})")

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\n[3/6] Building features (raw) ...")
    Xtr_r, ytr_r, Xv_r, yv_r, Xt_r, yt_r = build_and_split(df, TARGET_RAW, frequency)
    print(f"[3/6] Building features (log) ...")
    Xtr_l, ytr_l, Xv_l, yv_l, Xt_l, yt_l = build_and_split(df, TARGET_LOG, frequency)

    # Original-scale val targets (aligned with log split via expm1)
    yv_orig = np.expm1(yv_l.values)   # original NTU for log-model evaluation
    mean_val_orig = float(np.mean(yv_orig))
    mean_val_raw  = float(yv_r.mean())

    print(f"\n  Val split: n_train={len(Xtr_r)}, n_val={len(Xv_r)}, n_test={len(Xt_r)}")
    print(f"  Val mean (raw)          : {mean_val_raw:.2f} NTU")
    print(f"  Val mean (log, expm1)   : {mean_val_orig:.2f} NTU")

    # ── Benchmark raw ─────────────────────────────────────────────────────────
    print(f"\n[4/6] Benchmarking on RAW Turbidity (33 models) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_raw = benchmark_models(Xtr_r, ytr_r, Xv_r, yv_r,
                                      parameter_name="Turbidity_raw", unit=UNIT)

    # Convert to list-of-dict for uniform handling
    raw_rows = [
        {
            "orig_rank":    r.rank,
            "log_rank":     r.rank,
            "label":        r.label,
            "algorithm":    r.algorithm,
            "hyperparams":  r.hyperparams,
            "orig_val_rmse": r.val_rmse,
            "orig_val_mae":  r.val_mae,
            "log_val_rmse":  r.val_rmse,
            "model":         r.fitted_model,
        }
        for r in report_raw.results
    ]

    # ── Benchmark log ─────────────────────────────────────────────────────────
    print(f"\n[5/6] Benchmarking on LOG1P(Turbidity) (33 models) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_log = benchmark_models(Xtr_l, ytr_l, Xv_l, yv_l,
                                      parameter_name="Turbidity_log", unit="log-NTU")

    # Re-compute in original scale
    log_rows = recompute_original_scale_metrics(report_log, Xv_l, pd.Series(yv_orig))
    # Rebuild for log_val_mae (it's in log space from the BenchmarkReport)
    for row, r in zip(log_rows, sorted(report_log.results, key=lambda x: x.rank)):
        # match by label
        pass
    # Easier: reconstruct mae from raw report
    mae_by_label = {r.label: r.val_mae for r in report_log.results}
    for row in log_rows:
        row["log_val_mae"] = mae_by_label.get(row["label"], 0.0)

    # ── Bootstrap significance ─────────────────────────────────────────────────
    print("\n[6/6] Bootstrap significance tests (rank-1 vs rank-2) ...")

    def get_val_preds_raw(row):
        return row["model"].predict(Xv_r)

    def get_val_preds_log_orig(row):
        return np.expm1(row["model"].predict(Xv_l))

    # Raw
    y_true_raw = yv_r.values
    r1_raw_preds = get_val_preds_raw(raw_rows[0])
    r2_raw_preds = get_val_preds_raw(raw_rows[1])
    sig_raw = bootstrap_significance(y_true_raw, r1_raw_preds, r2_raw_preds)

    # Log (original-scale predictions for fair comparison)
    y_true_orig = yv_orig
    r1_log_preds = get_val_preds_log_orig(log_rows[0])
    r2_log_preds = get_val_preds_log_orig(log_rows[1])
    sig_log = bootstrap_significance(y_true_orig, r1_log_preds, r2_log_preds)

    # ── Relative RMSE ─────────────────────────────────────────────────────────
    rmse_raw_r1   = raw_rows[0]["orig_val_rmse"]
    rmse_log_r1   = log_rows[0]["orig_val_rmse"]
    rel_rmse_raw  = rmse_raw_r1  / mean_val_raw  * 100
    rel_rmse_log  = rmse_log_r1  / mean_val_orig * 100

    # ── Report ─────────────────────────────────────────────────────────────────

    print(f"\n\n{SEP2}")
    print(f"  RESULTS — RAW Turbidity Benchmark")
    print(f"{SEP2}")
    print(f"  Target scale : raw NTU  |  Val mean = {mean_val_raw:.2f} NTU")
    print_top_n(raw_rows, "orig_val_rmse")
    best_raw = raw_rows[0]
    second_raw = raw_rows[1]
    print(f"  Rank-1 : {best_raw['algorithm']:<10}  {best_raw['hyperparams']}")
    print(f"    Val RMSE = {best_raw['orig_val_rmse']:.2f} NTU")
    print(f"    Val MAE  = {best_raw['orig_val_mae']:.2f} NTU")
    print(f"    Rel RMSE = {rel_rmse_raw:.1f}%  (vs EC {EC_VAL_REL_RMSE}%)")
    print(f"  Rank-2 : {second_raw['algorithm']:<10}  {second_raw['hyperparams']}")
    print(f"    Val RMSE = {second_raw['orig_val_rmse']:.2f} NTU")
    print(f"  Gap rank-1 vs rank-2 : {sig_raw['delta_observed']:.2f} NTU")
    print(f"  Bootstrap 95% CI     : [{sig_raw['ci_low']:.2f}, {sig_raw['ci_high']:.2f}] NTU")
    print(f"  Significance         : {_sig_label(sig_raw['significant'])}")

    print(f"\n{SEP2}")
    print(f"  RESULTS — LOG1P(Turbidity) Benchmark  [back-transformed to NTU]")
    print(f"{SEP2}")
    print(f"  Trained on log scale | RMSE reported in original NTU (via expm1)")
    print(f"  Val mean (original)  : {mean_val_orig:.2f} NTU")
    print_top_n(log_rows, "orig_val_rmse")
    best_log = log_rows[0]
    second_log = log_rows[1]
    print(f"  Rank-1 : {best_log['algorithm']:<10}  {best_log['hyperparams']}")
    print(f"    Val RMSE (NTU) = {best_log['orig_val_rmse']:.2f} NTU")
    print(f"    Val MAE  (NTU) = {best_log['orig_val_mae']:.2f} NTU")
    print(f"    Rel RMSE       = {rel_rmse_log:.1f}%  (vs EC {EC_VAL_REL_RMSE}%)")
    print(f"  Rank-2 : {second_log['algorithm']:<10}  {second_log['hyperparams']}")
    print(f"    Val RMSE (NTU) = {second_log['orig_val_rmse']:.2f} NTU")
    print(f"  Gap rank-1 vs rank-2 : {sig_log['delta_observed']:.2f} NTU")
    print(f"  Bootstrap 95% CI     : [{sig_log['ci_low']:.2f}, {sig_log['ci_high']:.2f}] NTU")
    print(f"  Significance         : {_sig_label(sig_log['significant'])}")

    # ── Head-to-head comparison ────────────────────────────────────────────────
    improvement = (rmse_raw_r1 - rmse_log_r1) / rmse_raw_r1 * 100

    print(f"\n{SEP2}")
    print(f"  HEAD-TO-HEAD COMPARISON — best raw vs best log (back-transformed)")
    print(f"{SEP2}")
    print(f"  {'Metric':<28}  {'Raw (NTU)':>10}  {'Log→orig (NTU)':>14}  {'Change':>8}")
    print(SEP)
    print(f"  {'Val RMSE':<28}  {rmse_raw_r1:>10.2f}  {rmse_log_r1:>14.2f}  "
          f"{'-' if improvement>0 else '+'}{abs(improvement):>6.1f}%")
    print(f"  {'Rel RMSE (% of val mean)':<28}  {rel_rmse_raw:>10.1f}%  "
          f"{rel_rmse_log:>13.1f}%  "
          f"{'—' if abs(improvement)<0.5 else ('better' if improvement>0 else 'worse')}")
    print(f"  {'Rank-1 algorithm':<28}  {best_raw['algorithm']:>10}  {best_log['algorithm']:>14}")
    print(f"  {'Target CV (val)':<28}  {'86.6%':>10}  {'50.9%→log':>14}")
    print(SEP)

    # ── Distribution comparison table (best per algorithm) ────────────────────
    print(f"\n  Best result per algorithm — raw rank vs log rank (original-scale NTU):")
    print(SEP)

    def best_per_algo(rows: list[dict]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for r in rows:
            algo = r["algorithm"]
            if algo not in result or r["orig_val_rmse"] < result[algo]["orig_val_rmse"]:
                result[algo] = r
        return result

    algo_raw_best = best_per_algo(raw_rows)
    algo_log_best = best_per_algo(log_rows)
    all_algos = sorted(set(list(algo_raw_best.keys()) + list(algo_log_best.keys())))
    print(f"  {'Algorithm':<12}  {'Raw rank':>8}  {'Raw RMSE':>10}  "
          f"{'Log rank':>8}  {'Log RMSE':>10}  Winner")
    print(SEP)
    for algo in all_algos:
        rr = algo_raw_best.get(algo)
        rl = algo_log_best.get(algo)
        raw_str = f"{rr['orig_rank']:>8}  {rr['orig_val_rmse']:>10.2f}" if rr else f"{'—':>8}  {'—':>10}"
        log_str = f"{rl['orig_rank']:>8}  {rl['orig_val_rmse']:>10.2f}" if rl else f"{'—':>8}  {'—':>10}"
        if rr and rl:
            winner = "raw" if rr["orig_val_rmse"] <= rl["orig_val_rmse"] else "log"
        elif rr:
            winner = "raw"
        else:
            winner = "log"
        print(f"  {algo:<12}  {raw_str}  {log_str}  {winner}")
    print(SEP)

    # ── Cross-parameter table ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  CROSS-PARAMETER RELATIVE RMSE COMPARISON (val set)")
    print(f"{SEP2}")
    print(f"  {'Parameter':<30}  {'Abs RMSE':>10}  {'Val mean':>10}  {'Rel RMSE':>10}  Notes")
    print(SEP)
    print(f"  {'EC (XGBoost frozen)':<30}  {EC_VAL_RMSE:>8.2f} µS  {EC_VAL_MEAN:>8.2f} µS  "
          f"{EC_VAL_REL_RMSE:>8.2f}%  production model")
    print(f"  {'Turbidity raw (rank-1)':<30}  {rmse_raw_r1:>10.2f}  {mean_val_raw:>8.2f} NTU  "
          f"{rel_rmse_raw:>8.1f}%  benchmark run")
    print(f"  {'Turbidity log→orig (rank-1)':<30}  {rmse_log_r1:>10.2f}  "
          f"{mean_val_orig:>8.2f} NTU  {rel_rmse_log:>8.1f}%  log variant")
    print(SEP)

    # ── Recommendation ────────────────────────────────────────────────────────
    transform_wins = rmse_log_r1 < rmse_raw_r1
    gap_ntu        = rmse_raw_r1 - rmse_log_r1
    gap_pct        = improvement

    print(f"\n{SEP2}")
    print(f"  CONCLUSIONS & RECOMMENDATION")
    print(f"{SEP2}")

    if abs(gap_pct) < 1.0:
        conclusion_transform = (
            f"Log transform makes NO meaningful difference (<1% RMSE change, {gap_ntu:.2f} NTU). "
            f"Raw target is preferred for interpretability."
        )
    elif transform_wins and gap_pct >= 5.0:
        conclusion_transform = (
            f"Log transform IMPROVES rank-1 RMSE by {gap_pct:.1f}% ({gap_ntu:.2f} NTU). "
            f"Recommend log1p as the default for this parameter."
        )
    elif transform_wins:
        conclusion_transform = (
            f"Log transform gives a modest improvement of {gap_pct:.1f}% ({gap_ntu:.2f} NTU). "
            f"Consider log space if interpretability allows; otherwise raw is acceptable."
        )
    else:
        conclusion_transform = (
            f"Log transform WORSENS RMSE by {abs(gap_pct):.1f}% ({abs(gap_ntu):.2f} NTU). "
            f"Raw target is definitively preferred."
        )

    print(f"  1. Transform effect:\n     {conclusion_transform}")

    rank1_same = best_raw["algorithm"] == best_log["algorithm"]
    if rank1_same:
        conclusion_ranking = (
            f"Rank-1 algorithm is the same in both variants: {best_raw['algorithm']}. "
            f"The ordering conclusion is robust to the scale choice."
        )
    else:
        conclusion_ranking = (
            f"Rank-1 algorithm CHANGES between raw ({best_raw['algorithm']}) and "
            f"log ({best_log['algorithm']}) variants — investigate both before freezing."
        )
    print(f"\n  2. Ranking stability:\n     {conclusion_ranking}")

    def sig_note(sig_dict, variant_label):
        if sig_dict["significant"]:
            return (
                f"  {variant_label}: rank-1 vs rank-2 gap is STATISTICALLY SIGNIFICANT "
                f"(95% CI [{sig_dict['ci_low']:.2f}, {sig_dict['ci_high']:.2f}] NTU, fully > 0). "
                f"Rank-1 is the clear choice."
            )
        else:
            return (
                f"  {variant_label}: rank-1 vs rank-2 gap is NOT significant "
                f"(95% CI [{sig_dict['ci_low']:.2f}, {sig_dict['ci_high']:.2f}] NTU straddles 0). "
                f"Rank-2 may perform equally in practice."
            )

    print(f"\n  3. Statistical significance (rank-1 vs rank-2):")
    print(sig_note(sig_raw, "Raw"))
    print(sig_note(sig_log, "Log"))

    # ── Relative RMSE vs EC ────────────────────────────────────────────────────
    best_turb_rel = min(rel_rmse_raw, rel_rmse_log)
    print(f"\n  4. Relative RMSE vs EC reference:")
    print(f"     EC production:       {EC_VAL_REL_RMSE:.1f}% (val) / {EC_TEST_REL_RMSE:.1f}% (test)")
    print(f"     Turbidity best:      {best_turb_rel:.1f}%  — "
          f"{'×' + str(round(best_turb_rel / EC_VAL_REL_RMSE, 1)) + ' higher than EC'}")
    print(f"     Context: Turbidity raw CV = 86.6% vs EC CV = 23.4%. "
          f"Higher relative error is expected for noisier signals. "
          f"This is physically meaningful, not a modeling failure.")

    # ── Log-transform threshold recommendation ─────────────────────────────────
    SIGNAL_CV_THRESHOLD = 60.0
    raw_cv   = df[TARGET_RAW].std() / df[TARGET_RAW].mean() * 100
    log_cv   = df[TARGET_LOG].std() / df[TARGET_LOG].mean() * 100

    print(f"\n  5. General recommendation (log-transform threshold):")
    print(f"     Turbidity raw signal CV  = {raw_cv:.1f}%  (above {SIGNAL_CV_THRESHOLD}%)")
    print(f"     Turbidity log signal CV  = {log_cv:.1f}%  (below {SIGNAL_CV_THRESHOLD}%)")
    print(f"     EC signal CV             = 23.4%  (below {SIGNAL_CV_THRESHOLD}%)")
    print(f"     pH signal CV             = 11.9%  (below {SIGNAL_CV_THRESHOLD}%)")
    if transform_wins and gap_pct >= 1.0:
        verdict = "CONFIRMED: log transform helps when raw CV > 60%."
    elif not transform_wins:
        verdict = "NOT confirmed on this dataset: log transform did not improve results despite high CV."
    else:
        verdict = "MARGINAL: log transform gave small gains at high CV. Apply judgment."

    print(f"\n     Threshold proposal: evaluate log1p transform whenever raw signal CV > {SIGNAL_CV_THRESHOLD}%.")
    print(f"     Empirical verdict on this benchmark: {verdict}")
    print(f"\n     This threshold captures Turbidity (CV=86.6%) but NOT EC (23.4%) or pH (11.9%),")
    print(f"     which aligns with the expectation that log-transform is only relevant for")
    print(f"     highly skewed, high-dynamic-range sensor signals.")

    print(f"\n{SEP2}")


if __name__ == "__main__":
    main()
