# Monitor Stages — Operational Guide

`python run.py monitor` runs up to four pipeline stages each time a new sensor
measurement arrives. This guide explains which stages are active by default, why
some are opt-in, and how to control them.

---

## Stage overview

| # | Stage | Default | Flag to disable |
|---|-------|---------|-----------------|
| 1 | **Predict + SHAP** | always on | — |
| 2 | **Anomaly detection** | on (if detector exists) | `--no-anomaly` |
| 3 | **Retrain check** | **off** | `--check-retrain` to enable |
| 4 | **Multi-step forecast** | on (if config exists) | `--no-forecast` |

Stages 2 and 4 activate automatically once the corresponding artifact has been
trained (detector / forecaster config are both saved the first time you run
`onboard` or `review`). Stage 1 is unconditional.

---

## Stage 2 — Anomaly detection (default ON)

Requires `--actual-value` to compute the residual. Compares the residual against
the Isolation Forest trained on the train+val residual distribution. Returns a
score in [0, 1]; measurements with score ≥ 0.5 (configurable in
`system_config.json → anomaly_detection.score_threshold`) are flagged.

Because anomaly scoring is pure inference (no model training), it adds only a
few milliseconds. It is therefore safe to run on every measurement.

**Console output example:**
```
  ⚠ ANOMALY DETECTED — score=0.987 (threshold=0.500)
```
or, when normal:
```
  Anomaly check   : OK  (score=0.041, threshold=0.500)
```

**Missing detector:**
```
  Anomaly detection: not yet trained — skipping
```
Run `onboard` or `review` once to generate `anomaly_detector.pkl`.

---

## Stage 3 — Retrain check (default OFF, opt-in with `--check-retrain`)

Retrain check is disabled by default because it can take several seconds: it
runs a KS drift test on the new residual distribution versus the reference
distribution, and if drift is detected it trains a new candidate model on the
full accumulated dataset. In a live demo or a latency-sensitive deployment,
triggering this unexpectedly is disruptive.

Enable it explicitly when you have accumulated enough new measurements and want
to check whether the production model needs refreshing:

```bash
python run.py monitor \
    --parameter EC \
    --dataset data/processed/c1_with_wqi.csv \
    --new-row data/processed/ec_new_row_example.csv \
    --actual-value 231.4 \
    --check-retrain
```

**Console output when drift is not detected:**
```
  Retrain check   : drift_not_detected
```

**Console output when drift is detected and a candidate is queued:**
```
  Retrain alert   : Drift detected — candidate queued for review (approval_id: ...)
```

The candidate must then be reviewed with `python run.py approve`.

---

## Stage 4 — Multi-step forecast (default ON)

Recursively applies the frozen prediction model to forecast the next N steps.
`n_steps` is derived from the sensor's `forecast_horizon_hours` field in
`sensors_config.json` divided by the auto-detected measurement frequency:

```
n_steps = floor(forecast_horizon_hours / detected_freq_hours)   (minimum 1)
```

| Sensor type | `forecast_horizon_hours` | `n_steps` |
|-------------|--------------------------|-----------|
| Daily (24 h/step) | 72 (default) | **3** |
| Hourly (1 h/step) | 72 (default) | **72** |
| Daily, long horizon | 168 | 7 |

Uncertainty bands are empirical standard deviations estimated from the
validation set at training time, scaled by √step as a random-walk
approximation.

Forecasting is inference-only (no training), so it is fast enough to run on
every measurement. It activates automatically once a `forecaster_config.json`
exists in `models_store/<param>/` — generated automatically by `onboard` or
`review`.

**Console output example (daily EC sensor, default 3 steps):**
```
  Forecast (3 steps, 24h interval):
    D+1     756.58  ± 46.59
    D+2     756.58  ± 65.88
    D+3     756.58  ± 80.69
```

> **Note on flat forecasts:** tree-based models (XGBoost, RF) frequently
> produce identical predictions at all forecast steps because adjacent feature
> vectors fall in the same leaf nodes — this is a known property of recursive
> tree forecasting, not a pipeline bug. The uncertainty bands still widen
> correctly via the √step approximation.

---

## Simplified mode — prediction + SHAP only

For a first demo or when you just want the raw prediction without any
secondary stages:

```bash
python run.py monitor \
    --parameter EC \
    --dataset data/processed/c1_with_wqi.csv \
    --new-row data/processed/ec_new_row_example.csv \
    --no-anomaly \
    --no-forecast
```

This runs Stage 1 only (predict + SHAP). The JSONL export still fires.

---

## Artifacts required per stage

| Stage | File created by `onboard`/`review` |
|-------|------------------------------------|
| Anomaly detector | `models_store/<param>/anomaly_detector.pkl` |
| Forecaster | `models_store/<param>/forecaster_config.json` |
| Retrain manager | Built in-memory from `models_store/<param>/current_model.json` + dataset |

All artifacts are generated automatically the first time you run
`python run.py onboard --parameter <name>` or `python run.py review <name>`.

---

## Data layers — three complementary stores

Each measurement processed by `monitor` is written to up to three
independent stores. They differ in retention, schema, and use case:

| Layer | Location | Schema | Retention | Updated by |
|---|---|---|---|---|
| **Raw dataset** | `data/processed/<param>.csv` | measurements only (no predictions) | permanent | data team / ingestion |
| **Hot export** | `exports/<param>.jsonl` | full record (prediction + SHAP + anomaly) | **30-day rolling window** — older lines purged automatically | `process_new_measurement()` |
| **Cold archive** | `archive/<param>_<YYYY-MM>.jsonl[.gz]` | same full record as Hot | **permanent** — no purge, monthly compression | `process_new_measurement()` |

### Raw dataset

Contains raw sensor readings as collected — no model outputs. Lives in
`data/processed/` and is never modified by the pipeline. Use it to
re-train or re-benchmark from scratch.

### Hot export (`exports/`)

Designed for live dashboard consumption. Always contains at most 30 days
of data, kept in a single flat `.jsonl` per parameter. Atomic writes
(`.tmp` → `os.replace`) guarantee the IE dashboard can read it at any
time without seeing a partial file. Purge is timestamp-based (the
`"timestamp"` field inside each line), not modification-time-based.

### Cold archive (`archive/`)

Permanent audit trail. Same record schema as the Hot export. Two file
states:

- **Current month** — `archive/EC_2026-08.jsonl` (plain text, growing).
- **Past months** — `archive/EC_2026-07.jsonl.gz` (gzip, read-only).

Rotation is automatic: before each write, any plain-text `.jsonl` whose
month is earlier than the current calendar month is compressed to `.gz`
and the original removed. A typical gzip-compressed month of daily data
weighs **< 5 KB**; hourly data for a full year stays well below **5 GB**.

**Reading across months:**

```python
from src.monitor.archive import read_archive

# All records for EC in June + July 2026
records = read_archive(
    "EC",
    archive_dir="archive",
    start_date="2026-06-01",
    end_date="2026-07-31T23:59:59Z",
)
```

`read_archive()` decompresses `.gz` files in memory and applies
timestamp-based filtering — no temp files, no disk extraction.

**Listing archive files:**

```python
from src.monitor.archive import list_archive_files

for f in list_archive_files("EC"):
    print(f["month"], "gz=" + str(f["compressed"]), f["size_bytes"], "B")
```

**Configuring the archive directory:**

```json
// config/system_config.json
"exports": {
    "exports_dir":    "exports",
    "retention_days": 30,
    "archive_dir":    "archive"
}
```

The default `"archive"` puts the archive alongside `exports/` at the
project root. Change it to an external path (network drive, S3-mounted
directory) without touching any other code.

---

## IE team: bootstrapping the Warm tier from the Cold archive

> **Warm tier** = the IE dashboard's 12-month rolling window. It needs
> predictions + SHAP + anomaly scores — not just raw sensor readings.
> The Cold archive (`archive/`) is the authoritative source for this data.

Use the `export-archive` command to export any time range from the Cold
archive into a single, self-contained JSONL file for validated SQL Server
ingestion. The dashboard does not read this intermediate file directly. The
command handles compressed and uncompressed monthly
files transparently — no knowledge of gzip or the month-per-file layout
is needed on the IE side.

**Recommended: last 12 months**

```bash
python run.py export-archive \
    --parameter EC \
    --months 12 \
    --output warm_tier/EC_last_12m.jsonl
```

**Or with explicit dates:**

```bash
python run.py export-archive \
    --parameter pH \
    --start-date 2025-08-01 \
    --end-date   2026-07-31 \
    --output warm_tier/pH_fy2026.jsonl
```

**Several configured parameters (shell example):**

```bash
for PARAM in EC pH Turbidity; do
    python run.py export-archive \
        --parameter "$PARAM" \
        --months 12 \
        --output "warm_tier/${PARAM}_last_12m.jsonl"
done
```

**Output format:** one JSON object per line, oldest record first.
Fields: `timestamp`, `parameter_name`, `predicted_value`, `actual_value`,
`shap_top_features`, `is_anomaly`, `anomaly_score`, `retrain_alert`.

**Notes:**
- `export-archive` does not require `config/system_config.json` — it
  reads only from the archive directory, so it can be run on any machine
  that has a copy of `archive/`.
- The `--archive-dir` flag overrides the default `archive/` path if the
  Cold archive lives elsewhere (e.g. a network share).
- The output file is written atomically (`.tmp` → rename), so a
  partial/interrupted export never leaves a corrupt file.

**Load the exported file into the SQL Server Warm tier:**

```bash
python warm_tier_etl.py --init-schema
python warm_tier_etl.py warm_tier/EC_last_12m.jsonl \
    --site-id C-1 --parameter EC --unit "µS/cm"
```

This second step is manual and idempotent. It validates each JSONL row and
uses `(site, parameter, timestamp)` as the deterministic SQL identity. See
[`dashboard/WARM_TIER.md`](dashboard/WARM_TIER.md) for database provisioning,
configuration, security, query behavior, and tests. No scheduled or automatic
Cold-to-Warm transfer is currently implemented.
