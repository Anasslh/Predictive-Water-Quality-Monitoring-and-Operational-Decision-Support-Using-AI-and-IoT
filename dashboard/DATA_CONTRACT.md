# Data Contract — Dashboard ⇄ Pipeline

The dashboard consumes **only** the two files per parameter written by
`src/monitor/export.py`. It never imports `src/`, model pickles, or training
code. This document records the contract **as confirmed from the source code and
the real export files**, not from summaries.

Source of truth:
- `src/monitor/export.py` → `_result_to_record()`, `write_status_export()`
- `src/monitor/__init__.py` → `MonitorResult`
- `config/system_config.json` → `exports` (retention), `validation.physical_bounds`
- Real files: `exports/EC.jsonl`, `exports/EC_status.json`

---

## 1. Filename patterns

| File | Pattern | Produced by |
|------|---------|-------------|
| Measurements (rolling) | `exports/<parameter>.jsonl` | `append_measurement()` |
| Status snapshot | `exports/<parameter>_status.json` | `write_status_export()` |
| Temp (atomic write) | `*.jsonl.tmp`, `*.json.tmp` | ignored by the dashboard |

`<parameter>` is the free-form `parameter_name` from `sensors_config.json`
(e.g. `EC`, `pH`, `Turbidity`). **Parameters are discovered dynamically**: any
top-level `*.jsonl` (excluding `*_status.json`) is a parameter. Sub-directories
(e.g. `_canary_disabled/`) and `.tmp` files are ignored.

Both files are written **atomically** (`.tmp` then `os.replace`), so a reader
always sees a complete file — never a partial write.

---

## 2. `<parameter>.jsonl` — one JSON object per line

Field order and types exactly as emitted by `_result_to_record()`:

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `timestamp` | string (ISO-8601) | no* | May be tz-aware or naive; **naive is assumed UTC** (matches `export._parse_ts`). *A line with an unparseable timestamp is retained but cannot be placed on the time axis. |
| `parameter_name` | string | no | Echoes the parameter. |
| `predicted_value` | number | **yes** | Model one-step prediction. `null` if the prediction stage failed. |
| `actual_value` | number | **yes** | Reconstructed as `prediction + residual`; `null` when no measured value was supplied for the row (prediction-only). |
| `shap_top_features` | array | no | May be `[]`. Each item: `{ "feature": string, "shap_value": number, "direction": "positive"|"negative" }`. |
| `is_anomaly` | boolean | **yes** | `null` when no detector was fitted or no `actual_value` was available to score. |
| `anomaly_score` | number | **yes** | Normalised `[0, 1]`. `null` when not scored. |
| `retrain_alert` | string | **yes** | Human-readable alert; `null` on most cycles. |

**Not present in the contract:** there is **no `forecast` field** and **no `R²`**.
`MonitorResult.forecast` is computed but dropped by `_result_to_record()`. The
dashboard treats `forecast` as an **optional** record field (code-ready) and
adds Forecast navigation only if a future export includes valid predictions.

Retention: the pipeline purges lines older than
`system_config.json → exports.retention_days` (default **30**) on every write,
based on the `timestamp` field. The dashboard re-applies the same window on read.

### Real example (`exports/EC.jsonl`)
```json
{"timestamp": "2026-07-23T11:41:22.285161+00:00", "parameter_name": "EC",
 "predicted_value": 186.17, "actual_value": 769.0,
 "shap_top_features": [{"feature": "EC_roll3_mean", "shap_value": -5.59, "direction": "negative"}],
 "is_anomaly": true, "anomaly_score": 1.0, "retrain_alert": null}
```

---

## 3. `<parameter>_status.json`

Exactly as assembled by `write_status_export()`:

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `parameter_name` | string | no | |
| `unit` | string | no | May be `""` if `sensors_config.json` lacked the unit. |
| `model_version` | string | **yes** | `null` when the status writer had no `models_store_path` (observed `null` in the real EC file even though a model is frozen). |
| `last_promoted_at` | string (ISO) | **yes** | |
| `last_measurement_at` | string (ISO) | **yes** | |
| `performance_30d` | object | no | See below. |
| `pending_approvals` | integer | no | |
| `consecutive_rejections` | integer | no | |
| `queried_at` | string (ISO) | no | |

`performance_30d`:

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `skill_vs_persistence_pct` | number | **yes** | % RMSE reduction vs naive persistence. `null` if `< 10` measurements. |
| `rmse` | number | **yes** | Model RMSE over the window. |
| `mae` | number | **yes** | |
| `n_measurements` | integer | no | Records with both predicted & actual. |
| `insufficient_data` | boolean | optional | `true` (with null metrics) when `< 10` measurements. |

There is **no `r2` and no `drift_status`** field. The dashboard omits unavailable
optional fields and never fabricates them.

### Real example (`exports/EC_status.json`)
```json
{"parameter_name": "EC", "unit": "µS/cm", "model_version": null,
 "last_promoted_at": null, "last_measurement_at": "2026-07-23T16:23:16.190576+00:00",
 "performance_30d": {"skill_vs_persistence_pct": 26.72, "rmse": 425.638, "mae": 311.7738, "n_measurements": 16},
 "pending_approvals": 0, "consecutive_rejections": 0, "queried_at": "2026-07-23T16:23:16.245684+00:00"}
```

---

## 4. WQI (not in the monitoring export contract)

WQI is **not** exported. It exists only as static per-row columns in the
processed training artifact `data/processed/c1_with_wqi.csv`
(`WQI_new`, `Class_new`, `Qi_pH`, `Qi_EC`, `Qi_Turbidity`, `W_*`), produced by
`src/data/compute_wqi.py` (WAWQI, Brown et al. 1972 + BIS IS 10500:2012 drinking
-water standards). The dashboard reads it **read-only** for the methodology page
only. Absent CSV → the WQI block is omitted (no placeholder).

---

## 5. Methodology-only reference values (documented, never operational)

| Parameter | Reference (Si) | Source |
|-----------|----------------|--------|
| pH | 8.5 | BIS IS 10500:2012 / WHO 2022 (aesthetic upper) |
| EC | 1500 µS/cm | BIS IS 10500:2012 (permissible) |
| Turbidity | 5 NTU | BIS IS 10500:2012 (permissible) |

Physical plausibility bounds (`system_config.json`, sensor-validation ranges —
**not** quality thresholds): pH 0–14, EC 0–5000 µS/cm, Turbidity 0–10000 NTU.
These values and the WAWQI standards appear only on Data & Methodology. They are
not drawn on operational monitoring charts and are not alert thresholds.

---

## 6. Error behaviour (guaranteed by the loader)

| Condition | Dashboard behaviour |
|-----------|--------------------|
| Missing `.jsonl` | parameter has no records (not discovered) |
| Missing `_status.json` | `status = None`; health tiles show `—` |
| Empty file | empty result, no error |
| Invalid JSON line | line skipped, counted in `data quality → skipped lines` |
| Non-dict JSON line | skipped, counted |
| `null` / missing optional field | coerced to `None`; optional UI row/tile omitted |
| Invalid timestamp | record kept, omitted from the time axis |
| Exact duplicate lines | duplicates removed |
| Conflicting rows with one timestamp | last complete source row kept; fields never merged |
| Records out of order | stable chronological sort applied |
| Missing `actual_value` | remains null; never replaced by prediction; measured chart gap preserved |
| NaN / Inf number | treated as `None` |
| One parameter's file corrupt | that parameter degrades; others unaffected |

## 7. Current EC export quality note

The inspected `exports/EC.jsonl` contains 103 valid, chronologically ordered rows
with 103 unique timestamps: there are no duplicate timestamps. It contains 34
measured rows and 69 prediction-only rows. Several research/replay runs record
alternating measured values (`188.1742` and `769.0`) only fractions of a second
apart; those genuine exported rows explain the closely spaced zigzag pattern.
The dashboard preserves the records and the null-measurement gaps rather than
smoothing, substituting, or hiding them.

The default `historical` source mode labels this as a historical dataset and
disables operational freshness alerts. A future continuous deployment can set
`WQD_DATA_SOURCE_MODE=continuous` after its cadence/SLA is approved.

## 8. Cold export → Warm SQL → dashboard mapping

The Hot contract above is unchanged. `run.py export-archive` copies the same
measurement objects from the permanent Cold archive into a bounded flat JSONL
file; `warm_tier_etl.py` validates and upserts those records into SQL Server.
See [`WARM_TIER.md`](WARM_TIER.md) for setup and operational behavior.

| Hot / Cold JSON field | Archive representation | Warm SQL column | Dashboard field | Handling |
|---|---|---|---|---|
| `timestamp` | ISO-8601 string | `MeasurementTimestamp datetimeoffset(7)` | `MeasurementRecord.timestamp` | Normalized to UTC; invalid rows rejected |
| `parameter_name` | string | `ParameterName nvarchar(128)` | record field / map key | Generic; required |
| `predicted_value` | number/null | `PredictedValue float NULL` | `predicted_value` | Numeric/null mapping |
| `actual_value` | number/null | `ActualValue float NULL` | `actual_value` | Null preserved; never filled from prediction |
| `shap_top_features` | JSON list | `ShapTopFeatures nvarchar(max)` | `shap_top_features` | Canonical JSON; `ISJSON` constraint |
| `is_anomaly` | boolean/null | `IsAnomaly bit NULL` | `is_anomaly` | Three-state meaning preserved |
| `anomaly_score` | number/null | `AnomalyScore float NULL` | `anomaly_score` | Null preserved |
| `retrain_alert` | string/null | `RetrainAlert nvarchar(2048) NULL` | `retrain_alert` | Empty text normalized to null |
| `forecast` | optional object | `ForecastJson nvarchar(max) NULL` | `forecast` | Schema-ready; current exports do not provide it |
| not in record export | CLI/record metadata | `Unit nvarchar(64) NULL` | `ParameterData.unit` | Supplied during ingestion until exports carry units |
| not in export | CLI configuration | `SiteId nvarchar(128)` | query scope | Added metadata |
| not in export | filename | `SourceFile nvarchar(512)` | not displayed | Added lineage metadata |
| normalized whole row | SHA-256 | `SourceRecordHash binary(32)` | not displayed | Idempotency/change detection |
| not in export | SQL UTC default | `IngestedAt datetimeoffset(7)` | not displayed | Added ingestion metadata |

`<parameter>_status.json` is not archived in SQL Server. Model performance,
current model version, pending approvals, and rejection counts are unavailable
in Historical archive mode; the dashboard does not mix current Hot status into
historical views.
