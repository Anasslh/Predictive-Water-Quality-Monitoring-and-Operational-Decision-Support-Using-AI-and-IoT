# sensors_config.json — Schema Reference

Used by `src/pipeline/orchestrator.py` to onboard any new sensor parameter
without writing parameter-specific code.

## Top-level fields

| Field | Type | Description |
|---|---|---|
| `_schema_version` | string | Config schema version for migration tracking |
| `timestamp_column_candidates` | string[] | Ordered list of column names to try as the timestamp; first parseable, monotone-increasing column wins |
| `sensors` | object[] | One entry per sensor parameter |

## Per-sensor fields

| Field | Type | Required | Description |
|---|---|---|---|
| `column_name` | string | ✓ | Exact column name in the raw CSV |
| `parameter_name` | string | ✓ | Human label used in model versioning filenames and reports |
| `unit` | string | ✓ | Physical unit displayed in reports and figures |
| `wqi_standard` | number | ✓ | WHO/BIS reference value for WQI computation (Si standard) |
| `co_variables` | string[] | — | Other sensor columns to include as lag-1 co-features. Omit to skip cross-variable features. |
| `lag_hours` | number[] | — | Lag depths in hours. Default [24, 48, 72] (3 lags at daily resolution). |
| `rolling_hours` | number[] | — | Rolling window durations in hours. Default [72, 168] (3-day and 7-day at daily resolution). |
| `_note` | string | — | Human-only comment; ignored by code. |

## Adding a new sensor

1. Append a new object to `"sensors"`.
2. Run `python src/pipeline/orchestrator.py` (or call `onboard_new_parameter()`
   from code) to trigger the benchmark and human approval flow.
3. No other file needs to be modified.

## Hour-to-row conversion

`lag_hours` and `rolling_hours` are converted to row counts at runtime:

    n_rows = floor(hours / measurement_frequency_hours)

Examples:
- Daily data (24 h): lag_hours=[24,48,72] → lag steps 1, 2, 3
- Hourly data (1 h): lag_hours=[24,48,72] → lag steps 24, 48, 72
