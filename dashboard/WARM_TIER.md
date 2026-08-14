# SQL Server Warm Tier

## Purpose and status

The optional Warm Tier provides bounded historical queries for the read-only Streamlit dashboard. It complements the existing stores:

| Tier | Storage | Purpose | Retention |
|---|---|---|---|
| Current / Hot | `exports/<parameter>.jsonl` and status JSON | Current dashboard monitoring | Rolling 30 days by default |
| Historical / Warm | SQL Server `dbo.Fact_WaterQuality` | Operator-selected historical views | Operationally managed; dashboard queries are bounded |
| Permanent / Cold | `archive/<parameter>_<YYYY-MM>.jsonl[.gz]` | Authoritative processed-record archive | No automatic purge |

With no `DB_CONN_STR`, Current monitoring remains fully usable and the dashboard states that Historical archive is not configured.

## Verified data flow

```text
ParameterMonitor.process_new_measurement()
        ├── Hot rolling export: exports/<parameter>.jsonl
        └── Cold monthly archive: archive/<parameter>_<YYYY-MM>.jsonl[.gz]
                                      ↓ manual CLI export
python run.py export-archive → flat, bounded JSONL
                                      ↓ manual CLI ingestion
python warm_tier_etl.py → SQL Server dbo.Fact_WaterQuality
                                      ↓ bounded read-only query
dashboard/services/db_service.py → typed ParameterData
                                      ↓ global source selection
all dashboard views
```

The Cold write is attempted after each successful monitoring call and is non-fatal to monitoring if it fails. Cold export and SQL ingestion are manual. There is no scheduler, service, queue, automatic retry worker, or file watcher.

## SQL Server assessment

SQL Server is a **reasonable choice that requires deployment documentation**:

- Structured site/parameter/time/anomaly data fits relational constraints and queries.
- Transactions, filtered indexes, JSON validation, and UTC-aware timestamps meet the MVP needs.
- It fits Microsoft environments and a possible future Power BI integration.
- SQL Server Express can support local academic/MVP work within its limits; Developer Edition is non-production only. Production licensing and hosting must be selected explicitly.
- ODBC drivers, server provisioning, certificates, backups, and operations add reproducibility complexity compared with file-only development.

SQL Server is preserved; this project has no technical reason to migrate databases during the Warm MVP.

## Schema

Versioned scripts live in [`database/warm_tier/`](../database/warm_tier/):

- `000_create_database.sql` provisions a named database through `sqlcmd`.
- `001_initial_schema.sql` creates the version table, fact table, constraints, and indexes idempotently.

`dbo.Fact_WaterQuality` is parameter-generic and has no EC/pH/Turbidity-specific columns.

| Column | SQL Server type | Null | Purpose |
|---|---|---:|---|
| `MeasurementId` | `bigint IDENTITY` | no | Surrogate primary key and query tie-breaker |
| `SiteId` | `nvarchar(128)` | no | Configured site identifier |
| `ParameterName` | `nvarchar(128)` | no | Generic parameter identifier |
| `Unit` | `nvarchar(64)` | yes | Data-driven unit, optionally supplied at ingestion |
| `MeasurementTimestamp` | `datetimeoffset(7)` | no | Original timestamp normalized to UTC |
| `PredictedValue` / `ActualValue` | `float` | yes | Predicted and measured values |
| `ShapTopFeatures` | `nvarchar(max)` | yes | JSON SHAP list guarded by `ISJSON` |
| `IsAnomaly` | `bit` | yes | True/false/null anomaly state |
| `AnomalyScore` | `float` | yes | Optional anomaly score |
| `RetrainAlert` | `nvarchar(2048)` | yes | Optional exported alert text |
| `ForecastJson` | `nvarchar(max)` | yes | Optional future forecast block guarded by `ISJSON` |
| `ModelVersion` | `nvarchar(256)` | yes | Optional record metadata; absent today |
| `SourceFile` | `nvarchar(512)` | no | Ingested archive filename |
| `SourceRecordHash` | `binary(32)` | no | SHA-256 of the normalized whole record |
| `IngestedAt` | `datetimeoffset(7)` | no | SQL ingestion/update time |

The unique key is `(SiteId, ParameterName, MeasurementTimestamp)`. Indexes support parameter/time, site/parameter/time, recent ranges, and anomaly-only queries. Model-performance history is not stored, so no performance-history index or Warm status UI is claimed.

## Configuration and security

[`.env.example`](../.env.example) contains placeholders only. The application reads process environment variables and does not load or commit `.env` automatically.

Required: `DB_CONN_STR`.

Optional:

```text
WQD_WARM_SITE_ID=C-1
WQD_WARM_LOOKBACK_DAYS=365
WQD_DB_TIMEOUT_SECONDS=5
```

- Prefer Windows integrated authentication where appropriate, or inject SQL credentials through a secret manager.
- Local development may use `Encrypt=yes;TrustServerCertificate=yes`. Production should use a trusted certificate without bypassing validation.
- Connection strings are excluded from settings representations and Streamlit cache hashing.
- Raw driver messages are never displayed to operators or written by the Warm service logger.

## Reproducible setup

Install SQL Server, a compatible Microsoft ODBC Driver, and the dashboard requirements:

```bash
python -m pip install -r dashboard/requirements.txt
```

Provision a database with Windows integrated authentication:

```powershell
sqlcmd -S "localhost\SQLEXPRESS" -E -C `
  -v "DatabaseName=NileWaterQuality_Warm" `
  -i database\warm_tier\000_create_database.sql
```

Set the connection string and initialize the schema:

```powershell
$env:DB_CONN_STR = "Driver={ODBC Driver 17 for SQL Server};Server=localhost\SQLEXPRESS;Database=NileWaterQuality_Warm;Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;"
python warm_tier_etl.py --init-schema
```

Export a Cold range and ingest it:

```powershell
python run.py export-archive --parameter EC --months 12 --output warm_tier/EC_last_12m.jsonl
python warm_tier_etl.py warm_tier/EC_last_12m.jsonl --site-id C-1 --parameter EC --unit "µS/cm"
```

Run the dashboard:

```powershell
python -m streamlit run dashboard/app.py
```

## Ingestion semantics

- Input is UTF-8 JSONL from `run.py export-archive`.
- Empty lines are ignored; an empty file is a successful no-op.
- Corrupt JSON, invalid timestamps/numbers/booleans, parameter mismatches, and invalid SHAP/forecast shapes are rejected per line and counted.
- Valid timestamps are normalized to UTC.
- Within one file, the last whole valid row for a duplicate site/parameter/timestamp wins; fields are never merged.
- SQL uses the same key under serializable isolation. Identical reingestion is unchanged; a changed record replaces the whole payload; a new key is inserted.
- Any database failure rolls back the full batch, so retrying the same file is safe.
- Source filename, record hash, and ingestion timestamp provide lineage.
- Scheduling and processed-file manifests remain future deployment work.

## Dashboard semantics

- The operator chooses `Current monitoring` or `Historical archive`; internal Hot/Warm terms are not required.
- Selection is global across Overview, Parameter Detail, Alerts, Model Monitoring, Methodology, and data-gated Forecast.
- Warm queries are scoped by site and explicit bounded date range, ordered chronologically, and cached for five minutes.
- Warm-only parameters and stored units are discovered from rows; SQL is parameter-generic.
- Missing actual values remain gaps and are never replaced by predictions.
- Anomaly markers remain tied to measured values.
- Current model metrics and pending approvals are not historical. Warm mode shows them unavailable instead of borrowing Hot status.
- The bounded result is held in memory. This is acceptable for the current one-year volume but is not pagination for large deployments.

## Failure behavior

Missing configuration removes the unavailable archive option. Configured failures are mapped to localized safe messages for missing dependencies, authentication, unavailable server/database, missing schema, timeout, or other query failure. The app remains running and the operator can return to Current monitoring.

## Testing

Unit and Streamlit tests do not need SQL Server:

```bash
python -m pytest dashboard/tests -q
```

The live test is explicitly gated and must target an isolated database:

```powershell
$env:WQD_TEST_DB_CONN_STR = "<isolated test database connection string>"
python -m pytest dashboard/tests/integration/test_warm_tier_sqlserver.py -q
```

It initializes the schema, ingests generic/null/anomaly records, verifies duplicate reingestion, performs a bounded query, and removes only rows for its unique test site.

## Remaining limitations

- No scheduled ingestion or automatic transfer.
- No historical model-status/performance table.
- Units must be supplied at ingestion until archive records include them.
- One configured site per deployment; no operator site picker.
- No result pagination.
- Backup, retention/deletion policy, monitoring, least-privilege roles, certificate deployment, and production licensing remain deployment responsibilities.
