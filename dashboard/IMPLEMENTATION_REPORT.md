# Implementation Report — Dashboard MVP

**Branch:** `feature/dashboard-mvp`
**Date:** 2026-07-24
**Scope:** Operational decision-support dashboard for the Predictive Water
Quality Monitoring project (Industrial Engineering track).

---

## 1. What was implemented

A read-only, bilingual (English / Arabic, RTL-aware) Streamlit dashboard that
consumes the ML pipeline's export files and presents six views: Overview,
Parameter detail, Forecast, Anomalies & alerts, Model health, and Data &
methodology. It is fully decoupled from the ML code — it imports **no** module
from `src/` and reads **no** model pickle. New parameters are discovered
dynamically from the exports directory.

Key properties:
- **Design system**: centralised tokens (`config/theme.py` + `assets/theme.css`),
  restrained operations-console look, consistent chart grammar (actual / predicted /
  forecast / anomaly / reference / gaps).
- **Robust by construction**: missing files, empty files, corrupt JSON lines,
  null/optional fields, bad timestamps, duplicates and out-of-order records are
  all handled without crashing; one bad parameter never takes down the app.
- **Scientifically conservative**: no invented thresholds; generalist framing
  with no drinking-water verdict; forecast/R²/drift shown only if the data
  exists; WQI confined to the methodology page with full caveats.

## 2. Architecture

```
app.py ── discovery ─┐
                     ├─ services (loaders → validation → transforms)  ← only disk I/O
config/ i18n/ ───────┤        │
models/ (schemas) ───┘        ▼
components/ (charts, layout, shap, format)  ← pure presentation
views/ (6 pages)  ← compose components from an immutable AppContext
```

Strict separation: **loaders** (I/O) → **validation** (cleaning) → **schemas**
(typed data) → **transforms** (derived metrics) → **views** (render). Charts and
transforms import no Streamlit, so they are unit-testable headless.

## 3. Files created

All new; nothing under `src/` was modified.

```
dashboard/
  app.py
  requirements.txt   run_dashboard.sh   run_dashboard.ps1
  .streamlit/config.toml
  assets/theme.css
  config/    settings.py  theme.py  reference_limits.py
  i18n/      strings.py   translator.py
  models/    schemas.py
  services/  discovery.py loaders.py validation.py transforms.py wqi.py
  components/ format.py charts.py layout.py shap_panel.py
  views/     overview.py parameter_detail.py forecast.py anomalies.py
             model_health.py methodology.py
  tests/     conftest.py test_schemas.py test_validation.py test_discovery.py
             test_loaders.py test_transforms.py test_charts.py test_smoke.py
  README.md  DATA_CONTRACT.md  ASSUMPTIONS.md  IMPLEMENTATION_REPORT.md
  (+ __init__.py in each package)
```

## 4. Files modified

- `.gitignore` — added `.pytest_cache/` and `**/secrets.toml`.

No ML pipeline file was changed. (`.venv/` was already ignored.)

## 5. Design decisions

| Decision | Rationale |
|----------|-----------|
| Streamlit + Plotly + Pandas | Matches the concept note's named tools and the repo's Python/open-source convention; no second frontend framework. |
| Read exports only, never `src/` | Keeps the dashboard deployable on a separate machine; ML pipeline stays untouched and stable. |
| Tokenised design + native theme in `config.toml` | Interface stays intentional even if the injected CSS is disabled/unsupported. |
| Forecast view wired but empty | `MonitorResult.forecast` is not serialised by the exporter; showing nothing beats fabricating. Code activates automatically if a `forecast` field appears. |
| WQI on methodology page only | It uses drinking-water standards with heavy documented limitations and is not in the live contract; surfaced as a historical diagnostic, not a verdict. |
| `water_use_profile` gate | Lets a future approved deployment relabel documented reference values without inventing new ones. |
| Freshness heuristic from data cadence | No SLA exists in the repo; a self-calibrating, clearly-documented UI heuristic avoids inventing a threshold. |

## 6. Tests run and results

Command (from repo root):
```
.venv/Scripts/python -m pytest dashboard/tests -q
```

**Result: 51 passed** (2.4 s). Breakdown:

| File | Tests | Covers |
|------|-------|--------|
| `test_schemas.py` | 11 | float/bool/timestamp coercion, SHAP parse, record/status/forecast parsing, bad ints, insufficient-data |
| `test_validation.py` | 4 | dedupe, retention (keeps undated), sort (undated last), full pipeline |
| `test_discovery.py` | 5 | discovery, ignores status/tmp/subdirs, missing/empty dir, status-only |
| `test_loaders.py` | 9 | valid/corrupt/missing/empty JSONL, status valid/missing/malformed, integration, missing-status |
| `test_transforms.py` | 10 | trend, freshness (fresh/stale/unknown), anomaly list/state, health tokens, counts |
| `test_charts.py` | 6 | actual-vs-predicted (EN/AR/empty), forecast None vs present, WQI bar |
| `test_smoke.py` | 6 | AppTest end-to-end: real exports, empty dir, missing dir, all 6 views on messy data, Arabic RTL, forecast empty state |

**Live run**: `streamlit run dashboard/app.py` served HTTP 200 and rendered the
real `exports/EC.jsonl` — anomaly (769 µS/cm), trend +580.83, prediction, 30-day
skill, freshness and health all sourced from the live files. A headless
`AppTest` pass confirmed all six views render without exception against the real
exports in both English and Arabic.

> Screenshots could not be captured in this environment (the browser pane was
> not compositing frames); verification was done via the live HTTP server,
> page-text extraction, and the AppTest harness.

## 7. Known limitations

- **Only EC has live exports today.** pH and Turbidity have frozen models but no
  `.jsonl` yet; they will appear automatically once their `monitor` runs.
- **No forecast data** in the current export contract → Forecast view shows an
  empty state.
- **No R² / drift status** exported → shown as `—`.
- Single site, single year of data (C-1). WQI is a drinking-water historical
  diagnostic only.
- No authentication (local MVP).

## 8. Remaining work (priority order)

1. **(Optional, small) Persist forecasts** — add `result.forecast` to
   `_result_to_record()` in `src/monitor/export.py` (additive, ~4 lines) so the
   Forecast view populates. Documented but not done to avoid touching the pipeline.
2. Generate pH / Turbidity exports (run `monitor` for them) to see multi-parameter
   overview with real data.
3. Optional acknowledgment layer (if the read-only decision changes).
4. Deployment target + auth once chosen.
5. Accuracy-by-horizon panel once forecast history is exported.
