# Implementation Report — Dashboard MVP

**Branch:** `feature/dashboard-mvp`
**Last reviewed:** 2026-07-31
**Scope:** Operational decision-support dashboard for the Predictive Water
Quality Monitoring project (Industrial Engineering track).

---

## 1. What was implemented

A read-only, bilingual (English / Arabic, RTL-aware) Streamlit dashboard that
consumes the ML pipeline's export files and currently presents five views:
Overview, Parameter detail, Anomalies & alerts, Model monitoring, and Data &
methodology. Forecast remains implemented but is inserted into navigation only
when a valid exported forecast exists. It is fully decoupled from the ML code — it imports **no** module
from `src/` and reads **no** model pickle. New parameters are discovered
dynamically from the exports directory.

Key properties:
- **Design system**: centralised tokens (`config/theme.py` + `assets/theme.css`),
  restrained operations-console look, consistent chart grammar (measured /
  predicted / forecast / anomaly / gaps).
- **Robust by construction**: missing files, empty files, corrupt JSON lines,
  null/optional fields, bad timestamps, exact/conflicting duplicates and out-of-order records are
  all handled without crashing; one bad parameter never takes down the app.
- **Scientifically conservative**: no invented thresholds; generalist framing
  with no drinking-water verdict; optional null fields are omitted; no model
  health class is inferred; WQI is confined to Methodology with full caveats.

## 2. Architecture

```
app.py ── discovery ─┐
                     ├─ services (loaders → validation → transforms)  ← only disk I/O
config/ i18n/ ───────┤        │
models/ (schemas) ───┘        ▼
components/ (charts, layout, shap, format)  ← pure presentation
views/ (5 visible + 1 data-gated)  ← compose components from an immutable AppContext
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
| Forecast navigation data-gated | `MonitorResult.forecast` is not serialised by the current exporter. The view code remains ready and navigation activates automatically when valid forecast predictions appear. |
| WQI on methodology page only | It uses drinking-water standards with heavy documented limitations and is not in the monitoring export contract; surfaced as a historical diagnostic, not a verdict. |
| Historical source mode | Current research/replay exports show “Historical dataset” and do not trigger a live-sensor freshness warning. A configured continuous mode retains the cadence heuristic for future approved use. |
| Neutral model monitoring | Status availability and raw metrics are separate from anomaly state and source freshness. No “Healthy” badge exists without an approved classification rule. |
| Whole-record duplicate policy | Exact rows are removed; conflicting same-timestamp rows keep the last source record. Fields are never merged, so missing measurements are never filled from predictions. |
| Methodology-only references | Drinking-water WQI standards remain documented, but operational charts contain no WQI-derived reference or alert line. |

## 6. Tests run and results

Command (from repo root):
```
.venv/Scripts/python -m pytest dashboard/tests -q
```

**Latest automated result: 63 passed in 5.55 s.** Coverage includes discovery,
valid/malformed/empty/missing exports, null status optionals, duplicate and
out-of-order records, single and consecutive measured-value gaps, no
measured/predicted substitution, chart naming/tooltips/dashes, anomaly marker
positions, affected-parameter anomaly counts, source-aware freshness, Forecast
navigation gating, WQI confinement, and all visible EN/AR views.

The final live-browser review is performed against the repository's real EC
exports using the documented local command. Runtime URL and final verification
results are reported in the task handoff rather than asserted here permanently.

## 7. Known limitations

- **Only EC has monitoring exports today.** pH and Turbidity have frozen models but no
  `.jsonl` yet; they will appear automatically once their `monitor` runs.
- **No forecast data** in the current export contract → Forecast navigation is hidden.
- **No R² / drift status** exported; unavailable optional fields are omitted.
- The EC file contains repeated historical/replay runs in tight succession,
  producing sharp measured-value alternation despite unique timestamps.
- Single site, single year of data (C-1). WQI is a drinking-water historical
  diagnostic only and is not a water-safety verdict.
- No authentication (local MVP).

## 8. Remaining work (priority order)

1. Define and implement an approved additive forecast export contract in the ML
   pipeline as a separately reviewed task; the dashboard will then enable its view.
2. Generate pH / Turbidity exports (run `monitor` for them) to see multi-parameter
   overview with real data.
3. Optional acknowledgment layer (if the read-only decision changes).
4. Deployment target + auth once chosen.
5. Approve continuous-source freshness SLA and model-health classification rules
   before enabling live alert/status semantics.
