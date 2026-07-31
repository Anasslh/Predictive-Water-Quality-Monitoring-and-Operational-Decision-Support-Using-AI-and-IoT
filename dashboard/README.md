# Water Quality Monitoring — Operational Decision-Support Dashboard

A read-only Streamlit dashboard for the *Predictive Water Quality Monitoring*
project. It consumes the ML pipeline's exported files and helps an operator
answer, at a glance:

1. What is the current water-quality state?
2. What changed recently?
3. Is any reading abnormal?
4. What does the model predict next? *(the view appears only when forecasts are exported)*
5. Why did the model produce this prediction? *(SHAP, in plain language)*
6. Which model-monitoring metrics are available?
7. Is retraining review required?

It is **decoupled** from the ML code: it reads only `exports/<param>.jsonl` and
`exports/<param>_status.json` and never imports `src/` or model pickles. It can
run on a separate machine that only receives a synced `exports/` folder.

> **Scope note.** Under the default *generalist* profile the dashboard makes **no
> drinking-water safety claim**. Drinking-water WQI references are confined to
> the methodology page and are not operational chart limits. See `ASSUMPTIONS.md`.

---

## Purpose & audience

Water-utility operators and monitoring engineers. The interface is a restrained
operations console (dense, legible, no decorative UI), bilingual **English /
Arabic** (RTL supported).

---

## Installation

Requires Python 3.11+ (developed on 3.11). From the **repository root**:

```bash
python -m venv .venv
# Windows:
.venv/Scripts/python -m pip install -r dashboard/requirements.txt
# macOS / Linux:
.venv/bin/python  -m pip install -r dashboard/requirements.txt
```

The dashboard dependencies (Streamlit, Plotly, Pandas) are intentionally
separate from the ML pipeline's `requirements.txt` — no scikit-learn / xgboost /
shap needed to run the dashboard.

## Running locally

```bash
streamlit run dashboard/app.py
```

or use the helper scripts (they auto-select the project venv):

```bash
./dashboard/run_dashboard.sh        # bash / WSL / macOS / Linux
```
```powershell
./dashboard/run_dashboard.ps1       # Windows PowerShell
```

Then open <http://localhost:8501>.

## Configuration (environment variables)

All optional; sensible defaults resolve relative to the repository root.

| Variable | Default | Purpose |
|----------|---------|---------|
| `WQD_EXPORTS_DIR` | `<repo>/exports` | Where `<param>.jsonl` / `<param>_status.json` live. Point at a synced folder on a separate host. |
| `WQD_PROCESSED_DIR` | `<repo>/data/processed` | Location of `c1_with_wqi.csv` (methodology page). |
| `WQD_SITE_NAME` | `C-1 — Ramgarh Station` | Header site label. |
| `WQD_DATA_SOURCE_MODE` | `historical` | `historical` disables freshness-alert semantics; `continuous` enables the documented cadence heuristic. |
| `WQD_DATA_SOURCE_NOTE` | `Historical research data` (translated fallback) | Optional deployment-specific source qualifier. |
| `WQD_WATER_USE` | `generalist` | `generalist` \| `drinking` \| `irrigation` (affects reference-line labelling only). |
| `WQD_DEFAULT_LANG` | `en` | `en` \| `ar`. |
| `WQD_RETENTION_DAYS` | `30` | Rolling window (mirror the pipeline). |
| `WQD_FRESH_MULTIPLIER` | `3.0` | Freshness heuristic (see `ASSUMPTIONS.md`). |

Example:
```bash
WQD_EXPORTS_DIR=/synced/exports WQD_DEFAULT_LANG=ar streamlit run dashboard/app.py
```

## Expected files

Per monitored parameter, in `WQD_EXPORTS_DIR`:
- `<param>.jsonl` — rolling measurement log
- `<param>_status.json` — model/status snapshot

Both are produced by `python run.py monitor …`. See `DATA_CONTRACT.md` for the
exact confirmed schema. With no exports present, the dashboard shows a calm
"no exported parameters" state rather than failing.

---

## Views

| View | Shows |
|------|-------|
| **Overview** | Latest measured state, trend, affected-parameter anomaly count, source status, and neutral model-monitoring availability. |
| **Parameter detail** | Measured vs predicted history, preserved gaps, anomaly markers, date filter, data quality, SHAP explanation. |
| **Forecast** | Multi-step forecast **only when valid forecast values are exported**; otherwise the navigation item is absent. |
| **Anomalies & alerts** | All flagged anomalies across parameters, chronological. |
| **Model monitoring** | Available RMSE, MAE, skill vs persistence, counts and optional version/update fields; no inferred health class. |
| **Data & methodology** | Data source, units, limitations, and the historical WQI diagnostic with full caveats. |

---

## Project layout

```
dashboard/
├── app.py               # entry point: header, nav, language, routing
├── config/              # settings (env), theme tokens, documented reference values
├── i18n/                # EN/AR strings + translator (RTL aware)
├── models/              # typed, tolerant schemas mirroring the export contract
├── services/            # discovery, loaders (only disk I/O), validation, transforms, wqi
├── components/          # formatting, Plotly chart builders, layout, SHAP panel
├── views/               # the six pages
├── assets/theme.css     # design system stylesheet
├── tests/               # unit + Streamlit AppTest coverage
└── *.md                 # README, DATA_CONTRACT, ASSUMPTIONS, IMPLEMENTATION_REPORT
```

## Tests

From the repository root:
```bash
.venv/Scripts/python -m pytest dashboard/tests -q      # Windows
.venv/bin/python      -m pytest dashboard/tests -q      # POSIX
```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| "No exported parameters found" | `WQD_EXPORTS_DIR` has no `*.jsonl`. Run the pipeline or set the variable. |
| A parameter is missing | It has no `<param>.jsonl` yet (e.g. pH/Turbidity before their first `monitor` run). |
| Model version / optional metric is absent | Null or unavailable optional fields are intentionally omitted instead of rendered as `—`. |
| Forecast is absent from navigation | No valid forecast values exist in the current export contract. The existing view activates automatically when they do. |
| `ModuleNotFoundError: dashboard` | Run from the **repository root**, or use the helper scripts. |
| Arabic layout looks LTR | Toggle the language in the sidebar, or set `WQD_DEFAULT_LANG=ar`. |

## Record and chart policies

- Records are sorted chronologically. Exact duplicate rows are removed; when
  conflicting rows share one parseable timestamp, the last complete source row
  wins. Fields are never merged between rows.
- A missing measured value remains missing. It is never filled from the model
  prediction, and the measured line uses `connectgaps=False` so gaps remain visible.
- “Parameters with active anomalies” counts affected parameters whose latest
  evaluated record is anomalous; the Anomalies view remains record-level.
- The current EC export has unique timestamps, but repeated research/replay runs
  create sharply alternating measured values within fractions of a second. This
  upstream data-quality characteristic is disclosed in Methodology, not hidden.
