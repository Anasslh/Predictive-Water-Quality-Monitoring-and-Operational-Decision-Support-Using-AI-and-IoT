# Water Quality Monitoring — Operational Decision-Support Dashboard

A read-only Streamlit dashboard for the *Predictive Water Quality Monitoring*
project. It consumes the ML pipeline's exported files and helps an operator
answer, at a glance:

1. What is the current water-quality state?
2. What changed recently?
3. Is any reading abnormal?
4. What does the model predict next? *(when forecasts are exported)*
5. Why did the model produce this prediction? *(SHAP, in plain language)*
6. Is the model currently healthy?
7. Is retraining review required?

It is **decoupled** from the ML code: it reads only `exports/<param>.jsonl` and
`exports/<param>_status.json` and never imports `src/` or model pickles. It can
run on a separate machine that only receives a synced `exports/` folder.

> **Scope note.** Under the default *generalist* profile the dashboard makes **no
> drinking-water safety claim**. It shows model-monitored measurements against
> documented reference values only. See `ASSUMPTIONS.md`.

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
| **Overview** | Latest state, trend, anomaly & health per parameter; system KPIs. |
| **Parameter detail** | Actual vs predicted history, anomaly markers, reference line, date filter, data quality, SHAP explanation. |
| **Forecast** | Multi-step forecast **when exported** (empty state otherwise — never fabricated). |
| **Anomalies & alerts** | All flagged anomalies across parameters, chronological. |
| **Model health** | Version, RMSE, MAE, skill vs persistence, pending approvals, rejections. |
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
├── tests/               # 51 tests (unit + Streamlit AppTest smoke)
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
| Model version / metrics show `—` | `status.json` field is `null`/absent (e.g. `model_version`, or R²/drift which are not exported). |
| Forecast view is empty | Forecasts are not in the current export contract (by design — not fabricated). |
| `ModuleNotFoundError: dashboard` | Run from the **repository root**, or use the helper scripts. |
| Arabic layout looks LTR | Toggle the language in the sidebar, or set `WQD_DEFAULT_LANG=ar`. |
