# Assumptions

Only genuinely necessary, non-scientific assumptions are recorded here. Each is
modular and can be replaced once the project stabilises. **No scientific
standard, safety threshold, WQI weight, or operational rule is assumed or
invented in this file** — those come only from the repository or explicit
approval.

| # | Assumption | Source / reason | Impact | Replacement path |
|---|-----------|-----------------|--------|------------------|
| 1 | **Water-use profile = `generalist`** (no drinking/irrigation safety verdict). | User decision (2026-07-24): "generalist for now, may change later". Concept note describes a generalist system; the only WQI uses drinking-water standards. | No safe/unsafe language. Drinking-water references remain on Methodology only and are never alert/chart thresholds. | Approve a deployment-specific water-use context and operational limits before changing operational semantics. |
| 2 | **Site = "C-1 — Ramgarh Station", single site.** | The only dataset/exports in the repo are the C-1 station. | Header shows one site; no site filter is rendered (there is only one). | Set `WQD_SITE_NAME` / `WQD_SITE_ID`; add a site selector when multi-site exports exist. |
| 3 | **Data-source mode = `historical`.** | Current exports are historical research/replay data, not an expected continuous IoT feed. | Header and Overview show “Historical dataset”; no Delayed/freshness alert is evaluated. | For a real feed set `WQD_DATA_SOURCE_MODE=continuous`, provide `WQD_DATA_SOURCE_NOTE`, and approve a cadence/SLA. The existing `3 ×` median-gap heuristic is then available but remains non-scientific. |
| 4 | **Language default = English**, with an Arabic (RTL) toggle. | User decision: bilingual EN/AR. Repo, concept note and target paper are English. | Both languages fully supported; English is the initial view. | Set `WQD_DEFAULT_LANG=ar` to open in Arabic. |
| 5 | **Read-only MVP** — no operator actions stored. | User decision. | No write-back, no acknowledgment state, no datastore. | Add a modular acknowledgment layer (local JSON) separate from the ML pipeline if approved later. |
| 6 | **No authentication** for the MVP. | Not requested; local/operational deployment assumed. | App is open on its host/port. | Front with the deployment's existing auth (reverse proxy / SSO) or Streamlit auth when a target is chosen. |
| 7 | **Branding**: text title only, no logo. | No logo/brand asset provided. | Neutral professional header; no invented brand marks. | Drop a logo asset in `dashboard/assets/` and reference it in `components/layout.py`. |
| 8 | **No model-health classification.** | No approved threshold/rule exists for RMSE, MAE, skill, rejections, or drift. | The UI uses a neutral “Model monitoring active” label and shows exported metrics separately. | Add an approved, documented and tested classification rule before introducing pass/fail colors or “Healthy” wording. |
| 9 | **Duplicate timestamp policy = last complete source row wins.** | The export has no sequence/revision identifier and fields must not be combined across monitoring results. | Deterministic whole-record selection; no measured/predicted substitution. | Replace with an exporter-provided unique ID/revision policy when available. |

## Deliberately NOT assumed

- No safety classification of water (safe/unsafe to drink or irrigate).
- No WQI thresholds, weights, ideal values, or classes beyond those already
  computed by `src/data/compute_wqi.py`.
- No forecast values — Forecast navigation stays hidden until the export contract
  includes valid forecast data (see `DATA_CONTRACT.md`).
- No R² or drift status — unavailable optional fields are omitted.
- No model-health pass/fail, regulatory compliance state, or operator action.
