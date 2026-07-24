# Assumptions

Only genuinely necessary, non-scientific assumptions are recorded here. Each is
modular and can be replaced once the project stabilises. **No scientific
standard, safety threshold, WQI weight, or operational rule is assumed or
invented in this file** — those come only from the repository or explicit
approval.

| # | Assumption | Source / reason | Impact | Replacement path |
|---|-----------|-----------------|--------|------------------|
| 1 | **Water-use profile = `generalist`** (no drinking/irrigation safety verdict). | User decision (2026-07-24): "generalist for now, may change later". Concept note describes a generalist system; the only WQI uses drinking-water standards. | No safe/unsafe language anywhere. Reference lines shown neutrally with provenance. | Set `WQD_WATER_USE=drinking` (or `irrigation`) once a deployment context is approved. Gate already implemented in `config/reference_limits.py`; only relabels documented values, never invents new ones. |
| 2 | **Site = "C-1 — Ramgarh Station", single site.** | The only dataset/exports in the repo are the C-1 station. | Header shows one site; no site filter is rendered (there is only one). | Set `WQD_SITE_NAME` / `WQD_SITE_ID`; add a site selector when multi-site exports exist. |
| 3 | **Data-freshness heuristic**: a parameter is "Delayed" when its newest record is older than `3 ×` the series' own median inter-arrival gap. | No freshness/SLA value exists in the repo. This is a **UI display heuristic**, not a scientific or safety threshold. | Drives only the header/overview freshness pill wording. | Tune via `WQD_FRESH_MULTIPLIER`, or replace with an operator-approved SLA when defined. |
| 4 | **Language default = English**, with an Arabic (RTL) toggle. | User decision: bilingual EN/AR. Repo, concept note and target paper are English. | Both languages fully supported; English is the initial view. | Set `WQD_DEFAULT_LANG=ar` to open in Arabic. |
| 5 | **Read-only MVP** — no operator actions stored. | User decision. | No write-back, no acknowledgment state, no datastore. | Add a modular acknowledgment layer (local JSON) separate from the ML pipeline if approved later. |
| 6 | **No authentication** for the MVP. | Not requested; local/operational deployment assumed. | App is open on its host/port. | Front with the deployment's existing auth (reverse proxy / SSO) or Streamlit auth when a target is chosen. |
| 7 | **Branding**: text title only, no logo. | No logo/brand asset provided. | Neutral professional header; no invented brand marks. | Drop a logo asset in `dashboard/assets/` and reference it in `components/layout.py`. |

## Deliberately NOT assumed

- No safety classification of water (safe/unsafe to drink or irrigate).
- No WQI thresholds, weights, ideal values, or classes beyond those already
  computed by `src/data/compute_wqi.py`.
- No forecast values — the forecast view stays empty until the export contract
  includes forecast data (see `DATA_CONTRACT.md`).
- No R² or drift status — not exported, shown as `—`.
