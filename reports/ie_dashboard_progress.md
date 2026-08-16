# Progress — IE Track: Decision-Support Dashboard MVP

**Track:** Industrial Engineering
**Last reviewed:** 2026-07-31
**Status:** MVP corrections implemented locally; awaiting user review before commit.
**Branch:** `feature/dashboard-mvp`

---

## Context

The concept note assigns the **decision-support dashboard**, KPI presentation,
and data visualization to the IE track (weeks 4–5). The CS pipeline already
publishes a clean export contract (`exports/<param>.jsonl` +
`<param>_status.json`) as the intended IE↔CS hand-off. This deliverable builds
the dashboard on top of that contract, with no change to the ML pipeline.

## Delivered

- A read-only Streamlit dashboard (`dashboard/`) with five currently visible views:
  Overview, Parameter detail, Anomalies & alerts, Model monitoring, and Data &
  methodology. Forecast is code-ready and appears only with valid exported values.
- Consumes the real exports read-only; dynamic parameter discovery; robust to
  missing/partial/corrupt data.
- Bilingual **English / Arabic** (RTL), restrained operations-console design.
- Presents exported model-monitoring metrics neutrally (RMSE, MAE, skill vs
  persistence, pending approvals, rejections) and translates SHAP into
  operator-readable influence language.
- 63 automated tests (unit + Streamlit end-to-end) — all passing in the latest run.

## Scientific / safety posture

- **Generalist** framing: no drinking-water safety verdict is shown.
- WQI (drinking-water WAWQI, documented limitations) is confined to the
  methodology page as a historical diagnostic.
- Historical mode disables live freshness-alert semantics.
- Forecast navigation and optional status fields appear only where exports
  support them; no model-health class, R², drift status, or forecast is fabricated.
- Drinking-water WQI references are confined to Methodology and are not used as
  operational chart thresholds.

Full detail: `dashboard/IMPLEMENTATION_REPORT.md`,
`dashboard/DATA_CONTRACT.md`, `dashboard/ASSUMPTIONS.md`.

## Next

1. Define and approve an additive forecast export contract as a separate ML-pipeline task.
2. Generate pH / Turbidity exports to populate the multi-parameter overview.
3. Approve continuous-source freshness SLA and model-health classification rules.
4. Decide deployment target and authentication.

## Warm historical tier audit — August 2026

- Verified that the first SQL Server submission was committed on remote `main`
  but was not reproducible or globally integrated.
- Added a versioned SQL schema, safe environment-driven configuration,
  validated/idempotent archive ingestion, bounded read-only queries, and a
  global Current monitoring / Historical archive source selector on an audit
  branch pending review.
- Preserved parameter genericity, null measured values, anomaly meaning,
  chronological sorting, read-only dashboard behavior, and Methodology-only WQI.
- Warm mode deliberately omits current model status and pending approvals until
  a historical status contract exists.
- Cold export and SQL ingestion remain manual; no automation is claimed.
- Added unit/Streamlit coverage and a separately gated live SQL Server
  integration test. Exact final counts and runtime evidence belong in the audit
  handoff after the complete suite and browser review.

Full Warm architecture and setup: `dashboard/WARM_TIER.md`.
