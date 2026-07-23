# Monitor Stages — Operational Guide

`python run.py monitor` runs up to four pipeline stages each time a new sensor
measurement arrives. This guide explains which stages are active by default, why
some are opt-in, and how to control them.

---

## Stage overview

| # | Stage | Default | Flag to disable |
|---|-------|---------|-----------------|
| 1 | **Predict + SHAP** | always on | — |
| 2 | **Anomaly detection** | on (if detector exists) | `--no-anomaly` |
| 3 | **Retrain check** | **off** | `--check-retrain` to enable |
| 4 | **Multi-step forecast** | on (if config exists) | `--no-forecast` |

Stages 2 and 4 activate automatically once the corresponding artifact has been
trained (detector / forecaster config are both saved the first time you run
`onboard` or `review`). Stage 1 is unconditional.

---

## Stage 2 — Anomaly detection (default ON)

Requires `--actual-value` to compute the residual. Compares the residual against
the Isolation Forest trained on the train+val residual distribution. Returns a
score in [0, 1]; measurements with score ≥ 0.5 (configurable in
`system_config.json → anomaly_detection.score_threshold`) are flagged.

Because anomaly scoring is pure inference (no model training), it adds only a
few milliseconds. It is therefore safe to run on every measurement.

**Console output example:**
```
  ⚠ ANOMALY DETECTED — score=0.987 (threshold=0.500)
```
or, when normal:
```
  Anomaly check   : OK  (score=0.041, threshold=0.500)
```

**Missing detector:**
```
  Anomaly detection: not yet trained — skipping
```
Run `onboard` or `review` once to generate `anomaly_detector.pkl`.

---

## Stage 3 — Retrain check (default OFF, opt-in with `--check-retrain`)

Retrain check is disabled by default because it can take several seconds: it
runs a KS drift test on the new residual distribution versus the reference
distribution, and if drift is detected it trains a new candidate model on the
full accumulated dataset. In a live demo or a latency-sensitive deployment,
triggering this unexpectedly is disruptive.

Enable it explicitly when you have accumulated enough new measurements and want
to check whether the production model needs refreshing:

```bash
python run.py monitor \
    --parameter EC \
    --dataset data/processed/c1_with_wqi.csv \
    --new-row data/processed/ec_new_row_example.csv \
    --actual-value 231.4 \
    --check-retrain
```

**Console output when drift is not detected:**
```
  Retrain check   : drift_not_detected
```

**Console output when drift is detected and a candidate is queued:**
```
  Retrain alert   : Drift detected — candidate queued for review (approval_id: ...)
```

The candidate must then be reviewed with `python run.py approve`.

---

## Stage 4 — Multi-step forecast (default ON)

Recursively applies the frozen prediction model to forecast the next N steps.
`n_steps` is derived from the sensor's `forecast_horizon_hours` field in
`sensors_config.json` divided by the auto-detected measurement frequency:

```
n_steps = floor(forecast_horizon_hours / detected_freq_hours)   (minimum 1)
```

| Sensor type | `forecast_horizon_hours` | `n_steps` |
|-------------|--------------------------|-----------|
| Daily (24 h/step) | 72 (default) | **3** |
| Hourly (1 h/step) | 72 (default) | **72** |
| Daily, long horizon | 168 | 7 |

Uncertainty bands are empirical standard deviations estimated from the
validation set at training time, scaled by √step as a random-walk
approximation.

Forecasting is inference-only (no training), so it is fast enough to run on
every measurement. It activates automatically once a `forecaster_config.json`
exists in `models_store/<param>/` — generated automatically by `onboard` or
`review`.

**Console output example (daily EC sensor, default 3 steps):**
```
  Forecast (3 steps, 24h interval):
    D+1     756.58  ± 46.59
    D+2     756.58  ± 65.88
    D+3     756.58  ± 80.69
```

> **Note on flat forecasts:** tree-based models (XGBoost, RF) frequently
> produce identical predictions at all forecast steps because adjacent feature
> vectors fall in the same leaf nodes — this is a known property of recursive
> tree forecasting, not a pipeline bug. The uncertainty bands still widen
> correctly via the √step approximation.

---

## Simplified mode — prediction + SHAP only

For a first demo or when you just want the raw prediction without any
secondary stages:

```bash
python run.py monitor \
    --parameter EC \
    --dataset data/processed/c1_with_wqi.csv \
    --new-row data/processed/ec_new_row_example.csv \
    --no-anomaly \
    --no-forecast
```

This runs Stage 1 only (predict + SHAP). The JSONL export still fires.

---

## Artifacts required per stage

| Stage | File created by `onboard`/`review` |
|-------|------------------------------------|
| Anomaly detector | `models_store/<param>/anomaly_detector.pkl` |
| Forecaster | `models_store/<param>/forecaster_config.json` |
| Retrain manager | Built in-memory from `models_store/<param>/current_model.json` + dataset |

All artifacts are generated automatically the first time you run
`python run.py onboard --parameter <name>` or `python run.py review <name>`.
