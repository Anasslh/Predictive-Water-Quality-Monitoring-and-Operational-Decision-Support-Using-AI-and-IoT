# water-quality-cs

Predictive Water Quality Monitoring — CS side (ML + XAI)
SESC, Nile University

## Getting started

```bash
pip install -r requirements.txt
```

## Shared foundations (read before writing any code)

- `src/models/base.py` — abstract class `ParameterModel` that every per-parameter
  model (pH, EC, turbidity) must implement: `fit` / `predict` / `explain`.
- `src/xai/shap_wrapper.py` — call `compute_shap_explanation(model, X_row)` once
  your model is trained to get the top SHAP features as a plain list of dicts.

See `src/models/ec/train.py` for a complete working example.

## Team

3 people — one parameter each (pH, EC, turbidity). Assignments to be decided together.
See `reports/implementation_plan.md` for the detailed pipeline and scope.

## Data

- `data/raw/` — main dataset (IEEE DataPort, Ramgarh, 365 daily rows) + Kaggle/CPCB (reference only)
- See `reports/week2_report.md` (§5) for the dataset assessment.
