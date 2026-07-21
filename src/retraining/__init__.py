"""
src/retraining — Periodic retraining module.

Generic, parameter-agnostic retraining pipeline that works for any model
that respects the ParameterModel contract (base.py).

Public surface:
  from src.retraining.data_loader     import load_available_data
  from src.retraining.drift_detector  import check_drift
  from src.retraining.retrain_manager import RetrainManager
  from src.retraining.model_versioning import save_model_version, load_latest_version, load_specific_version
"""
