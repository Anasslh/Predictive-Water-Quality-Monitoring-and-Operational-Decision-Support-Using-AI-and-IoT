#!/usr/bin/env bash
# build_deployment_zip.sh — Canonical packaging script for the deployment zip.
#
# USAGE
#   bash build_deployment_zip.sh
#   bash build_deployment_zip.sh --output my_custom_name.zip
#
# DESIGN
#   Uses an EXPLICIT whitelist of files, not a directory sweep.
#   Any new production file must be added manually to FILES_TO_INCLUDE below.
#   This prevents research / evaluation / per-parameter scripts from silently
#   entering the deployment package — the root cause of three prior packaging bugs.
#
# EXCLUDED (intentionally, documented):
#   src/models/ec/          — EC-specific legacy model classes; pipeline uses
#                             the generic _GenericXGBoostModel / _GenericRFModel /
#                             _GenericSVRModel from model_benchmark.py (confirmed
#                             via import audit: zero references from production code)
#   src/data/compute_wqi.py — standalone WQI computation script; not part of the
#                             prediction/monitoring pipeline
#   src/data/feature_engineering.py — early per-EC feature engineering script
#                             superseded by feature_engineering_generic.py
#   src/evaluation/         — research / experiment scripts; not production code
#   src/anomaly/validate_ec_*.py — standalone validation scripts; not production
#   src/forecasting/validate_ec_forecast.py — standalone validation script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT="${1:-}"
if [ -z "$OUTPUT" ] && [ "${1:-}" = "--output" ]; then
  OUTPUT="${2:-water_quality_pipeline_deploy.zip}"
fi
OUTPUT="${OUTPUT:-water_quality_pipeline_deploy.zip}"
# Strip --output flag if passed positionally
if [[ "$OUTPUT" == --* ]]; then
  OUTPUT="${2:-water_quality_pipeline_deploy.zip}"
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "[build] Staging to: $STAGE"
echo "[build] Output:     $OUTPUT"

# ── Explicit file whitelist ────────────────────────────────────────────────────
# To add a new production file: append it here.
# Format: "source_path_relative_to_project_root"

FILES_TO_INCLUDE=(
  # Root entry points
  "run.py"
  "run.sh"
  "run_tests.sh"
  "README.md"
  "requirements.txt"
  "GUIDE_MONITOR_STAGES.md"

  # Configuration
  "config/sensors_config.json"
  "config/system_config.json"

  # Reference dataset + monitor example row
  "data/processed/c1_with_wqi.csv"
  "data/processed/ec_new_row_example.csv"

  # Core package marker
  "src/__init__.py"
  "src/_logging.py"
  "src/config.py"

  # Anomaly detection
  "src/anomaly/__init__.py"
  "src/anomaly/detector.py"
  "src/anomaly/explain.py"
  "src/anomaly/residual.py"
  "src/anomaly/test_anomaly_wired.py"

  # Data utilities (production only — NOT compute_wqi.py / feature_engineering.py)
  "src/data/__init__.py"
  "src/data/split.py"
  "src/data/validation.py"

  # Forecasting
  "src/forecasting/__init__.py"
  "src/forecasting/feature_engineering_generic.py"
  "src/forecasting/frequency_detector.py"
  "src/forecasting/recursive_forecaster.py"
  "src/forecasting/test_forecaster_mechanics.py"
  "src/forecasting/test_n_steps_calculation.py"

  # Model base class + empty parameter packages
  "src/models/__init__.py"
  "src/models/base.py"
  "src/models/ph/__init__.py"
  "src/models/turbidity/__init__.py"
  # NOTE: src/models/ec/ intentionally excluded (see header comment)

  # Monitor
  "src/monitor/__init__.py"
  "src/monitor/archive.py"
  "src/monitor/export.py"
  "src/monitor/test_archive.py"
  "src/monitor/test_export.py"
  "src/monitor/test_monitor_raw_input.py"
  "src/monitor/test_status_export.py"

  # Pipeline (onboarding, benchmarking, timestamp detection)
  "src/pipeline/__init__.py"
  "src/pipeline/model_benchmark.py"
  "src/pipeline/orchestrator.py"
  "src/pipeline/test_batch_onboard.py"
  "src/pipeline/test_monitor_stages.py"
  "src/pipeline/test_onboarding_guard.py"
  "src/pipeline/test_status_pending.py"
  "src/pipeline/test_timestamp_detector.py"
  "src/pipeline/timestamp_detector.py"

  # Retraining workflow
  "src/retraining/__init__.py"
  "src/retraining/approval.py"
  "src/retraining/cli_approve.py"
  "src/retraining/data_loader.py"
  "src/retraining/drift_detector.py"
  "src/retraining/model_versioning.py"
  "src/retraining/retrain_manager.py"
  "src/retraining/test_atomic_writes.py"
  "src/retraining/test_cli_approve.py"
  "src/retraining/test_consecutive_rejections.py"

  # XAI
  "src/xai/__init__.py"
  "src/xai/shap_wrapper.py"
)

# ── Copy files into staging, preserving directory structure ───────────────────

MISSING=()
for f in "${FILES_TO_INCLUDE[@]}"; do
  if [ ! -f "$f" ]; then
    MISSING+=("$f")
    continue
  fi
  dest="$STAGE/$f"
  mkdir -p "$(dirname "$dest")"
  cp "$f" "$dest"
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo ""
  echo "[build] ERROR — the following whitelisted files are missing from the repo:"
  for f in "${MISSING[@]}"; do
    echo "  MISSING: $f"
  done
  echo "Fix the whitelist or restore the missing files before packaging."
  exit 1
fi

# ── Build zip ─────────────────────────────────────────────────────────────────

rm -f "$OUTPUT"
(cd "$STAGE" && zip -r "$SCRIPT_DIR/$OUTPUT" . --exclude="*/__pycache__/*" --exclude="*.pyc" -q)

# ── Report ────────────────────────────────────────────────────────────────────

FILE_COUNT=$(unzip -l "$OUTPUT" | tail -n +4 | head -n -2 | awk '{print $NF}' | grep -v '/$' | wc -l)
echo "[build] Done. $FILE_COUNT files in $OUTPUT"
echo ""
echo "[build] Full listing:"
unzip -l "$OUTPUT" | tail -n +4 | head -n -2 | awk '{print $NF}' | grep -v '/$' | sort | sed 's/^/  /'
echo ""
unzip -l "$OUTPUT" | tail -n 2
