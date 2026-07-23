#!/usr/bin/env bash
# run.sh — Thin bash wrapper around run.py.
#
# Usage:
#   ./run.sh onboard --dataset data/processed/c1_with_wqi.csv --parameter Turbidity
#   ./run.sh monitor  --parameter EC --new-row data/processed/ec_X_test.csv
#   ./run.sh status
#   ./run.sh approve
#
# If a Python virtual environment exists in venv/ or .venv/ at the project
# root, it is activated automatically before running run.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Activate venv if present ──────────────────────────────────────────────────
for VENV_DIR in "$SCRIPT_DIR/venv" "$SCRIPT_DIR/.venv"; do
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$VENV_DIR/bin/activate"
        break
    fi
done

# ── Delegate to run.py ────────────────────────────────────────────────────────
python "$SCRIPT_DIR/run.py" "$@"
