#!/usr/bin/env bash
# Launch the Water Quality Monitoring dashboard (bash / WSL / macOS / Linux).
# Run from anywhere; paths are resolved relative to the repository root.
#
#   ./dashboard/run_dashboard.sh
#
# Optional overrides (see dashboard/config/settings.py):
#   WQD_EXPORTS_DIR=/synced/exports WQD_DEFAULT_LANG=ar ./dashboard/run_dashboard.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/Scripts/python.exe"        # Windows venv layout
[ -x "$PY" ] || PY="$REPO_ROOT/.venv/bin/python" # POSIX venv layout
[ -x "$PY" ] || PY="python"                      # ambient fallback

exec "$PY" -m streamlit run "dashboard/app.py" --server.port 8501
