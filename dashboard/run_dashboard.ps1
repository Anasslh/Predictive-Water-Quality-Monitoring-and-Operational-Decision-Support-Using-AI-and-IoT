# Launch the Water Quality Monitoring dashboard (Windows PowerShell).
# Run from anywhere; paths are resolved relative to the repository root.
#
#   ./dashboard/run_dashboard.ps1
#
# Optional overrides (see dashboard/config/settings.py):
#   $env:WQD_EXPORTS_DIR = "D:\synced\exports"
#   $env:WQD_DEFAULT_LANG = "ar"

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Prefer the project virtual environment if present, else the ambient python.
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

& $Py -m streamlit run "dashboard/app.py" --server.port 8501
