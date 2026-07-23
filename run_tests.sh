#!/usr/bin/env bash
# run_tests.sh — Run all test_*.py files in the project.
#
# Discovery:  finds every file matching src/**/test_*.py (recursive).
# Execution:  each file is run with  python <file>  — tests must define
#             their own _run_all() / __main__ block (no pytest dependency).
#
# Exit code:  0 if all test files pass, 1 if any fail.
#
# Usage:
#   ./run_tests.sh
#   ./run_tests.sh --pytest      # use pytest discovery instead
#
# With --pytest: runs  pytest src/ -v  and exits with pytest's exit code.
# Pytest is only needed if you pass --pytest explicitly.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Activate venv if present ──────────────────────────────────────────────────
for VENV_DIR in "$ROOT/venv" "$ROOT/.venv"; do
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$VENV_DIR/bin/activate"
        break
    fi
done

# ── pytest mode ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--pytest" ]]; then
    echo "Running: pytest src/ -v"
    pytest src/ -v
    exit $?
fi

# ── Direct runner mode (no pytest required) ───────────────────────────────────
SEP="────────────────────────────────────────────────────────────────"
SEP2="════════════════════════════════════════════════════════════════"

mapfile -t TEST_FILES < <(find src -name "test_*.py" | sort)

if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "No test_*.py files found under src/"
    exit 0
fi

echo ""
echo "$SEP2"
echo "  run_tests.sh — ${#TEST_FILES[@]} test file(s) found"
echo "$SEP2"

PASSED=0
FAILED=0
FAILED_FILES=()

for f in "${TEST_FILES[@]}"; do
    echo ""
    echo "  $SEP"
    echo "  Running: $f"
    echo "  $SEP"
    if python "$f"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_FILES+=("$f")
    fi
done

echo ""
echo "$SEP2"
echo "  SUMMARY: $PASSED passed, $FAILED failed  (out of ${#TEST_FILES[@]} files)"
echo "$SEP2"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "  FAILED files:"
    for f in "${FAILED_FILES[@]}"; do
        echo "    ✗  $f"
    done
    echo ""
    exit 1
fi

echo ""
echo "  All tests passed."
echo ""
exit 0
