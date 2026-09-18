#!/bin/bash
# ============================================================================
# IreneRewrite — CI Entrypoint Script (P5.9)
# ============================================================================
# Dispatches to: test, benchmark, shell
# Usage: docker compose run <service> [test|benchmark|shell]
# ============================================================================

set -euo pipefail

ACTION="${1:-test}"

echo "=========================================="
echo "IreneRewrite CI — $ACTION"
echo "Python: $(python --version 2>&1)"
echo "Solver: ${IRENE_CI_SOLVER:-unset}"
echo "=========================================="

case "$ACTION" in
    test)
        echo "[test] Running pytest suite..."
        python -m pytest \
            Irene/tests/ tests/ \
            --cov=Irene --cov-report=term-missing \
            --tb=short -q
        ;;
    benchmark)
        echo "[benchmark] Running gallery quick-mode..."
        cd benchmarks
        python run_gallery.py \
            --solver "${IRENE_CI_SOLVER:-clarabel}" \
            --quick \
            --timeout 120 \
            --output-dir ./results/
        ;;
    shell)
        echo "[shell] Dropping to interactive shell..."
        exec bash
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        echo "Usage: $0 [test|benchmark|shell]" >&2
        exit 1
        ;;
esac
