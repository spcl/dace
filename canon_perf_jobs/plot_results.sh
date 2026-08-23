#!/bin/bash
# Plot bar charts from a finished local array + loops sweep.
#
# Usage:
#   bash canon_perf_jobs/plot_results.sh [PRESET] [OUT_DIR]
# Defaults: PRESET=S, OUT_DIR=canon_perf_jobs/corpus_perf_results_${PRESET}_plots

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PRESET="${1:-S}"
PLOT_OUT="${2:-$HERE/corpus_perf_results_${PRESET}_plots}"

export PYENV_VERSION="${PYENV_VERSION:-py12}"

python3 "$HERE/plot_autoopt_vs_canon.py" \
    --array-dir "$HERE/corpus_perf_results_array_${PRESET}" \
    --loops-dir "$HERE/corpus_perf_results_loops_${PRESET}" \
    --out-dir "$PLOT_OUT"

echo "plots: $PLOT_OUT"
