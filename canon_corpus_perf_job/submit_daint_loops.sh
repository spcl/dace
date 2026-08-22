#!/bin/bash
# Daint/Alps submission: tsvc + tsvc_2_5.
#
# Distributes kernels across N nodes x 4 MPI ranks per node.
#
# Usage:
#   bash canon_corpus_perf_job/submit_daint_loops.sh [NODES]
# Env overrides:
#   PRESET=S
#   OMP_NUM_THREADS=72
#   TIMELIMIT=08:00:00
#   ACCOUNT=g34
#   PARTITION=normal
#   CANON_PERF_ARMS=0
#   OUT_DIR=...

set -euo pipefail

NODES="${1:-1}"
HERE=$(cd "$(dirname "$0")" && pwd)
PRESET="${PRESET:-S}"
OUT_DIR="${OUT_DIR:-$HERE/corpus_perf_results_loops_${PRESET}}"

export CANON_PERF_ARMS="${CANON_PERF_ARMS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-72}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONHASHSEED=0
export TIMELIMIT="${TIMELIMIT:-08:00:00}"
export OUT_DIR

mkdir -p "$OUT_DIR"

sbatch --job-name=dace-corpus-loops \
    --account="${ACCOUNT:-g34}" \
    --partition="${PARTITION:-normal}" \
    --exclusive \
    --time="$TIMELIMIT" \
    --nodes="$NODES" \
    --ntasks-per-node=4 \
    --cpus-per-task="$OMP_NUM_THREADS" \
    --output="$OUT_DIR/dace-corpus-loops-%j.out" \
    --error="$OUT_DIR/dace-corpus-loops-%j.out" \
    "$HERE/submit_corpus_perf.sh" "$NODES" --suites loops --preset "$PRESET" --facet perf --out "$OUT_DIR"
