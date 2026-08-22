#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# Daint/Alps submission: ONE corpus per job, on ONE node = 4 ranks.
#
# A Grace-Hopper node is four modules, each a 72-core Grace CPU with its own H100, so one node maps
# to 4 ranks x 72 threads with nothing left over. ``corpus_perf_job.py`` shards that corpus's
# kernels across the four ranks positionally (SLURM_PROCID of SLURM_NTASKS), and each rank takes a
# private build scratch, so the four never share a .dacecache.
#
# Four jobs rather than the two grouped ones (submit_daint_array.sh = poly+np,
# submit_daint_loops.sh = tsvc+tsvc25): each corpus then gets a queue slot, a time limit and a
# results directory of its own, and a slow polybench sweep cannot eat the wall clock npbench needed.
# Both grouped scripts still work and are unchanged.
#
# Usage:
#   bash canon_corpus_perf_job/submit_daint_corpus.sh poly
#   bash canon_corpus_perf_job/submit_daint_corpus.sh tsvc 2        # 2 nodes = 8 ranks
#
# Env overrides: PRESET, OMP_NUM_THREADS, TIMELIMIT, ACCOUNT, PARTITION, CANON_PERF_ARMS, OUT_DIR.
#
# ⛔ DO NOT replace the sbatch below with a direct ``srun``. Slurm's slurmstepd starts every task
# with SIGCHLD BLOCKED; the mask survives fork+exec, reaches cmake, and cmake's KWSys then waits in
# select() forever for a wakeup that never comes -- a configure that looks stuck but is really a
# lost signal. The only place the unblock works is corpus_perf_job.py's module scope, because srun
# execs it DIRECTLY with no shell in between. submit_corpus_perf.sh preserves that chain
# (``srun ... "$PY" corpus_perf_job.py``); a wrapper shell, setsid, nohup, timeout or
# ``env --ignore-signal=CHLD`` in between all reintroduce the hang. The rank refuses to start if it
# finds the mask still set, so a broken chain fails in the first second instead of after an hour.

set -euo pipefail

SUITE="${1:-}"
NODES="${2:-1}"
HERE=$(cd "$(dirname "$0")" && pwd)

case "$SUITE" in
    poly)   DEFAULT_TIME=04:00:00 ;;   # 30 kernels, but the paper preset makes them the big ones
    np)     DEFAULT_TIME=04:00:00 ;;   # 24 kernels; softmax alone peaks around 12 GiB
    tsvc)   DEFAULT_TIME=06:00:00 ;;   # 151 kernels, small each
    tsvc25) DEFAULT_TIME=04:00:00 ;;   # 72 kernels
    *)
        echo "usage: $(basename "$0") <poly|np|tsvc|tsvc25> [NODES]" >&2
        echo "  poly/np divide by the corpus numpy reference; tsvc/tsvc25 divide by seq-cpp." >&2
        exit 2
        ;;
esac

PRESET="${PRESET:-paper}"
OUT_DIR="${OUT_DIR:-$HERE/corpus_perf_results_${SUITE}_${PRESET}}"

# 72 threads is one rank's whole Grace CPU. Set here AND passed to srun as --cpus-per-task inside
# submit_corpus_perf.sh, because Slurm binds the task to that many cores and OpenMP has to agree
# with the binding or the ranks oversubscribe each other's cores and every timing is noise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-72}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONHASHSEED=0
export CANON_PERF_ARMS="${CANON_PERF_ARMS:-1}"
export TIMELIMIT="${TIMELIMIT:-$DEFAULT_TIME}"
export OUT_DIR

mkdir -p "$OUT_DIR"

sbatch --job-name="dace-corpus-${SUITE}" \
    --account="${ACCOUNT:-g34}" \
    --partition="${PARTITION:-normal}" \
    --exclusive \
    --time="$TIMELIMIT" \
    --nodes="$NODES" \
    --ntasks-per-node=4 \
    --cpus-per-task="$OMP_NUM_THREADS" \
    --output="$OUT_DIR/dace-corpus-${SUITE}-%j.out" \
    --error="$OUT_DIR/dace-corpus-${SUITE}-%j.out" \
    "$HERE/submit_corpus_perf.sh" "$NODES" --suites "$SUITE" --preset "$PRESET" --facet perf --out "$OUT_DIR"
