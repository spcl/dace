#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# ONE job for the four in-repo corpora (polybench, npbench, tsvc, tsvc_2_5) at the ``paper``
# preset: X nodes x 4 ranks, one corpus per rank at the default single node. It only pins the
# environment and fans the ranks out; tests/perf/corpus_perf_job.py picks each rank's slice and
# drives the existing measurement drivers.
#
#   sbatch tests/perf/submit_corpus_perf.sh            # 1 node  = 4 ranks, one corpus each
#   sbatch tests/perf/submit_corpus_perf.sh 4          # 4 nodes = 16 ranks, 4 per corpus
#   bash   tests/perf/submit_corpus_perf.sh 1 --preset S --limit 2   # no SLURM: mpirun locally
#
# Arg 1 is the node count (default 1); every later argument is forwarded to the rank script
# (--preset/--facet/--limit/--check/--force/--out). Without SLURM it falls back to mpirun, and
# without mpirun to a single rank, so the same script is the local smoke test.
#
# Env: PYTHON (interpreter), OUT_DIR (results), OMP_NUM_THREADS (default cores/ranks-per-node).

#SBATCH --job-name=dace-corpus-perf
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --time=12:00:00

set -euo pipefail

NODES="${1:-1}"
RANKS_PER_NODE=4
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT_DIR:-$REPO/tests/perf/corpus_perf_results}"
mkdir -p "$OUT"

# Node count is an argument, so it cannot be an #SBATCH line: re-submit ourselves with it.
if [ -z "${SLURM_JOB_ID:-}" ] && command -v sbatch >/dev/null 2>&1; then
    exec sbatch --nodes="$NODES" --ntasks-per-node="$RANKS_PER_NODE" \
        --output="$OUT/dace-corpus-perf-%j.out" --error="$OUT/dace-corpus-perf-%j.out" "$0" "$@"
fi

# Held FIXED across every pipeline under comparison -- an A/B with a moving thread count or a
# moving compiler flag set measures nothing. -ffp-contract=off because bit-exact comparison
# dies under FP contraction.
CORES="${SLURM_CPUS_ON_NODE:-$(nproc)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((CORES / RANKS_PER_NODE > 0 ? CORES / RANKS_PER_NODE : 1))}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONHASHSEED=0
export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export UCX_VFS_ENABLE=n
export DACE_cache_distaware=1
export DACE_compiler_cpu_args="-std=c++14 -fPIC -Wall -Wextra -O3 -march=native -ffast-math -fno-finite-math-only -ffp-contract=off"

# A batch script does NOT inherit the interactive PATH, so a bare ``python`` is /usr/bin/python
# and ``import dace`` dies in 5s looking like a corpus bug. Resolve it once and fail loudly.
PY="${PYTHON:-$(command -v python3)}"
"$PY" -c 'import dace' >/dev/null 2>&1 || {
    echo "FATAL: $PY cannot import dace -- wrong interpreter, set PYTHON=" >&2
    exit 1
}

JOB=("$PY" "$REPO/tests/perf/corpus_perf_job.py" --out "$OUT" "${@:2}")
echo "repo=$REPO nodes=$NODES ranks/node=$RANKS_PER_NODE OMP=$OMP_NUM_THREADS out=$OUT py=$PY"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks-per-node="$RANKS_PER_NODE" --cpus-per-task="$OMP_NUM_THREADS" "${JOB[@]}"
elif command -v mpirun >/dev/null 2>&1; then
    mpirun -np "$((NODES * RANKS_PER_NODE))" "${JOB[@]}"
else
    "${JOB[@]}"
fi

# Every rank has exited by here, so the gather is a plain re-export of what they wrote.
"$PY" "$REPO/tests/perf/corpus_perf_job.py" --out "$OUT" --aggregate "${@:2}"
