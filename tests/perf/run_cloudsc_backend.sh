#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# Graph backend (networkx vs rustworkx) wall time on CloudSC: simplify, config-prop
# (specialize_scalar + SDFG.specialize) + LoopUnroll, codegen, compile, serialize,
# deserialize. Median over N repetitions -- see graph_backend_cloudsc_bench.py.
#
#   tests/perf/run_cloudsc_backend.sh [reps] [outdir]
#
# Same shape as tests/perf/run_perf_ab.sh (the existing old-vs-new-checkout A/B driver,
# whose 3rd workload is this same simplify -> loopunroll -> codegen sequence on CloudSC):
# MPI anti-hang env vars, DACE_compiler_use_cache=0 so every repetition genuinely
# re-optimizes/re-codegens rather than serving a cached result, PYTHONPATH set explicitly
# since this runs as a plain script (not `python -m ...`).
#
# Runs directly too (bash tests/perf/run_cloudsc_backend.sh ...), no SLURM required --
# submit_cloudsc_backend.sh just wraps this with #SBATCH directives and the toolchain
# spack-loads needed for the compile phase.
set -euo pipefail

REPS="${1:-10}"
REPO=$(git rev-parse --show-toplevel)
OUTDIR="${2:-${REPO}/cloudsc_backend_results}"
mkdir -p "${OUTDIR}"

export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n MPI4PY_RC_INITIALIZE=0
export NOSTATUSBAR=1
export DACE_cache=unique
export DACE_compiler_use_cache=0
export DACE_default_build_folder="${OUTDIR}/dacecache"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-1}"
# sys.path[0] would otherwise be tests/perf/, not the repo root -- tests.corpus.cloudsc's
# absolute-from-root import fails without this (confirmed: ModuleNotFoundError: No module
# named 'tests.corpus').
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# Whatever `python3` resolves to in the calling environment (matches tests/perf/run_perf_ab.sh's
# own convention) -- set PYTHON_BIN explicitly to override, e.g. for a specific pyenv/venv.
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO}"

echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Reps: ${REPS}"
echo "Python: ${PYTHON_BIN}"
echo "Output dir: ${OUTDIR}"

"${PYTHON_BIN}" tests/perf/graph_backend_cloudsc_bench.py \
    --reps "${REPS}" \
    --output "${OUTDIR}/results.json" \
    --table-output "${OUTDIR}/results_table.md" \
    --tmp-dir "${OUTDIR}/scratch"

echo
echo "Done. Table:"
cat "${OUTDIR}/results_table.md"
