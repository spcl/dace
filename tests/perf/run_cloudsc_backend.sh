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

# submit_cloudsc_backend.sh's #SBATCH --output is a fixed path relative to the submit dir
# (SLURM can't expand $OUTDIR in a #SBATCH line), so it lands outside OUTDIR whenever a
# custom outdir is passed. Mirror everything into OUTDIR/run.log too, so the log is always
# findable next to dacecache/scratch regardless of where SLURM's own -o file went.
exec > >(tee -a "${OUTDIR}/run.log") 2>&1

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

# Pick an interpreter that actually satisfies dace's floor (setup.py: >=3.10). Do NOT just
# trust the name `python3`: on a Cray/SLES compute node that can be the ancient system 3.6
# while `python` is the loaded 3.12, which fails deep inside the import with a baffling
# "No module named 'dataclasses'" (stdlib since 3.7) instead of an actionable message.
python_ok() {
    command -v "$1" >/dev/null 2>&1 && "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        >/dev/null 2>&1
}

if [ -n "${PYTHON_BIN:-}" ]; then
    python_ok "${PYTHON_BIN}" || {
        echo "ERROR: PYTHON_BIN='${PYTHON_BIN}' is missing or older than the 3.10 dace requires." >&2
        exit 1
    }
else
    for candidate in python3 python; do
        python_ok "${candidate}" && PYTHON_BIN="${candidate}" && break
    done
    [ -n "${PYTHON_BIN:-}" ] || {
        echo "ERROR: no python >=3.10 found (tried: python3, python). Set PYTHON_BIN explicitly." >&2
        exit 1
    }
fi

# The whole point of this benchmark is networkx vs rustworkx. Without rustworkx the run still
# "succeeds" but silently measures one backend -- on an exclusive multi-hour allocation that is
# a wasted job, so fail fast instead. Override for a deliberate single-backend baseline run.
if ! "${PYTHON_BIN}" -c 'import rustworkx' >/dev/null 2>&1; then
    if [ "${ALLOW_MISSING_RUSTWORKX:-0}" != "1" ]; then
        echo "ERROR: rustworkx is not installed for ${PYTHON_BIN} -- this benchmark would only" >&2
        echo "       measure the networkx backend. Install it:" >&2
        echo "         ${PYTHON_BIN} -m pip install rustworkx        # or: pip install -e '.[fastgraph]'" >&2
        echo "       Set ALLOW_MISSING_RUSTWORKX=1 for a deliberate networkx-only baseline." >&2
        exit 1
    fi
    echo "WARNING: rustworkx missing, running networkx-only (ALLOW_MISSING_RUSTWORKX=1)."
fi

cd "${REPO}"

echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Reps: ${REPS}"
echo "Python: ${PYTHON_BIN} ($(${PYTHON_BIN} -c 'import sys; print(sys.version.split()[0])'))"
echo "Output dir: ${OUTDIR}"

# -u: stdout is a PIPE here (the tee above), so Python block-buffers it and a multi-hour run
# shows nothing until it exits -- the warnings still appear because stderr is unbuffered,
# which makes it look hung rather than merely quiet.
"${PYTHON_BIN}" -u tests/perf/graph_backend_cloudsc_bench.py \
    --reps "${REPS}" \
    --output "${OUTDIR}/results.json" \
    --table-output "${OUTDIR}/results_table.md" \
    --plot-output "${OUTDIR}/results_plot.png" \
    --tmp-dir "${OUTDIR}/scratch"

echo
echo "Done. Table:"
cat "${OUTDIR}/results_table.md"
