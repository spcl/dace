#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# ONE job for the four in-repo corpora (polybench 30, npbench 24, tsvc 151, tsvc_2_5 72 = 277
# kernels) at the ``paper`` preset, over the SIX comparison arms of the paper table. X nodes x 4
# ranks. It only pins the environment and fans the ranks out; corpus_perf_job.py picks
# each rank's kernels and drives the measurement.
#
#   sbatch submit_corpus_perf.sh            # 1 node  = 4 ranks
#   sbatch submit_corpus_perf.sh 4          # 4 nodes = 16 ranks
#   bash   submit_corpus_perf.sh 1 --preset S --limit 2   # no SLURM: mpirun locally
#
# Arg 1 is the node count (default 1); every later argument is forwarded to the rank script
# (--preset/--facet/--limit/--check/--force/--out). Without SLURM it falls back to mpirun, and
# without mpirun to a single rank, so the same script is the local smoke test.
#
# ONE flag picks the measured table: CANON_PERF_ARMS=1, the six arms plus the ``seq-cpp``
# denominator. See README.md for the arm x SDFG x compiler x baseline matrix.
#
# Env: PYTHON (interpreter), OUT_DIR (results, default next to the scripts), OMP_NUM_THREADS (default cores/ranks-per-node),
# REPS (best-of repetitions, default 50), DACE_JOB_SCRATCH (build scratch root), ACCOUNT /
# PARTITION / TIMELIMIT (site-specific sbatch settings -- the #SBATCH defaults below are CSCS
# values, override them for another site).

#SBATCH --job-name=dace-corpus-perf
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72

set -euo pipefail

NODES="${1:-1}"
RANKS_PER_NODE=4
# The three scripts travel together, so they are found relative to THIS file; the repo root is
# walked up to rather than counted, because the drivers run as ``python -m tests....`` from it and
# a hardcoded depth silently breaks the moment this folder is renamed or moved.
#
# Under sbatch, BASH_SOURCE is NOT this file: Slurm copies the script into its spool
# (/var/spool/slurmd/job<id>/slurm_script) and runs the copy, so the walk-up finds no checkout and
# the job dies before it starts. The submitting side therefore exports the folder it resolved, and
# the batch side takes that.
HERE="${CORPUS_JOB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$HERE"
while [ "$REPO" != "/" ] && [ ! -f "$REPO/dace/__init__.py" ]; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/dace/__init__.py" ] || {
    echo "FATAL: no dace checkout above $HERE" >&2
    [ -n "${SLURM_JOB_ID:-}" ] && echo "  (running from Slurm's spool copy; set CORPUS_JOB_HOME=/path/to/canon_corpus_perf_job)" >&2
    exit 1
}
export CORPUS_JOB_HOME="$HERE"
# Results NEXT TO the scripts (one JSON per kernel, plus a figure and two tables), so a finished run
# is one directory to copy off the cluster. Only the build scratch is memory-backed, and it is
# throwaway. The job script defaults to the same path, so bare and via-here runs land in one place.
OUT="${OUT_DIR:-$HERE/corpus_perf_results}"
mkdir -p "$OUT"

# Node count is an argument, so it cannot be an #SBATCH line: re-submit ourselves with it. The
# site settings go on the command line too, where they beat the #SBATCH defaults in the header.
if [ -z "${SLURM_JOB_ID:-}" ] && command -v sbatch >/dev/null 2>&1; then
    # --export carries CORPUS_JOB_HOME to the batch side, which runs Slurm's spool COPY of this
    # script and so cannot find the folder from its own path.
    exec sbatch --nodes="$NODES" --ntasks-per-node="$RANKS_PER_NODE" \
        --account="${ACCOUNT:-g34}" --partition="${PARTITION:-normal}" --time="${TIMELIMIT:-04:00:00}" \
        --export="ALL,CORPUS_JOB_HOME=$HERE" \
        --output="$OUT/dace-corpus-perf-%j.out" --error="$OUT/dace-corpus-perf-%j.out" "$0" "$@"
fi

# Held FIXED across every arm under comparison -- an A/B with a moving thread count or a moving
# compiler flag set measures nothing. -ffp-contract=off because bit-exact comparison dies under FP
# contraction. OMP_NUM_THREADS is exported before any python starts: the harness derives gcc's
# compile-time -ftree-parallelize-loops=N from it, and the preflight below refuses to launch if the
# two ever disagree.
#
# DERIVED from the node, not hardcoded: this same script is the local smoke test, and a fixed 72
# (a 288-core CSCS node / 4 ranks, which the formula reproduces there) is 4.5x oversubscription on
# a 16-core box. It is also not merely a runtime knob -- MEASURED on g++ 15.2 here, an autopar
# probe over a 4096x4096 affine nest emits GOMP_parallel at -ftree-parallelize-loops=4..33 and
# NOTHING at 48 and above, so a too-large thread count silently empties the gcc autopar arm. The
# preflight catches that (it refuses to launch), but the default should not walk into it.
CORES="${SLURM_CPUS_ON_NODE:-$(nproc)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((CORES / RANKS_PER_NODE > 0 ? CORES / RANKS_PER_NODE : 1))}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONHASHSEED=0
# Best-of repetitions, read by the timing harness. Measurement is a small part of the job -- all
# 277 kernels at 50 reps is ~9 minutes of timed work against hours of compiling -- so this buys
# tighter medians almost for free.
export CANON_PERF_REPS="${REPS:-50}"
# The whole point of the job: the six-arm table, not the two-pipeline default. Read at import, so
# it reaches the pytest entry point too. ``CANON_PERF_ARMS=0`` in the caller's environment still
# wins -- that is the documented smoke-run escape hatch on a box without a polyhedral clang.
export CANON_PERF_ARMS="${CANON_PERF_ARMS:-1}"
export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export UCX_VFS_ENABLE=n
export DACE_cache_distaware=1

# Build scratch, decided HERE and exported, so the path in the log is the path every rank and every
# cmake/compiler child inherits rather than something each rank defaults to on its own. In memory
# when /dev/shm demonstrably holds 4 GiB per rank on the node, else on disk. Never /tmp: it is the
# small tmpfs everything on the box shares and filling it fakes corpus failures.
SHM_FREE_GIB="$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 {print int($4 / 1048576)}' || true)"
NEED_GIB=$((4 * RANKS_PER_NODE))  # 4 GiB a rank, the same figure corpus_perf_job.SCRATCH_NEED_GIB uses
if [ -z "${DACE_JOB_SCRATCH:-}" ]; then
    if [ "${SHM_FREE_GIB:-0}" -ge "$NEED_GIB" ]; then
        DACE_JOB_SCRATCH=/dev/shm/dace-corpus-perf
    else
        DACE_JOB_SCRATCH="$HOME/.cache/dace-corpus-perf"
        echo "WARN: /dev/shm has ${SHM_FREE_GIB:-0} of $NEED_GIB GiB needed; compiles go to disk" >&2
    fi
fi
case "$DACE_JOB_SCRATCH" in
/tmp | /tmp/*)
    echo "FATAL: DACE_JOB_SCRATCH=$DACE_JOB_SCRATCH is under /tmp -- pick /dev/shm or a disk path" >&2
    exit 1
    ;;
esac
export DACE_JOB_SCRATCH TMPDIR="$DACE_JOB_SCRATCH/tmp"
mkdir -p "$TMPDIR"

# A batch script does NOT inherit the interactive PATH, so a bare ``python`` is /usr/bin/python and
# ``import dace`` dies in 5s looking like a corpus bug. Resolve it once and fail loudly.
#
# The probe is not a nicety. setup.py declares python_requires='>=3.10', and an unusable
# /usr/bin/python3 announces itself as "No module named 'dataclasses'" -- an error that reads like a
# missing PyPI package and invites installing the 3.6 backport of that name, which fixes nothing
# (dataclasses has been stdlib since 3.7; the wheel is a shim stdlib shadows on every version dace
# supports). So probe the interpreter for BOTH failure modes at once -- too old, and a stdlib too
# stripped to supply dataclasses -- by importing the module that actually fails. Demanding a printed
# token rather than a zero exit matters: PYTHON= pointed at a non-interpreter exits 0 happily.
py_ok() {
    [ -x "$1" ] || return 1
    [ "$("$1" -c 'import sys, dataclasses; print("ok" if sys.version_info[:2] >= (3, 10) else "old")' 2>/dev/null)" = ok ]
}
# Bare python3 is tried first but is exactly what breaks under sbatch (no interactive PATH, so it
# resolves to the system interpreter, not the module-loaded one), hence the versioned fallbacks.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
        found="$(command -v "$cand" || true)"
        if py_ok "$found"; then
            PY="$found"
            break
        fi
    done
fi
py_ok "$PY" || {
    echo "FATAL: '$PY' cannot run dace (needs >= 3.10 with a complete stdlib)" >&2
    [ -x "$PY" ] && echo "  it reports: $("$PY" -V 2>&1)" >&2
    echo "  set PYTHON=/path/to/python3.1x, or module-load it before sbatch: a batch script gets no" >&2
    echo "  interactive PATH, so 'python3' is the system one even when yours is newer." >&2
    echo "  Do NOT pip install 'dataclasses' -- that is a Python 3.6 backport of a stdlib module." >&2
    exit 1
}

# ONE preflight, on the node that does the work, ~2s: it resolves the interpreter, proves the pinned
# environment reached DaCe, and PROBES every arm -- each external pass must report itself on a known
# SCoP, and gcc's compile-time -ftree-parallelize-loops must equal OMP_NUM_THREADS, or the job dies
# here instead of publishing plain -O3 timings under an autopar label after hours of compiling. The
# evidence lines land in the job log, so no number is readable without its provenance.
# ``env -u`` because the harness registers the arms itself at import; one probe pass is enough.
echo "== preflight: arms=$CANON_PERF_ARMS omp=$OMP_NUM_THREADS py=$PY"
(cd "$REPO" && env -u CANON_PERF_ARMS "$PY" - "$CANON_PERF_ARMS") <<'PY'
import sys

import dace
from tests.passes.canonicalize.canonicalize_perf_corpus_test import ARMS, arms_requested, register_arms

if not dace.Config.get_bool('cache_distaware'):
    raise SystemExit('FATAL: DACE_cache_distaware did not reach dace -- ranks would share a build folder')
if arms_requested(sys.argv[1]):  # importing already probed the smoke-run pair, which needs no clang
    for label, line in register_arms(ARMS).items():
        print(f'  arm {label}: {line}')
PY

JOB=("$PY" "$HERE/corpus_perf_job.py" --out "$OUT" "${@:2}")
echo "repo=$REPO nodes=$NODES ranks/node=$RANKS_PER_NODE omp=$OMP_NUM_THREADS scratch=$DACE_JOB_SCRATCH out=$OUT"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # --export=ALL is srun's default, spelled out: a site that sets SLURM_EXPORT_ENV would otherwise
    # drop the pinned environment on the floor and every rank would measure something else.
    srun --export=ALL --ntasks-per-node="$RANKS_PER_NODE" --cpus-per-task="$OMP_NUM_THREADS" "${JOB[@]}"
elif command -v mpirun >/dev/null 2>&1; then
    mpirun -np "$((NODES * RANKS_PER_NODE))" "${JOB[@]}"
else
    "${JOB[@]}"
fi

# Every rank has exited by here, so the gather is a plain re-export of what they wrote.
"$PY" "$HERE/corpus_perf_job.py" --out "$OUT" --aggregate "${@:2}"
