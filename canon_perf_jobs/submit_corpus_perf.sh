#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# ONE job for the four in-repo corpora (polybench 30, npbench 24, tsvc 151, tsvc_2_5 72 = 277
# kernels) at the ``paper`` preset, over the EIGHT comparison arms of the paper table. X nodes x 4
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
# ONE flag picks the measured table: CANON_PERF_ARMS=1, the eight comparison arms plus the
# ``seq-cpp`` denominator, nine in all. See README.md for the arm x SDFG x compiler x baseline matrix.
#
# Env: PYTHON (interpreter), OUT_DIR (results, default next to the scripts), OMP_NUM_THREADS (default cores/ranks-per-node),
# REPS (best-of repetitions, default 50), DACE_JOB_SCRATCH (build scratch root), ACCOUNT /
# PARTITION / TIMELIMIT (site-specific sbatch settings -- the #SBATCH defaults below are CSCS
# values, override them for another site).

#SBATCH --job-name=dace-corpus-perf
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72

set -euo pipefail

NODES="${1:-1}"
RANKS_PER_NODE=4
# The job folder is named OUTRIGHT rather than derived from BASH_SOURCE, because under sbatch
# BASH_SOURCE is not this file: Slurm copies the script into its spool
# (/var/spool/slurmd/job<id>/slurm_script) and runs the copy, so anything relative to the script's
# own path resolves inside the spool and the job dies before it starts. Repointing this at another
# checkout is a one-line edit here.
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/canon_perf_jobs
HERE=$(pwd)

# Both toolchains come from spack, and a batch script inherits none of the interactive module
# environment, so without these the arms fall back to whatever /usr/bin holds -- a different
# compiler than the one the run is supposed to be measuring. The setup script MUST be sourced first
# and NOTHING may `spack load` above this point: `spack load` is a shell function, and the bare
# `bin/spack` a plain `sbatch`/`bash` finds instead exits 1 under `set -e` and kills the job.
export SPACK_ROOT="${SPACK_ROOT:-/capstor/scratch/cscs/ybudanaz/aarch64/spack}"
if [ -f "$SPACK_ROOT/share/spack/setup-env.sh" ]; then
    . "$SPACK_ROOT/share/spack/setup-env.sh"
fi
if command -v spack >/dev/null 2>&1; then
    # Variant pins because both packages are installed twice: the other gcc@16.1.0 is ~graphite
    # (kills the autopar arms) and the other llvm@22.1.5 is ~mlir.
    spack load gcc@16.1.0 +graphite
    spack load llvm@22.1.5 +mlir
fi

# The repo root is walked up to rather than counted: the drivers run as ``python -m tests....`` from
# it, and a hardcoded depth breaks silently the moment this folder is renamed or moved.
REPO="$HERE"
while [ "$REPO" != "/" ] && [ ! -f "$REPO/dace/__init__.py" ]; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/dace/__init__.py" ] || {
    echo "FATAL: no dace checkout above $HERE" >&2
    exit 1
}
# Results NEXT TO the scripts (one JSON per kernel, plus a figure and two tables), so a finished run
# is one directory to copy off the cluster. Only the build scratch is memory-backed, and it is
# throwaway. The job script defaults to the same path, so bare and via-here runs land in one place.
OUT="${OUT_DIR:-$HERE/corpus_perf_results}"
mkdir -p "$OUT"

# Node count is an argument, so it cannot be an #SBATCH line: re-submit ourselves with it. The
# site settings go on the command line too, where they beat the #SBATCH defaults in the header.
if [ -z "${SLURM_JOB_ID:-}" ] && command -v sbatch >/dev/null 2>&1; then
    exec sbatch --nodes="$NODES" --ntasks-per-node="$RANKS_PER_NODE" \
        --account="${ACCOUNT:-g34}" --partition="${PARTITION:-normal}" --time="${TIMELIMIT:-04:00:00}" \
        --output="$OUT/dace-corpus-perf-%j.out" --error="$OUT/dace-corpus-perf-%j.out" "$0" "$@"
fi

# Held FIXED across every arm under comparison -- an A/B with a moving thread count or a moving
# compiler flag set measures nothing. OMP_NUM_THREADS is exported before any python starts: the
# harness derives gcc's compile-time -ftree-parallelize-loops=N from it, and its per-arm probe
# refuses to launch if that ever stops emitting a parallel region.
#
# DERIVED from the node, not hardcoded: this same script is the local smoke test, and a fixed 72
# (a 288-core CSCS node / 4 ranks, which the formula reproduces there) is 4.5x oversubscription on
# a 16-core box.
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
# The whole point of the job: the nine-arm table, not the two-pipeline default. Read at import, so
# it reaches the pytest entry point too. ``CANON_PERF_ARMS=0`` in the caller's environment still
# wins -- that is the documented smoke-run escape hatch on a box without a polyhedral clang.
export CANON_PERF_ARMS="${CANON_PERF_ARMS:-1}"
# Per-kernel wall-clock cap, and it has to clear the NINE arms: CANON_PERF_MAX_CALL_MS caps one
# timed call at 180s, so nine arms are 1620s of timing alone before a single build. The alarm fires
# on whichever arm is running when it expires, so a cap sized for fewer arms loses a late arm for a
# reason that has nothing to do with that arm. It is still a cap: a hung kernel cannot eat the job.
export CANON_PERF_TIMEOUT="${CANON_PERF_TIMEOUT:-2400}"
# NOTE the gcc autopar width is deliberately NOT pinned here. ``resolve_autopar_hint`` probes this
# compiler and takes the largest width that actually emits a parallel region (40 on g++ 16.1.0 /
# Neoverse V2, where 41 and above silently emit none while Graphite is on). Exporting
# CANON_PERF_GCC_AUTOPAR_HINT_CAP would hardcode a site number the probe already discovers, and
# would go stale the moment the compiler changes.
#
# The four DaCe arms lower gemv/gemm to OpenBLAS; numpy -- the figure's reference -- is OpenBLAS, so
# expanding them to dace's shipped `pure` default instead compares a naive triple loop against a
# tuned kernel and loses by 20-40x on the kernels that ARE one BLAS call (atax, bicg, covariance).
# dace defaults library.blas to `pure` while lapack and linalg already default to OpenBLAS, so BLAS
# is the odd one out and has to be asked for. The serialize arms stay `pure` on purpose -- see the
# Arm.blas docstring. Ask spack for the prefix rather than hardcoding a hash, and fall back to
# whatever is already on the path so a non-spack box still runs.
# Per-COMPILER OpenBLAS: the clang arms get the %clang build (links libomp), the gcc arms the %gcc
# build (links libgomp), so no arm's BLAS drags the other OpenMP runtime into its process. The
# harness switches OPENBLAS_DIR per arm from these two. ``threads=openmp`` stays in both specs: a
# pthreads build collapses threads and measured 24x.
# The compiler is PINNED to the exact version each arm compiles with. A bare ``%gcc`` went
# ambiguous the moment openblas was built for both gcc@14.2.0 and gcc@16.1.0, and because the
# lookup was ``2>/dev/null || true`` it failed SILENTLY -- the gcc arms would have run with no
# OPENBLAS_DIR at all, which is a wrong measurement that still produces numbers. Ambiguity is now
# fatal here rather than empty.
if command -v spack >/dev/null 2>&1; then
    openblas_prefix() {
        local out
        if ! out=$(spack location -i openblas threads=openmp %"$1" 2>&1); then
            echo "FATAL: cannot resolve openblas threads=openmp %$1" >&2
            echo "$out" >&2
            exit 1
        fi
        echo "$out"
    }
    CANON_PERF_OPENBLAS_GCC="${CANON_PERF_OPENBLAS_GCC:-$(openblas_prefix "${CANON_PERF_GCC_SPEC:-gcc@16.1.0}")}"
    CANON_PERF_OPENBLAS_LLVM="${CANON_PERF_OPENBLAS_LLVM:-$(openblas_prefix "${CANON_PERF_LLVM_SPEC:-llvm@22.1.5}")}"
    [ -d "$CANON_PERF_OPENBLAS_GCC" ] && export CANON_PERF_OPENBLAS_GCC || unset CANON_PERF_OPENBLAS_GCC
    [ -d "$CANON_PERF_OPENBLAS_LLVM" ] && export CANON_PERF_OPENBLAS_LLVM || unset CANON_PERF_OPENBLAS_LLVM
fi
if [ -z "${OPENBLAS_DIR:-}" ]; then
    OPENBLAS_DIR="${CANON_PERF_OPENBLAS_GCC:-${CANON_PERF_OPENBLAS_LLVM:-}}"
    [ -d "$OPENBLAS_DIR" ] && export OPENBLAS_DIR
fi
# DELIBERATELY no openblas dir on LD_LIBRARY_PATH: the build links the full .so path, so cmake
# bakes each arm's dir into the kernel RUNPATH -- and LD_LIBRARY_PATH would override RUNPATH,
# forcing ONE variant onto every arm.
export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export UCX_VFS_ENABLE=n
export DACE_cache_distaware=1

# A spack-installed compiler puts its OpenMP runtime in its OWN prefix, which is on the link line
# and NOT on the loader path. So the build succeeds and the kernel dies at ctypes.CDLL with
# "libomp.so: cannot open shared object file" -- reported per kernel as an ERR, which reads like a
# corpus failure rather than a missing search path. Ask each compiler where its own runtime lives
# instead of guessing a prefix: -print-file-name answers with the absolute path when it resolves and
# echoes the argument back unchanged when it does not, hence the leading-slash test. Both runtimes
# go on the path because the nine arms span both toolchains (clang links libomp, gcc libgomp).
omp_runtime_dir() {
    lib="$("$1" -print-file-name="$2" 2>/dev/null || true)"
    case "$lib" in
    /*) [ -e "$lib" ] && dirname "$lib" ;;
    esac
}
for d in "$(omp_runtime_dir "${CANON_PERF_CLANG_CXX:-clang++}" libomp.so)" \
    "$(omp_runtime_dir "${CANON_PERF_GCC_CXX:-g++}" libgomp.so)"; do
    [ -n "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

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
# VERSIONED name, never bare ``python3``: setup.py declares python_requires='>=3.10', and a bare
# python3 under sbatch is the system interpreter, which here is old enough that dace dies on
# "import dataclasses" (stdlib only since 3.7). That error reads like a missing PyPI package and
# invites installing the wheel of that name -- a 3.6 backport stdlib shadows on every version dace
# supports, so it cannot help. Naming the version removes the ambiguity outright.
PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null 2>&1 || {
    echo "FATAL: no '$PY' on PATH" >&2
    echo "  a batch script gets no interactive PATH, so module-load it before sbatch (the module" >&2
    echo "  environment is exported to the job) or set PYTHON= to another >= 3.10 interpreter." >&2
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
import importlib.util
import sys

# Named BEFORE importing dace, and all at once. The perf driver is a dual-entry file -- a pytest
# module that also runs as a script -- so it imports pytest at top level even though the job never
# invokes pytest, and the plotter imports matplotlib. Discovering those one traceback at a time
# costs a queue wait each. ``find_spec`` only resolves the module, it does not import it, so this
# stays cheap enough to sit ahead of the expensive dace import.
missing = [m for m in ('numpy', 'pytest', 'matplotlib') if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"FATAL: missing modules: {' '.join(missing)}\n  pip install {' '.join(missing)}")

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

# Every rank has exited by here, so the gather is a plain re-export of what they wrote. ARMS=0 for
# this step ONLY: the exported 1 reaches the gather's own subprocess and re-probes all nine
# toolchains to copy numbers, so a post-sweep probe failure would cost the finished run its outputs.
CANON_PERF_ARMS=0 "$PY" "$HERE/corpus_perf_job.py" --out "$OUT" --aggregate "${@:2}"
