#!/bin/bash
# THE local job. One launch, no SLURM:
#     ./run_local_job.sh
#
# Compiles all FOUR variants of every kernel through the real build path --
#   cmake-oldcpu | cmake-newcpu | native-oldcpu | native-newcpu
#   ( {compiler.build_mode} x {compiler.cpu.implementation: legacy | experimental_readable} )
# -- times each with REPS repetitions reporting the MEDIAN, then plots the headline comparison:
#
#     speedup = median(cmake-oldcpu) / median(cmake-newcpu)      (>1 => the new codegen is faster)
#
# Defaults: 25 reps (median), 8 OpenMP threads, paper dataset, single rank, this machine's g++ and
# system BLAS. Every variant runs the identical pipeline (dace + simplify + LoopToMap + MapFusion +
# ConvertLengthOneArraysToScalars); only the build mode and the code generator vary.
#
# Knobs (env-overridable, e.g. `ONLY=atax REPS=5 ./run_local_job.sh`):
#   THREADS   OpenMP threads                       (default 8)
#   REPS      timed repetitions per variant        (default 25, reduced by MEDIAN)
#   PRESET    paper (full sizes) | S (small)       (default paper)
#   CORPUS    npbench | polybench | both           (default both)
#   ONLY      substring filter on kernel name      (default none)
#   KERNELS   explicit comma-separated kernel list (default none)
#   CXX       host compiler                        (default g++; try clang++)
#   CPP_STD   C++ standard, digits only            (default 20)
#   TIMEOUT   per-variant subprocess timeout, s    (default 900)
#   PYTHON    interpreter                          (default python3)
#   OUT       TSV path                             (default local/local_compare.tsv)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"

THREADS="${THREADS:-8}"
REPS="${REPS:-25}"
PRESET="${PRESET:-paper}"
CORPUS="${CORPUS:-both}"
ONLY="${ONLY:-}"
KERNELS="${KERNELS:-}"
CXX="${CXX:-g++}"
CPP_STD="${CPP_STD:-20}"
TIMEOUT="${TIMEOUT:-900}"
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$HERE/local_compare.tsv}"
PLOT_DIR="${PLOT_DIR:-$HERE/plots}"

# --- local env: this machine's toolchain, N threads, ONE rank ------------------------------
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=close OMP_PLACES=cores
# engine.configure_dace_process reads DACE_PERF_CXX -> compiler.cpu.executable.
export DACE_PERF_CXX="$CXX"
# Pin the C++ standard for the CMake build (the readable generator emits consteval helpers under 20).
export DACE_compiler_cpp_standard="$CPP_STD"
export DACE_PERF_CXX_STD="c++$CPP_STD"
export MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n
export PYTHONUNBUFFERED=1
# Single rank, single node: clear any stray SLURM_* so nothing re-partitions the sweep.
unset SLURM_PROCID SLURM_NTASKS 2>/dev/null || true
# Read-only scalars bind by const reference (the legacy ABI, and the readable default). Set
# DACE_compiler_cpu_const_scalar_abi=by_value to measure the const-value binding instead.
export DACE_compiler_cpu_const_scalar_abi="${CONST_SCALAR_ABI:-by_ref}"

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: compiler '$CXX' not found on PATH" >&2; exit 1
fi

# COLD build by default: engine.configure_dace_process redirects DaCe's build folder to node-local
# RAM (/dev/shm/dace_perf_jobs_<uid>_rank0), which survives between runs -- leaving it warm turns
# "compile the 4 variants" into a cache no-op (compile_ms ~5ms) instead of a real build. Set
# KEEP_CACHE=1 to reuse it (faster re-runs when only the runtime numbers matter).
BUILD_CACHE="/dev/shm/dace_perf_jobs_$(id -u)_rank0"
if [ "${KEEP_CACHE:-0}" != "1" ]; then
    rm -rf "$BUILD_CACHE"
    echo "   cold build (cleared $BUILD_CACHE; KEEP_CACHE=1 to reuse)"
fi

ARGS=(--preset "$PRESET" --corpus "$CORPUS" --reps "$REPS" --timeout "$TIMEOUT" --out "$OUT")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
[ -n "$KERNELS" ] && ARGS+=(--kernels "$KERNELS")

echo "== local 4-variant codegen job =="
echo "   repo=$REPO_ROOT  cxx=$CXX ($($CXX -dumpversion 2>/dev/null))  c++$CPP_STD  threads=$THREADS"
echo "   preset=$PRESET  corpus=$CORPUS  reps=$REPS (median)  const_scalar_abi=$DACE_compiler_cpu_const_scalar_abi"
echo "   variants: cmake-oldcpu cmake-newcpu native-oldcpu native-newcpu"
echo "   out=$OUT"

"$PYTHON" "$HERE/local_compare.py" "${ARGS[@]}" \
    || echo "warning: local_compare.py returned nonzero; plotting whatever rows landed"

echo "--- plotting speedup ---"
"$PYTHON" "$HERE/plot_speedup.py" --tsv "$OUT" --out-dir "$PLOT_DIR" \
    || echo "warning: plot_speedup.py returned nonzero"

echo "done: $OUT   plots: $PLOT_DIR"
