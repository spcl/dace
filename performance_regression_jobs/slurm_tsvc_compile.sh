#!/bin/bash
#SBATCH --job-name=dace-tsvc-compile-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node = one per Grace CPU (GH200 node = 4 Grace sockets)
#SBATCH --cpus-per-task=72       # 72 cores per Grace CPU -> 4 x 72 = 288 cores = the whole node
#SBATCH --hint=nomultithread     # Grace Neoverse-V2 has no SMT (1 thread/core); keep it explicit
#SBATCH --time=12:00:00          # single-compiler over BOTH corpora; scales x len(CXXES) x len(CORPORA)
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=tsvc_compile_%j.out
#SBATCH --error=tsvc_compile_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Compile-speed + post-compile-performance comparison of the 3 DaCe pipelines
# (auto_opt, parallel = simplify+loop2map+mapfusion, canon) on BOTH TSVC corpora
# -- TSVC2 and TSVC2.5 -- distributed over nodes * ntasks-per-node ranks total.
#
# ONE job, TWO metrics per corpus into the SAME results tree (results/<...>/<corpus>/):
#   1. <corpus>_perf.py               -> post-compile RUNTIME  (speedup.md, correctness.md)
#   2. tsvc_compile_perf.py --corpus  -> COMPILE speed         (compile_total.md,
#                                        compile_codegen.md, compile_cxx.md)
# where <corpus> in {tsvc2, tsvc2_5} -- one script now covers both corpora via --corpus.
#
# Sweeps every compiler in $CXXES over every corpus in $CORPORA. The DEFAULT is a
# single compiler (clang++) over both corpora. Set CXXES to more than one to turn
# this into the CROSS-COMPILER sweep: each compiler builds every DaCe lane once and
# is timed for runtime + cold-compile, results namespaced per compiler
# (engine.compiler_host_tag = <compiler>_<host>_<preset>) so each kernel gets one
# row per compiler -- read a lane DOWN the rows to compare it ACROSS compilers.
# An absent compiler is skipped, not fatal. Both drivers self-partition kernels
# by SLURM_PROCID/SLURM_NTASKS; the final --tables-only passes are the cross-rank
# (and cross-compiler) aggregation step.
#
# Submit with:
#   sbatch slurm_tsvc_compile.sh                                    # clang++, both corpora
#   CORPORA=tsvc2 sbatch slurm_tsvc_compile.sh                      # one corpus only
#   CXXES="g++ clang++ icpx nvc++" sbatch slurm_tsvc_compile.sh     # cross-compiler
# Adjust --nodes / --ntasks-per-node for however many ranks you want.

set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72" OPENBLAS_NUM_THREADS="72"        # one Grace CPU's worth of cores per rank
export OMP_PROC_BIND="close"       # pin OpenMP threads, packed within the rank's socket
export OMP_PLACES="cores"          # one OpenMP place per physical core
export OMP_MAX_ACTIVE_LEVELS=1     # one parallel level only: a BLAS call inside a DaCe omp region must serialize, not fork its own team (openmp-threaded OpenBLAS honors this; the old pthreads pool could not)
export PYTHONUNBUFFERED=1  # otherwise stdout is fully buffered (not a tty), so progress prints
                           # don't show up in the log until a buffer fills -- looks like a hang

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load llvm@22.1.5    # clang++ = DaCe codegen compiler + Polly for the native-clang-polly-autopar lane
spack load cmake
spack load openblas
spack load cuda
spack load cutensor

# CUDA + cuTENSOR paths: `spack load` sets PATH but not LD_LIBRARY_PATH / CPATH / LIBRARY_PATH, so
# point DaCe's nvcc discovery (CUDA_HOME) and the cuTENSOR library environment (links -lcutensor,
# includes cutensor.h) at the spack installs. OpenBLAS + LAPACK (LAPACKE) come from `spack load
# openblas`; all four (openblas, lapack, cuda, cutensor) are assumed present on the node.
export CUDA_HOME="$(spack location -i cuda 2>/dev/null || echo "${CUDA_HOME:-}")"
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
fi
_CUTENSOR_ROOT="$(spack location -i cutensor 2>/dev/null || true)"
if [ -n "$_CUTENSOR_ROOT" ]; then
    for _d in "$_CUTENSOR_ROOT"/lib "$_CUTENSOR_ROOT"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
    done
    [ -d "$_CUTENSOR_ROOT/include" ] && export CPATH="$_CUTENSOR_ROOT/include:${CPATH:-}"
    unset _CUTENSOR_ROOT
fi

# OpenBLAS (one libopenblas = BLAS + CBLAS + LAPACK + LAPACKE): `spack load openblas` sets PATH but
# NOT LD_LIBRARY_PATH, and the install sits off the ldconfig cache, so DaCe's detection
# (dace/libraries/blas/environments/openblas.py) needs BOTH the OPENBLAS_DIR env var it now checks
# AND the lib dir on LD_LIBRARY_PATH. Without them MatMul/potrf report "OpenBLAS not installed" and
# expand to a naive pure loop (~25x slower); with them gemm/k2mm/k3mm/cholesky route to cblas_dgemm.
export OPENBLAS_DIR="$(spack location -i openblas 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
if [ -n "$OPENBLAS_DIR" ]; then
    for _d in "$OPENBLAS_DIR"/lib "$OPENBLAS_DIR"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
    done
    [ -d "$OPENBLAS_DIR/include" ] && export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"
fi

# Runtime library paths so every compiled .so loads at ctypes time: `spack load` sets PATH
# but NOT LD_LIBRARY_PATH. A -fopenmp kernel needs libomp/libgomp (spack llvm), and the
# clang/gcc codegen links spack-gcc's libstdc++/libgcc_s -- without these, loading a kernel
# fails with e.g. 'libomp.so: cannot open shared object file'. Each dir is asked of the
# compiler itself (-print-file-name) so only real toolchain dirs are added, de-duplicated.
# (native_harness.openmp_rpath_flags also rpaths the OpenMP dir into the native .so; this is
# the fallback that additionally covers the DaCe CMake lanes.)
_add_ldpath() {
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$1:"*) ;;
        *) [ -d "$1" ] && export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}" ;;
    esac
}
_ldpath_for() {  # $1=compiler, rest=libs; add each resolved lib's dir
    command -v "$1" >/dev/null 2>&1 || return 0
    local _cc="$1"; shift
    for _lib in "$@"; do
        _p="$("$_cc" -print-file-name="$_lib" 2>/dev/null)"
        [ -f "$_p" ] && _add_ldpath "$(dirname "$_p")"
    done
}
# OpenMP runtime: clang -> libomp, gcc -> libgomp.
_ldpath_for clang++ libomp.so libgomp.so
_ldpath_for g++    libgomp.so
# spack-gcc C++ runtime (libstdc++/libgcc_s): ask g++/gcc -- clang -print-file-name points at
# the SYSTEM gcc, but the codegen links spack gcc via --gcc-install-dir.
_ldpath_for g++ libstdc++.so.6 libgcc_s.so.1
_ldpath_for gcc libstdc++.so.6 libgcc_s.so.1
unset -f _add_ldpath _ldpath_for

# HPTT (High-Performance Tensor Transpose) is an EXTERNAL dependency -- built out-of-tree and
# not vendored/committed (see performance_regression_jobs/.gitignore). Point DaCe's HPTT
# library environment at the local build so TensorTranspose nodes find its headers
# (<HPTT_ROOT>/include) and link+load its lib (<HPTT_ROOT>/lib/libhptt.so). Build per
# https://github.com/springer13/hptt; here it lives next to this script.
export HPTT_ROOT="$PWD/hptt"
[ -f "$HPTT_ROOT/lib/libhptt.so" ] && export LD_LIBRARY_PATH="$HPTT_ROOT/lib:${LD_LIBRARY_PATH:-}"
# For icpx add:  source /opt/intel/oneapi/setvars.sh
# For nvc++ add: add the nvhpc compilers/bin to PATH (or load its modulefile)

# Corpora and compilers to sweep. Defaults = both TSVC corpora, single clang++;
# override CORPORA to run one corpus, or CXXES for the cross-compiler comparison.
CORPORA="${CORPORA:-tsvc2 tsvc2_5}"
CXXES="${CXXES:-clang++}"
REPS="${REPS:-25}"
COMPILE_REPS="${COMPILE_REPS:-5}"

# srun pinning: --distribution=block:block gives rank i a contiguous 72-core block = Grace socket i
# (72-72-72-72 across the node's 4 Grace CPUs); --cpu-bind=cores pins each rank to its cores
# (verbose logs the CPU mask so you can confirm the split in the job output).
for corpus in $CORPORA; do
  echo "[compile] ========== corpus: $corpus =========="
  for CXX in $CXXES; do
    if ! command -v "$CXX" >/dev/null 2>&1; then
      echo "[compile] skip '$CXX' (not on PATH)"; continue
    fi
    echo "[compile] === $corpus / $CXX ==="
    # || echo so one compiler's failure never aborts the rest or the table passes.
    srun --cpu-bind=verbose,cores --distribution=block:block python3 "${corpus}_perf.py" --reps "$REPS" --cxx="$CXX" || echo "[compile] runtime sweep failed for $corpus/$CXX"
    srun --cpu-bind=verbose,cores --distribution=block:block python3 tsvc_compile_perf.py --corpus "$corpus" --compile-reps "$COMPILE_REPS" --cxx="$CXX" || echo "[compile] compile sweep failed for $corpus/$CXX"
  done
done

# Cross-rank (and cross-compiler) aggregation per corpus: re-scans the whole results tree.
for corpus in $CORPORA; do
  python3 "${corpus}_perf.py" --tables-only
  python3 tsvc_compile_perf.py --corpus "$corpus" --tables-only
done

# Re-plot the native-baseline boxplot from the freshly-written tree (best-effort: never fail the job).
# It reads both tsvc2 and tsvc2_5 and draws whichever are present.
python3 plot_tsvc_boxplot.py --results-dir results --out results/tsvc_boxplot.png \
  || echo "[plot] tsvc_boxplot failed"
