#!/bin/bash
#SBATCH --job-name=dace-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node = one per Grace CPU (GH200 node = 4 Grace sockets);
                                 # kernels self-partition across ranks (engine.my_slice)
#SBATCH --cpus-per-task=72       # 72 cores per Grace CPU -> 4 x 72 = 288 cores = the whole node
#SBATCH --hint=nomultithread     # Grace Neoverse-V2 has no SMT (1 thread/core); keep it explicit
#SBATCH --time=08:00:00          # 8h: a compile-heavy 4-lane sweep needs the headroom
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=perf_%j.out
#SBATCH --error=perf_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# ONE generic sbatch for all 8 jobs = {canon_vs, vector_vs} x {npbench, polybench,
# tsvc2, tsvc2_5}. Parameterized by environment (submit_perf_jobs.sh sets these
# per job via `sbatch --export`, and overrides --job-name/--output/--error):
#
#   EXPERIMENT   canon_vs | vector_vs
#   CORPUS       npbench | polybench | tsvc2 | tsvc2_5
#   CXX          C++ compiler for DaCe codegen (default: clang++)
#   RESULTS_DIR  results root for this job (default: results/$EXPERIMENT)
#
# Kernels are distributed across the job's 4 ranks automatically.

set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

EXPERIMENT="${EXPERIMENT:-canon_vs}"
CORPUS="${CORPUS:-tsvc2}"
CXX="${CXX:-clang++}"
DEVICE="${DEVICE:-cpu}"                       # cpu | gpu -- the dace lanes' target (native/numpy stay cpu)
RESULTS_DIR="${RESULTS_DIR:-results/$EXPERIMENT}"

# MPI anti-hang (must match run_perf.py's own os.environ.setdefault block) +
# threading. Exported so every rank / spawned subprocess inherits them.
export OMP_NUM_THREADS="72" OPENBLAS_NUM_THREADS="72"        # one Grace CPU's worth of cores per rank
export OMP_PROC_BIND="close"       # pin OpenMP threads, packed within the rank's socket
export OMP_PLACES="cores"          # one OpenMP place per physical core
export OMP_MAX_ACTIVE_LEVELS=1     # one parallel level only: a BLAS call inside a DaCe omp region must serialize, not fork its own team (openmp-threaded OpenBLAS honors this; the old pthreads pool could not)
export MPI4PY_RC_INITIALIZE="0"
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="self,vader"
export UCX_VFS_ENABLE="n"
export PYTHONUNBUFFERED=1  # otherwise stdout is fully buffered (not a tty) and progress
                          # prints don't appear until a buffer fills -- looks like a hang

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5
spack load cmake
spack load openblas threads=openmp
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
export OPENBLAS_DIR="$(spack location -i openblas threads=openmp 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
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

# GPU device: the dace lanes run on the GH200. Set the env the CPU sweep never needs -- Hopper arch
# (sm_90); nvcc host compiler g++-14, because CUDA 13.3 rejects host gcc>15 (the spack gcc 16 used for
# CPU); all kernels on stream 0 (DaCe still schedules streams; single default stream for now); and the
# always-on -O3 (-Xptxas/-Xcompiler) + fast-math device/host flags. Each rank binds ONE GPU via
# --gpus-per-task=1. Result folders carry a '-gpu' preset tag so gpu rows sit beside the cpu ones in the
# SAME results tree. (Validated on a debug node: canon_vs --device gpu, arc_distance dace-parallel 2376x.)
SRUN_GPU=""
if [ "$DEVICE" = "gpu" ]; then
    export DACE_compiler_cuda_cuda_arch=90
    export DACE_compiler_cuda_max_concurrent_streams=0
    export DACE_compiler_cuda_args="-Xptxas -O3 -Xcompiler -O3 -Xcompiler -march=native --use_fast_math -Xcompiler -Wno-unused-parameter"
    export CUDAHOSTCXX=/usr/bin/g++-14
    CXX=/usr/bin/g++-14
    SRUN_GPU="--gpus-per-task=1"
fi

echo "EXPERIMENT=$EXPERIMENT CORPUS=$CORPUS DEVICE=$DEVICE CXX=$CXX RESULTS_DIR=$RESULTS_DIR"

# --distribution=block:block hands rank i a contiguous 72-core block = Grace socket i, so the 4
# ranks split 72-72-72-72 across the node's 4 Grace CPUs; --cpu-bind=cores then pins each rank's
# threads to its own cores (verbose = log the CPU mask to the job output to confirm the split).
srun --cpu-bind=verbose,cores --distribution=block:block $SRUN_GPU python3 run_perf.py \
    --experiment "$EXPERIMENT" --corpus "$CORPUS" --device "$DEVICE" --reps 25 --cxx="$CXX" --results-dir="$RESULTS_DIR"

# One single-rank pass to (re)build the aggregate tables across every rank's output (device-agnostic:
# it scans the tree, so a shared cpu+gpu results dir yields one summary.csv with both device rows).
python3 run_perf.py --experiment "$EXPERIMENT" --corpus "$CORPUS" --tables-only --results-dir="$RESULTS_DIR"

# Re-plot from the freshly-written tree (best-effort: a plotting failure must never fail a
# measurement job). The plot scripts scan results/<experiment>/<corpus> and redraw every corpus
# present, so the last job to finish yields the complete multi-panel figure. PLOT_ROOT is the
# parent that CONTAINS the experiment dir (RESULTS_DIR defaults to results/$EXPERIMENT).
PLOT_ROOT="$(dirname "$RESULTS_DIR")"
case "$EXPERIMENT" in
  canon_vs)
    python3 plot_canon_vs.py --results-dir "$PLOT_ROOT" --out "$PLOT_ROOT/canon_vs.png" || echo "[plot] canon_vs failed"
    python3 plot_speedup_scatter.py --results-dir "$PLOT_ROOT" --out-prefix "$PLOT_ROOT/speedup_scatter" || echo "[plot] speedup_scatter failed"
    ;;
  vector_vs)
    python3 plot_vector_vs.py --results-dir "$PLOT_ROOT" --out "$PLOT_ROOT/vector_vs.png" || echo "[plot] vector_vs failed"
    ;;
esac
# tsvc corpora also feed the native-baseline boxplot (both experiments write the same tree).
case "$CORPUS" in
  tsvc2|tsvc2_5)
    python3 plot_tsvc_boxplot.py --results-dir "$PLOT_ROOT" --out "$PLOT_ROOT/tsvc_boxplot.png" || echo "[plot] tsvc_boxplot failed"
    ;;
esac
