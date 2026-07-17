#!/bin/bash
#SBATCH --job-name=dace-cloudsc-variants
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # ONE rank: a single kernel, 4 variant cells run sequentially
#SBATCH --cpus-per-task=72       # one full Grace socket for the OpenMP runtime reps
#SBATCH --hint=nomultithread     # Neoverse-V2: 1 thread/core
#SBATCH --time=00:30:00          # performance ALWAYS on debug, <=30 min
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=cloudsc_variants_%j.out
#SBATCH --error=cloudsc_variants_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# CloudSC under the four DaCe build variants -- {build_mode: cmake, native} x
# {cpu.implementation: legacy, experimental_readable} -- codegen + compile wall + runtime
# (median) + correctness vs a sequential reference. The expensive SDFG preparation (parse +
# simplify + ShortLoopUnroll + LoopToMap chain, MINUTES) is Phase A, cached as
# cloudsc_variants/cache/cloudsc_parallel.sdfgz and only rebuilt when missing.
#
# PRE-BUILD THE CACHE ON THE LOGIN NODE FIRST (transform/compile work is allowed there):
#     python3 cloudsc_variants/cloudsc_variants_perf.py --build-cache-only
# A cold Phase A inside this 30-min debug window can blow the whole allocation.
#
# Toolchain env block identical to codegen_variants/slurm_codegen_variants.sh (see that
# script + slurm_npbench_polybench_compile.sh for the rationale on each export).
set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72" OPENBLAS_NUM_THREADS="72"
export OMP_PROC_BIND="close"
export OMP_PLACES="cores"
export OMP_MAX_ACTIVE_LEVELS=1     # one parallel level only: a BLAS call inside a DaCe omp region must serialize, not fork its own team (openmp-threaded OpenBLAS honors this; the old pthreads pool could not)
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
export PYTHONPATH=/capstor/scratch/cscs/ybudanaz/aarch64/dace:${PYTHONPATH:-}

spack load llvm@22.1.5    # clang++ = DaCe codegen compiler + Polly for the native-clang-polly-autopar lane
spack load cmake            # the cmake variants still run a real CMake configure+build
spack load openblas
spack load cuda

# nvcc discovery (CUDA_HOME) + runtime lib paths (spack load sets PATH only).
export CUDA_HOME="$(spack location -i cuda 2>/dev/null || echo "${CUDA_HOME:-}")"
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
fi
export OPENBLAS_DIR="$(spack location -i openblas 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
if [ -n "$OPENBLAS_DIR" ]; then
    for _d in "$OPENBLAS_DIR"/lib "$OPENBLAS_DIR"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
    done
    [ -d "$OPENBLAS_DIR/include" ] && export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"
fi

RUN_REPS="${RUN_REPS:-25}"
COMPILE_REPS="${COMPILE_REPS:-1}"

# Phase A gate: the perf script auto-builds a missing cache, but that belongs on the LOGIN
# node -- warn loudly if this job is about to pay it inside the debug window.
CACHE=cloudsc_variants/cache/cloudsc_parallel.sdfgz
if [ ! -f "$CACHE" ]; then
    echo "WARNING: $CACHE missing -- Phase A (parse + unroll + LoopToMap chain) will run"
    echo "inside this 30-min debug job and may exceed the window. Pre-build it on the login"
    echo "node first:  python3 cloudsc_variants/cloudsc_variants_perf.py --build-cache-only"
fi

srun --cpu-bind=verbose,cores --distribution=block:block \
    python3 cloudsc_variants/cloudsc_variants_perf.py --compile-reps "$COMPILE_REPS" --run-reps "$RUN_REPS"

python3 cloudsc_variants/cloudsc_variants_perf.py --tables-only
