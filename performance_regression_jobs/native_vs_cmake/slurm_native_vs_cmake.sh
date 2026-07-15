#!/bin/bash
#SBATCH --job-name=dace-native-vs-cmake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks on 1 node -- one per Grace CPU (GH200 = 4 Grace sockets)
#SBATCH --cpus-per-task=72       # 72 cores/socket -> 4 x 72 = 288 = whole node
#SBATCH --hint=nomultithread     # Neoverse-V2: 1 thread/core
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=native_vs_cmake_%j.out
#SBATCH --error=native_vs_cmake_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# compiler.build_mode = cmake (BEFORE) vs native (AFTER): DaCe codegen + C++ compile
# wall + runtime over NPBench+PolyBench at the paper preset, single- and multi-core,
# distributed over nodes * ntasks-per-node ranks (default 1 x 4). Kernels self-
# partition by SLURM_PROCID/SLURM_NTASKS; the trailing --tables-only pass is the
# cross-rank aggregation that writes native_vs_cmake.md.
#
# native mode needs nvcc + g++ on PATH exactly like the CMake path; the toolchain /
# library environment below is the same one slurm_npbench_polybench_compile.sh sets
# (see that script for the rationale on each export). Adjust the SBATCH header for
# your account / partition / node count.
set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72"       # multi-core lane thread count (single-core lane forces 1)
export OMP_PROC_BIND="close"
export OMP_PLACES="cores"
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate

spack load gcc@16.1.0
spack load llvm@22.1.5
spack load cmake            # the BEFORE lane still runs a real CMake configure+build
spack load openblas
spack load cuda

# nvcc discovery for native mode (CUDA_HOME) + runtime lib paths (spack load sets PATH only).
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

REPS="${REPS:-10}"
COMPILE_REPS="${COMPILE_REPS:-3}"

# --distribution=block:block gives rank i a contiguous socket; --cpu-bind=cores pins it.
srun --cpu-bind=verbose,cores --distribution=block:block \
    python3 native_vs_cmake/native_vs_cmake_compile.py --compile-reps "$COMPILE_REPS" --run-reps "$REPS"

# cross-rank aggregation (re-scans every rank's rows).
python3 native_vs_cmake/native_vs_cmake_compile.py --tables-only
