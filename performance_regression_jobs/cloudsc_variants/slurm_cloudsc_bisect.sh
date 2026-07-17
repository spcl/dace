#!/bin/bash
#SBATCH --job-name=dace-cloudsc-bisect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00          # chain alone is ~55 min; snapshots add little
#SBATCH --partition=normal       # NOT a perf measurement -- SDFG-prep, so the
#SBATCH --account=g34            # 1.5h-normal-job exception applies (same as cache build)
#SBATCH --output=cloudsc_bisect_%j.out
#SBATCH --error=cloudsc_bisect_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Chain-bisect phase A: rerun the cloudsc parallelize chain saving a per-stage
# .sdfgz snapshot (cloudsc_variants/cache/bisect/). Analysis that regenerates
# SDFG phases always runs as a batch job; the downstream codegen syntax checks
# (cloudsc_bisect_check.py) only LOAD these snapshots and run on the login node.
#
# Env block identical to slurm_cloudsc_cache.sh.
set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export PYTHONUNBUFFERED=1
export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
export PYTHONPATH=/capstor/scratch/cscs/ybudanaz/aarch64/dace:${PYTHONPATH:-}

# gcc@16.1.0 +graphite: its g++ gives clang a modern libstdc++ (--gcc-install-dir), gfortran,
# AND the Graphite -floop-parallelize-all the native-gcc-autopar lane needs.
spack load gcc@16.1.0
spack load llvm@22.1.5    # clang++ = DaCe codegen compiler + Polly for the native-clang-polly-autopar lane
spack load cmake
spack load openblas
spack load cuda

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

srun --cpu-bind=verbose,cores python3 cloudsc_variants/cloudsc_bisect_chain.py
