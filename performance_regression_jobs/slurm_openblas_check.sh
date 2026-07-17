#!/bin/bash
#SBATCH --job-name=dace-openblas-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=openblas_check_%j.out
#SBATCH --error=openblas_check_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Diagnose whether the OpenBLAS DaCe links/loads is the multithreaded build and
# whether that threading actually engages at runtime -- see
# openblas_threading_check.py's docstring. Two spack openblas installs exist
# (threads=pthreads and threads=none); this checks which one a real
# DaCe-compiled GEMM kernel actually resolves and whether it scales.
set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72"
export OMP_PROC_BIND="close"
export OMP_PLACES="cores"
export OMP_MAX_ACTIVE_LEVELS=1     # one parallel level only: a BLAS call inside a DaCe omp region must serialize, not fork its own team (openmp-threaded OpenBLAS honors this; the old pthreads pool could not)
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
export PYTHONPATH=/capstor/scratch/cscs/ybudanaz/aarch64/dace:${PYTHONPATH:-}

spack load gcc@16.1.0
spack load openblas threads=openmp

export OPENBLAS_DIR="$(spack location -i openblas threads=openmp 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
if [ -n "$OPENBLAS_DIR" ]; then
    for _d in "$OPENBLAS_DIR"/lib "$OPENBLAS_DIR"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
    done
    [ -d "$OPENBLAS_DIR/include" ] && export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"
fi

srun --cpu-bind=verbose,cores python3 openblas_openmp_verify.py
