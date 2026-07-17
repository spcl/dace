#!/bin/bash
#SBATCH --job-name=dace-blas-matrix-c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=blas_matrix_c_%j.out
#SBATCH --error=blas_matrix_c_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Discriminator matrix for: pure-C dgemm (openmp OpenBLAS) SCALES 59x in the batch
# step, yet the same library inside the python/DaCe verify (srun-launched) collapsed
# linearly. Cells:
#   A1 pure-C gemm under `srun -n1 -c72 --cpu-bind=cores`   (srun step, explicit CPUs)
#   A2 pure-C gemm under `srun -n1 --cpu-bind=cores`        (srun step, NO -c: modern
#      slurm gives 1 CPU -- if this collapses, any srun-launched python inherits it)
#   B  batch-step python: numpy imported, spack openblas via ctypes, dgemm on numpy
#      buffers (python-process composition WITHOUT DaCe)
#   C  batch-step python: the canon-pipeline DaCe GEMM kernel (the exact earlier
#      repro, minus srun)
set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_PROC_BIND="close"
export OMP_PLACES="cores"
export OMP_MAX_ACTIVE_LEVELS=1
export PYTHONUNBUFFERED=1
export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
export PYTHONPATH=/capstor/scratch/cscs/ybudanaz/aarch64/dace:${PYTHONPATH:-}

spack load openblas
export OPENBLAS_DIR="$(spack location -i openblas)"
export LD_LIBRARY_PATH="$OPENBLAS_DIR/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"

gcc -O2 -fopenmp -o /dev/shm/iso_omp blas_isolation.c -I"$OPENBLAS_DIR/include" \
    -L"$OPENBLAS_DIR/lib" -Wl,-rpath,"$OPENBLAS_DIR/lib" -lopenblas

python3 blas_matrix_c.py
echo MATRIX_C_DONE
