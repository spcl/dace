#!/bin/bash
#SBATCH --job-name=dace-blas-isolation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=blas_isolation_%j.out
#SBATCH --error=blas_isolation_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Layer isolation for the OpenBLAS 72-thread collapse (see blas_isolation.c):
#   A. pure-C cblas_dgemm vs spack openmp build, OMP_NUM_THREADS swept
#   B. pure-C cblas_dgemm vs spack pthreads build, OPENBLAS_NUM_THREADS swept
#   C. pure-C '#pragma omp parallel for' triad control (libgomp health)
#   D. numpy A@B (venv-bundled scipy-openblas) as an independent BLAS
# Each measurement runs DIRECTLY in the batch step (full 72-core cgroup): an inner
# `srun -n1` would NOT inherit --cpus-per-task on modern slurm and would squeeze
# every OMP thread onto one core -- poisoning exactly what this job measures.
set -eu
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_PROC_BIND="close"
export OMP_PLACES="cores"
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate


OMP_BLAS=$(spack location -i openblas)
PTH_BLAS=$(spack location -i openblas threads=pthreads)

gcc -O2 -fopenmp -o /dev/shm/iso_omp blas_isolation.c -I"$OMP_BLAS/include" -L"$OMP_BLAS/lib" \
    -Wl,-rpath,"$OMP_BLAS/lib" -lopenblas
gcc -O2 -fopenmp -o /dev/shm/iso_pth blas_isolation.c -I"$PTH_BLAS/include" -L"$PTH_BLAS/lib" \
    -Wl,-rpath,"$PTH_BLAS/lib" -lopenblas
echo "built. omp blas=$OMP_BLAS pth blas=$PTH_BLAS"

echo "===== C. triad control (libgomp only) ====="
for t in 1 8 32 72; do
    OMP_NUM_THREADS=$t /dev/shm/iso_omp triad 2>/dev/null
done

echo "===== A. pure-C dgemm, OPENMP-threaded OpenBLAS ====="
for t in 1 8 32 72; do
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t /dev/shm/iso_omp gemm 2048 2>/dev/null
done

echo "===== B. pure-C dgemm, PTHREADS OpenBLAS ====="
for t in 1 8 32 72; do
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t /dev/shm/iso_pth gemm 2048 2>/dev/null
done

echo "===== D. numpy A@B (venv-bundled BLAS, independent of spack) ====="
python3 -c "import numpy; print('numpy', numpy.__version__); numpy.__config__.show()" 2>/dev/null | head -20
for t in 1 8 32 72; do
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t python3 - << 'PY'
import numpy as np, os, time
n = 2048
rng = np.random.default_rng(0)
A, B = rng.random((n, n)), rng.random((n, n))
C = A @ B  # warmup
best = 1e30
for _ in range(3):
    t0 = time.perf_counter(); C = A @ B; dt = time.perf_counter() - t0
    best = min(best, dt)
print(f"NUMPY threads={os.environ['OPENBLAS_NUM_THREADS']} best_ms={best*1e3:.2f} "
      f"gflops={2*n**3/best/1e9:.1f}")
PY
done
echo ISOLATION_DONE
