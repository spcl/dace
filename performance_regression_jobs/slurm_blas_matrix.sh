#!/bin/bash
#SBATCH --job-name=dace-blas-matrix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=blas_matrix_%j.out
#SBATCH --error=blas_matrix_%j.err
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

spack load gcc@16.1.0
spack load openblas
export OPENBLAS_DIR="$(spack location -i openblas)"
export LD_LIBRARY_PATH="$OPENBLAS_DIR/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"

gcc -O2 -fopenmp -o /dev/shm/iso_omp blas_isolation.c -I"$OPENBLAS_DIR/include" \
    -L"$OPENBLAS_DIR/lib" -Wl,-rpath,"$OPENBLAS_DIR/lib" -lopenblas

echo "===== A1. pure-C gemm, srun -n1 -c72 ====="
for t in 1 72; do
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t srun -n1 -c72 --cpu-bind=verbose,cores /dev/shm/iso_omp gemm 2048
done
echo "===== A2. pure-C gemm, srun -n1 (no -c) ====="
for t in 1 72; do
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t srun -n1 --cpu-bind=verbose,cores /dev/shm/iso_omp gemm 2048
done

echo "===== B. batch-step python + numpy + ctypes dgemm (no DaCe) ====="
for t in 1 72; do
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t python3 - << 'PY'
import ctypes, os, time
import numpy as np
lib = ctypes.CDLL("libopenblas.so")
lib.openblas_get_num_threads.restype = ctypes.c_int
n = 2048
rng = np.random.default_rng(0)
A = np.asfortranarray(rng.random((n, n)))
B = np.asfortranarray(rng.random((n, n)))
C = np.zeros((n, n), order="F")
CblasColMajor, CblasNoTrans = 102, 111
dgemm = lib.cblas_dgemm
dgemm.argtypes = [ctypes.c_int]*3 + [ctypes.c_int]*3 + [ctypes.c_double,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
    ctypes.c_void_p, ctypes.c_int]
def call():
    dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
          A.ctypes.data, n, B.ctypes.data, n, 0.0, C.ctypes.data, n)
call()
best = 1e30
for _ in range(3):
    t0 = time.perf_counter(); call(); best = min(best, time.perf_counter() - t0)
print(f"B threads={os.environ['OMP_NUM_THREADS']} openblas_reports={lib.openblas_get_num_threads()} "
      f"best_ms={best*1e3:.2f} gflops={2*n**3/best/1e9:.1f}")
PY
done

echo "===== C. batch-step python + DaCe canon GEMM kernel (no srun) ====="
python3 - << 'PY'
import os, sys, time, ctypes
sys.path.insert(0, os.getcwd())
import engine
engine.configure_dace_process()
import dace, numpy as np

N = 2048
@dace.program
def gemm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    C[:] = A @ B

sdfg = engine.pipeline_canon(gemm.to_sdfg(simplify=True), device='cpu')
sdfg.name = 'blas_matrix_gemm'
sdfg.save('/dev/shm/blas_matrix_gemm.sdfgz', compress=False)
sdfg.compile()
print("compiled; running children...")
import subprocess
child = r'''
import ctypes, os, sys, time
sys.path.insert(0, %r)
import engine
engine.configure_dace_process()
import dace, numpy as np
sdfg = dace.SDFG.from_file("/dev/shm/blas_matrix_gemm.sdfgz")
csdfg = sdfg.compile()
lib = ctypes.CDLL("libopenblas.so")
lib.openblas_get_num_threads.restype = ctypes.c_int
n = 2048
rng = np.random.default_rng(0)
A = np.asfortranarray(rng.random((n, n)))
B = np.asfortranarray(rng.random((n, n)))
C = np.zeros((n, n), order="F")
csdfg(A=A, B=B, C=C)
best = 1e30
for _ in range(3):
    t0 = time.perf_counter(); csdfg(A=A, B=B, C=C); best = min(best, time.perf_counter() - t0)
print("C threads=%%s openblas_reports=%%d best_ms=%%.2f gflops=%%.1f" %% (
    os.environ["OMP_NUM_THREADS"], lib.openblas_get_num_threads(), best*1e3, 2*n**3/best/1e9))
''' % (os.getcwd(),)
for t in ('1', '72'):
    env = dict(os.environ); env['OMP_NUM_THREADS'] = t; env['OPENBLAS_NUM_THREADS'] = t
    r = subprocess.run(['python3', '-c', child], capture_output=True, text=True, env=env, timeout=300)
    print(r.stdout.strip() or f'FAILED: {r.stderr[-400:]}')
PY
echo MATRIX_DONE
