#!/usr/bin/env python3
"""Final verification of the OpenBLAS threads=openmp switch (job scripts now load it).

The pthreads build collapsed 24x at 72 threads on this box (openblas_source_check.py:
NO_AFFINITY spin-wait pool, unpinned). The new threads=openmp build (spack hash
rsaxs76) runs BLAS threading on libgomp -- same runtime as DaCe's own omp regions,
honoring OMP_PROC_BIND/OMP_PLACES pinning and OMP_MAX_ACTIVE_LEVELS=1 (nested BLAS
calls inside a DaCe parallel region serialize instead of forking a second team).

Checks:
  1. The lib the job env resolves is the OPENMP build (get_parallel()==2, USE_OPENMP).
  2. The canon-pipeline GEMM kernel links/loads that same lib.
  3. Fresh-process thread sweep (env set before interpreter start): DGEMM must now
     actually SCALE with OMP_NUM_THREADS instead of collapsing.
Each child prints its own env + what OpenBLAS reports, closing the earlier
"child reported 1 thread" ambiguity.
"""
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

N = 2048
THREAD_COUNTS = (1, 8, 32, 72)
CACHE_SDFGZ = '/dev/shm/openblas_openmp_verify_gemm.sdfgz'

CHILD_SCRIPT = r'''
import ctypes, os, sys, time
sys.path.insert(0, {perf_dir!r})
import engine
engine.configure_dace_process()
import dace, numpy as np
sdfg = dace.SDFG.from_file({cache_path!r})
csdfg = sdfg.compile()
lib = ctypes.CDLL("libopenblas.so")
lib.openblas_get_num_threads.restype = ctypes.c_int
lib.openblas_get_parallel.restype = ctypes.c_int
resolved = [l.strip().split()[-1] for l in open("/proc/self/maps") if "libopenblas" in l]
n = {n}
rng = np.random.default_rng(0)
A = np.asfortranarray(rng.random((n, n)))
B = np.asfortranarray(rng.random((n, n)))
C = np.zeros((n, n), order="F")
csdfg(A=A, B=B, C=C)  # warmup
samples = []
for _ in range(3):
    t0 = time.perf_counter()
    csdfg(A=A, B=B, C=C)
    samples.append(time.perf_counter() - t0)
samples.sort()
print("ENVREPORT", os.environ.get("OMP_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS"),
      os.environ.get("OMP_MAX_ACTIVE_LEVELS"), lib.openblas_get_parallel(), sorted(set(resolved))[0])
print("RESULT", {threads}, lib.openblas_get_num_threads(), samples[len(samples)//2])
'''


def main():
    engine.configure_dace_process()
    print('=' * 70)
    print('1. Which OpenBLAS does this job env resolve?')
    print('=' * 70)
    lib = ctypes.CDLL('libopenblas.so')
    lib.openblas_get_parallel.restype = ctypes.c_int
    lib.openblas_get_config.restype = ctypes.c_char_p
    mode = lib.openblas_get_parallel()
    print(f'  openblas_get_parallel() = {mode} (0=serial 1=pthreads 2=OpenMP)')
    print(f'  openblas_get_config() = {lib.openblas_get_config().decode()}')
    resolved = sorted({l.strip().split()[-1] for l in open('/proc/self/maps') if 'libopenblas' in l})
    print(f'  resolved: {resolved}')

    print()
    print('=' * 70)
    print(f'2. canon-pipeline {N}x{N} DGEMM: compile + linkage')
    print('=' * 70)
    import dace

    @dace.program
    def gemm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        C[:] = A @ B

    sdfg = gemm.to_sdfg(simplify=True)
    sdfg = engine.pipeline_canon(sdfg, device='cpu')
    sdfg.name = 'openblas_openmp_verify_gemm'
    sdfg.save(CACHE_SDFGZ, compress=False)
    t0 = time.perf_counter()
    sdfg.compile()
    print(f'  compiled in {time.perf_counter() - t0:.1f}s')
    for root, _, files in os.walk(sdfg.build_folder):
        for fn in files:
            if fn.endswith('.so') and not fn.startswith('libdacestub'):
                so = os.path.join(root, fn)
                out = subprocess.run(['ldd', so], capture_output=True, text=True, timeout=30).stdout
                for line in out.splitlines():
                    if 'openblas' in line.lower():
                        print(f'  {os.path.basename(so)}: {line.strip()}')

    print()
    print('=' * 70)
    print('3. Fresh-process OMP_NUM_THREADS sweep (must SCALE now)')
    print('=' * 70)
    perf_dir = os.path.dirname(os.path.abspath(__file__))
    baseline = None
    results = {}
    for threads in THREAD_COUNTS:
        env = dict(os.environ)
        env['OMP_NUM_THREADS'] = str(threads)
        env['OPENBLAS_NUM_THREADS'] = str(threads)
        script = CHILD_SCRIPT.format(perf_dir=perf_dir, cache_path=CACHE_SDFGZ, n=N, threads=threads)
        proc = subprocess.run(['python3', '-c', script], capture_output=True, text=True, env=env, timeout=180)
        median = actual = envline = None
        for line in proc.stdout.splitlines():
            if line.startswith('ENVREPORT'):
                envline = line
            if line.startswith('RESULT'):
                _, t, actual, median = line.split()
                actual, median = int(actual), float(median)
        if median is None:
            print(f'  threads={threads}: CHILD FAILED. stdout={proc.stdout[-300:]!r} stderr={proc.stderr[-600:]!r}')
            continue
        if baseline is None:
            baseline = median
        results[threads] = median
        gflops = (2 * N ** 3 / 1e9) / median
        print(f'  OMP_NUM_THREADS={threads:3d} (openblas reports {actual:3d}): median={median * 1000:9.2f} ms '
              f'({gflops:7.1f} GFLOP/s) speedup vs 1-thread: {baseline / median:6.2f}x')
        print(f'    child env/introspection: {envline}')

    print()
    print('=' * 70)
    print('VERDICT')
    print('=' * 70)
    scaled = results and 1 in results and 72 in results and results[1] / results[72] > 4
    print(f'  OpenMP-threaded build loaded: {mode == 2}')
    print(f'  1->72 thread speedup: {results[1] / results[72]:.2f}x' if scaled or (1 in results and 72 in results)
          else '  (sweep incomplete)')
    print(f'  MULTITHREADED OPENBLAS WORKING: {"YES" if (mode == 2 and scaled) else "NO"}')
    return 0 if (mode == 2 and scaled) else 2


if __name__ == '__main__':
    sys.exit(main())
