#!/usr/bin/env python3
"""Cell C of the BLAS discriminator matrix (standalone file -- @dace.program needs
real source, not a heredoc): the canon-pipeline DaCe GEMM kernel in the BATCH step
(no srun), 1 vs 72 threads in fresh children. A1/A2/B all scale (pure-C under srun,
python+numpy+ctypes); if C alone collapses, the fault is in DaCe's compile/load path."""
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

N = 2048
CACHE = '/dev/shm/blas_matrix_gemm.sdfgz'

CHILD = r'''
import ctypes, os, sys, time
sys.path.insert(0, {perf_dir!r})
import engine
engine.configure_dace_process()
import dace, numpy as np
sdfg = dace.SDFG.from_file({cache!r})
csdfg = sdfg.compile()
lib = ctypes.CDLL("libopenblas.so")
lib.openblas_get_num_threads.restype = ctypes.c_int
n = {n}
rng = np.random.default_rng(0)
A = np.asfortranarray(rng.random((n, n)))
B = np.asfortranarray(rng.random((n, n)))
C = np.zeros((n, n), order="F")
csdfg(A=A, B=B, C=C)
best = 1e30
for _ in range(3):
    t0 = time.perf_counter(); csdfg(A=A, B=B, C=C); best = min(best, time.perf_counter() - t0)
print("C threads=%s openblas_reports=%d best_ms=%.2f gflops=%.1f" % (
    os.environ["OMP_NUM_THREADS"], lib.openblas_get_num_threads(), best * 1e3, 2 * n ** 3 / best / 1e9))
'''


def main():
    engine.configure_dace_process()
    import dace

    @dace.program
    def gemm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        C[:] = A @ B

    sdfg = engine.pipeline_canon(gemm.to_sdfg(simplify=True), device='cpu')
    sdfg.name = 'blas_matrix_gemm'
    sdfg.save(CACHE, compress=False)
    sdfg.compile()
    print('compiled; running children...', flush=True)
    child = CHILD.format(perf_dir=os.path.dirname(os.path.abspath(__file__)), cache=CACHE, n=N)
    for t in ('1', '72'):
        env = dict(os.environ)
        env['OMP_NUM_THREADS'] = t
        env['OPENBLAS_NUM_THREADS'] = t
        engine.restore_cpu_affinity()  # the fix under test: un-poison the inherited mask
        r = subprocess.run(['python3', '-c', child], capture_output=True, text=True, env=env, timeout=300)
        print(r.stdout.strip() or f'FAILED: {r.stderr[-500:]}', flush=True)
    # Also once IN-PROCESS (no child) at 72 threads -- discriminates child-spawn vs in-process.
    import numpy as np
    os.environ['OMP_NUM_THREADS'] = '72'
    lib = ctypes.CDLL('libopenblas.so')
    lib.openblas_get_num_threads.restype = ctypes.c_int
    csdfg = sdfg.compile()
    rng = np.random.default_rng(0)
    A = np.asfortranarray(rng.random((N, N)))
    B = np.asfortranarray(rng.random((N, N)))
    C = np.zeros((N, N), order='F')
    csdfg(A=A, B=B, C=C)
    best = 1e30
    for _ in range(3):
        t0 = time.perf_counter()
        csdfg(A=A, B=B, C=C)
        best = min(best, time.perf_counter() - t0)
    print(f'C-inproc threads=72 openblas_reports={lib.openblas_get_num_threads()} '
          f'best_ms={best * 1e3:.2f} gflops={2 * N ** 3 / best / 1e9:.1f}')


if __name__ == '__main__':
    main()
