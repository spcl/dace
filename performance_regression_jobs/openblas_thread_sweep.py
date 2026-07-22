#!/usr/bin/env python3
"""Bisect the OpenBLAS pthreads thread-count collapse found by openblas_source_check.py:
2048x2048 DGEMM is 24x SLOWER at 72 threads than at 1 (OMP_NUM_THREADS has zero effect;
purely an openblas_set_num_threads() axis). Sweeps 1,2,4,8,16,32,48,64,72 to find whether
this is a sudden cliff or a monotonic collapse, and whether it's a fixed problem size
artifact by also trying N=512/4096.
"""
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

THREAD_COUNTS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 72)
SIZES = (512, 2048)


def build_and_canon(n):
    import dace

    @dace.program
    def gemm(A: dace.float64[n, n], B: dace.float64[n, n], C: dace.float64[n, n]):
        C[:] = A @ B

    sdfg = gemm.to_sdfg(simplify=True)
    sdfg = engine.pipeline_canon(sdfg, device='cpu')
    sdfg.name = f'openblas_sweep_gemm_{n}'
    return sdfg


def main():
    engine.configure_dace_process()
    import numpy as np
    lib = ctypes.CDLL('libopenblas.so')
    lib.openblas_set_num_threads.argtypes = [ctypes.c_int]

    for n in SIZES:
        print('=' * 70)
        print(f'N={n}')
        print('=' * 70)
        sdfg = build_and_canon(n)
        csdfg = sdfg.compile()
        rng = np.random.default_rng(0)
        A = np.asfortranarray(rng.random((n, n)))
        B = np.asfortranarray(rng.random((n, n)))
        C = np.zeros((n, n), order='F')

        baseline = None
        for threads in THREAD_COUNTS:
            lib.openblas_set_num_threads(threads)
            csdfg(A=A, B=B, C=C)  # warmup
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                csdfg(A=A, B=B, C=C)
                samples.append(time.perf_counter() - t0)
            samples.sort()
            median = samples[len(samples) // 2]
            if baseline is None:
                baseline = median
            gflops = (2 * n ** 3 / 1e9) / median
            print(f'  threads={threads:3d}: median={median * 1000:9.2f} ms ({gflops:7.1f} GFLOP/s) '
                  f'{median / baseline:6.2f}x vs 1-thread')
        print()


if __name__ == '__main__':
    main()
