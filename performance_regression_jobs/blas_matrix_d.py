#!/usr/bin/env python3
"""Cell D of the BLAS discriminator matrix: WHAT in the DaCe child poisons the
72-thread dgemm? (C-child collapses at 1.8 GFLOP/s; the identical child minus
dace imports [cell B] hits 2166 GFLOP/s; the DaCe kernel itself scales perfectly
at 4 threads in-process.) Variants, all fresh children at OMP_NUM_THREADS=72:
  d1  baseline repro (= C-child) + OMP_DISPLAY_ENV=verbose so libgomp prints the
      ICVs it ACTUALLY adopted (stderr) -- any places/bind pathology shows here.
  d2  identical loads incl. kernel warmup, but TIME a direct ctypes cblas_dgemm
      instead of the csdfg call -- separates "loading dace/kernel poisons the
      process" from "the CompiledSDFG invocation path is the problem".
  d3  d1 without engine.configure_dace_process().
  d4  d1 with OMP_PLACES/OMP_PROC_BIND removed from the child env.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

N = 2048
CACHE = '/dev/shm/blas_matrix_gemm_d.sdfgz'

CHILD = r'''
import ctypes, os, sys, time
sys.path.insert(0, {perf_dir!r})
variant = {variant!r}
if variant != "d3":
    import engine
    engine.configure_dace_process()
else:
    import engine  # import only (module-level setdefaults) -- no configure
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
csdfg(A=A, B=B, C=C)  # warmup ALWAYS via the kernel (loads + first-call init)
if variant == "d2":
    CblasColMajor, CblasNoTrans = 102, 111
    dgemm = lib.cblas_dgemm
    dgemm.argtypes = [ctypes.c_int]*3 + [ctypes.c_int]*3 + [ctypes.c_double,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_int]
    def call():
        dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
              A.ctypes.data, n, B.ctypes.data, n, 0.0, C.ctypes.data, n)
else:
    def call():
        csdfg(A=A, B=B, C=C)
call()
best = 1e30
for _ in range(3):
    t0 = time.perf_counter(); call(); best = min(best, time.perf_counter() - t0)
print("%s openblas_reports=%d best_ms=%.2f gflops=%.1f" % (
    variant, lib.openblas_get_num_threads(), best * 1e3, 2 * n ** 3 / best / 1e9))
'''


def main():
    engine.configure_dace_process()
    import dace

    @dace.program
    def gemm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        C[:] = A @ B

    sdfg = engine.pipeline_canon(gemm.to_sdfg(simplify=True), device='cpu')
    sdfg.name = 'blas_matrix_gemm_d'
    sdfg.save(CACHE, compress=False)
    sdfg.compile()
    print('compiled; running variant children at OMP_NUM_THREADS=72...', flush=True)

    perf_dir = os.path.dirname(os.path.abspath(__file__))
    for variant in ('d1', 'd2', 'd3', 'd4'):
        env = dict(os.environ)
        env['OMP_NUM_THREADS'] = '72'
        env['OPENBLAS_NUM_THREADS'] = '72'
        if variant == 'd1':
            env['OMP_DISPLAY_ENV'] = 'verbose'
        if variant == 'd4':
            env.pop('OMP_PLACES', None)
            env.pop('OMP_PROC_BIND', None)
        script = CHILD.format(perf_dir=perf_dir, cache=CACHE, n=N, variant=variant)
        r = subprocess.run(['python3', '-c', script], capture_output=True, text=True, env=env, timeout=300)
        print(r.stdout.strip() or f'{variant} FAILED: {r.stderr[-400:]}', flush=True)
        if variant == 'd1':
            icvs = [l for l in r.stderr.splitlines()
                    if any(k in l for k in ('OMP_', 'OPENMP DISPLAY', 'GOMP'))]
            print('  d1 libgomp ICVs:', flush=True)
            for l in icvs[:25]:
                print(f'    {l}', flush=True)


if __name__ == '__main__':
    main()
