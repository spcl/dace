#!/usr/bin/env python3
"""Disentangle the 23x "72 threads is SLOWER than 1 thread" catastrophe found by
openblas_canon_check.py. Two independent parallelism sources are suspects:
  (a) an OpenMP region DaCe's frame codegen wraps around the CPU_Multicore-scheduled
      Gemm tasklet (oversubscribing on top of OpenBLAS's own pthread pool), and
  (b) OpenBLAS's internal pthread pool alone (openblas_set_num_threads).
This dumps the generated C++ around the cblas_dgemm call (to see what, if
anything, wraps it), lists every .so DaCe produced (the "stub" naming hints at
more than one), ldd's each, and runs the OMP_NUM_THREADS x OPENBLAS_NUM_THREADS
2x2 grid to see which axis (or their product) causes the blowup.
"""
import ctypes
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

N = 2048


def build_and_canon():
    import dace

    M, K, Nn = N, N, N

    @dace.program
    def gemm(A: dace.float64[M, K], B: dace.float64[K, Nn], C: dace.float64[M, Nn]):
        C[:] = A @ B

    sdfg = gemm.to_sdfg(simplify=True)
    return engine.pipeline_canon(sdfg, device='cpu')


def dump_generated_source(sdfg):
    from dace.codegen import codegen
    print('=' * 70)
    print('1. Generated C++ around the cblas_dgemm call')
    print('=' * 70)
    objects = codegen.generate_code(sdfg)
    for obj in objects:
        if obj.language not in ('cpp',):
            continue
        src = obj.clean_code
        for m in re.finditer(r'cblas_\w*gemm', src):
            lines = src.splitlines()
            # locate the line number of the match
            upto = src[:m.start()].count('\n')
            lo, hi = max(0, upto - 20), min(len(lines), upto + 5)
            print(f'  --- match at generated line {upto + 1} (file {obj.name}) ---')
            for i in range(lo, hi):
                marker = '>>' if i == upto else '  '
                print(f'  {marker}{i + 1:5d}: {lines[i]}')
            print()


def list_and_ldd_sos(build_folder):
    print('=' * 70)
    print('2. Every .so DaCe produced + ldd')
    print('=' * 70)
    sos = []
    for root, _, files in os.walk(build_folder):
        for fn in files:
            if fn.endswith('.so'):
                sos.append(os.path.join(root, fn))
    for so in sorted(sos):
        print(f'  {so}')
        out = subprocess.run(['ldd', so], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if any(k in line.lower() for k in ('blas', 'gomp', 'not found')):
                print(f'      {line.strip()}')
    return sos


def check_dlopen_in_source(sdfg):
    from dace.codegen import codegen
    print()
    print('=' * 70)
    print('3. Does the generated code dlopen anything at runtime?')
    print('=' * 70)
    objects = codegen.generate_code(sdfg)
    any_dlopen = False
    for obj in objects:
        if obj.language != 'cpp':
            continue
        for m in re.finditer(r'dlopen\([^)]*\)', obj.clean_code):
            any_dlopen = True
            print(f'  {obj.name}: {m.group(0)}')
    if not any_dlopen:
        print('  none found')


def scaling_grid(csdfg):
    print()
    print('=' * 70)
    print('4. OMP_NUM_THREADS x OPENBLAS_NUM_THREADS 2x2 grid')
    print('=' * 70)
    import numpy as np
    lib = ctypes.CDLL('libopenblas.so')
    lib.openblas_set_num_threads.argtypes = [ctypes.c_int]

    rng = np.random.default_rng(0)
    A = np.asfortranarray(rng.random((N, N)))
    B = np.asfortranarray(rng.random((N, N)))
    C = np.zeros((N, N), order='F')

    max_threads = max(1, int(os.environ.get('OMP_NUM_THREADS_ORIG', os.environ.get('OMP_NUM_THREADS', '72'))))
    results = {}
    for omp_t in (1, max_threads):
        for ob_t in (1, max_threads):
            os.environ['OMP_NUM_THREADS'] = str(omp_t)
            lib.openblas_set_num_threads(ob_t)
            os.environ['OPENBLAS_NUM_THREADS'] = str(ob_t)
            csdfg(A=A, B=B, C=C)  # warmup
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                csdfg(A=A, B=B, C=C)
                samples.append(time.perf_counter() - t0)
            samples.sort()
            median = samples[len(samples) // 2]
            results[(omp_t, ob_t)] = median
            gflops = (2 * N ** 3 / 1e9) / median
            print(f'  OMP_NUM_THREADS={omp_t:3d}  openblas_threads={ob_t:3d}: '
                  f'median={median * 1000:8.2f} ms ({gflops:6.1f} GFLOP/s)')
    return results


def main():
    engine.configure_dace_process()
    os.environ['OMP_NUM_THREADS_ORIG'] = os.environ.get('OMP_NUM_THREADS', '72')
    sdfg = build_and_canon()
    dump_generated_source(sdfg)

    sdfg.name = 'openblas_source_check_gemm'
    t0 = time.perf_counter()
    csdfg = sdfg.compile()
    print(f'compiled in {time.perf_counter() - t0:.1f}s\n')

    list_and_ldd_sos(sdfg.build_folder)
    check_dlopen_in_source(sdfg)
    results = scaling_grid(csdfg)

    print()
    print('=' * 70)
    print('VERDICT')
    print('=' * 70)
    max_threads = int(os.environ['OMP_NUM_THREADS_ORIG'])
    baseline = results.get((1, 1))
    for key in sorted(results):
        omp_t, ob_t = key
        ratio = results[key] / baseline if baseline else float('nan')
        print(f'  OMP={omp_t:3d} OB={ob_t:3d}: {ratio:.2f}x vs (1,1) baseline')
    worst = max(results, key=lambda k: results[k])
    print(f'\n  Slowest cell: OMP={worst[0]} OB={worst[1]} ({results[worst] * 1000:.1f} ms)')
    if worst == (max_threads, max_threads) and results[worst] > baseline * 3:
        print('  -> DOUBLE-PARALLELISM OVERSUBSCRIPTION: outer OMP region x OpenBLAS pthreads both active.')
    elif worst[0] == max_threads and results[worst] > baseline * 3:
        print('  -> The OUTER OMP_NUM_THREADS axis alone causes the blowup (DaCe-emitted wrapper).')
    elif worst[1] == max_threads and results[worst] > baseline * 3:
        print('  -> The OpenBLAS-internal thread axis alone causes the blowup.')


if __name__ == '__main__':
    main()
