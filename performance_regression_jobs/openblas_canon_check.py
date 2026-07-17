#!/usr/bin/env python3
"""Verify the ACTUAL perf-job pipeline (``engine.pipeline_canon``) lifts a GEMM into
a BLAS library node with the OpenBLAS implementation selected -- not a synthetic
standalone check. Supersedes the earlier openblas_threading_check.py finding: that
script compiled a raw ``@dace.program`` via plain ``sdfg.compile()``, which never
calls canonicalize/finalize -- so it always got ``library.blas.default_implementation
= 'pure'`` (the DaCe global default) regardless of what OpenBLAS is actually
installed. The real npbench_polybench/canon_vs harness always routes through
``engine.pipeline_canon`` (or ``pipeline_parallel`` via ``set_fastest_library_impls``)
before compiling, per engine.py's own comment: "Without this, library nodes stay
implementation=None and a GEMM lowers to a naive `pure` triple loop (e.g. mlp
12,443ms vs ~200ms)."

Checks, in order:
  1. Build a plain GEMM SDFG (implementation=None, the frontend default).
  2. Run it through ``engine.pipeline_canon`` -- the SAME call the canon lane
     makes -- and inspect every LibraryNode's ``.implementation``.
  3. Compile the FINALIZED sdfg, ldd the .so for openblas linkage.
  4. Correct thread-scaling test: ``openblas_set_num_threads()`` via ctypes on
     the ALREADY-loaded library (env-var-after-load is a no-op for a pthreads
     pool sized at first call -- the bug in the first diagnostic run).
"""
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine  # noqa: E402

N = 2048


def build_gemm_sdfg():
    import dace

    M, K, Nn = N, N, N

    @dace.program
    def gemm(A: dace.float64[M, K], B: dace.float64[K, Nn], C: dace.float64[M, Nn]):
        C[:] = A @ B

    return gemm.to_sdfg(simplify=True)


def report_libnodes(sdfg, label):
    from dace.sdfg import nodes as sdnodes
    print(f'  LibraryNodes after {label}:')
    found = False
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, sdnodes.LibraryNode):
            found = True
            print(f'    {type(node).__name__} "{node.label}" implementation={node.implementation!r} '
                  f'schedule={node.schedule}')
    if not found:
        print('    (none -- already expanded or fused away)')
    return found


def main():
    engine.configure_dace_process()
    print('=' * 70)
    print('1. Plain GEMM SDFG (frontend default, no canon/finalize)')
    print('=' * 70)
    sdfg = build_gemm_sdfg()
    report_libnodes(sdfg, 'to_sdfg(simplify=True)')

    print()
    print('=' * 70)
    print('2. engine.pipeline_canon(sdfg) -- the ACTUAL canon-lane pipeline')
    print('=' * 70)
    sdfg = engine.pipeline_canon(sdfg, device='cpu')
    has_libnode = report_libnodes(sdfg, 'pipeline_canon')

    gemm_impls = []
    from dace.sdfg import nodes as sdnodes
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, sdnodes.LibraryNode) and 'gemm' in type(node).__name__.lower() or (
                isinstance(node, sdnodes.LibraryNode) and 'matmul' in type(node).__name__.lower()):
            gemm_impls.append(node.implementation)
    print(f'\n  GEMM/MatMul libnode implementation(s): {gemm_impls}')

    print()
    print('=' * 70)
    print('3. Compile the FINALIZED sdfg; check OpenBLAS linkage')
    print('=' * 70)
    sdfg.name = 'openblas_canon_check_gemm'
    t0 = time.perf_counter()
    csdfg = sdfg.compile()
    print(f'  compiled in {time.perf_counter() - t0:.1f}s')

    so_path = None
    for root, _, files in os.walk(sdfg.build_folder):
        for fn in files:
            if fn.startswith('lib') and fn.endswith('.so'):
                so_path = os.path.join(root, fn)
    print(f'  .so = {so_path}')
    if so_path:
        out = subprocess.run(['ldd', so_path], capture_output=True, text=True, timeout=30).stdout
        print('  ldd (full):')
        for line in out.splitlines():
            print(f'    {line.strip()}')
        blas_linked = any('blas' in l.lower() for l in out.splitlines())
        print(f'\n  Links against a BLAS library: {blas_linked}')

    print()
    print('=' * 70)
    print('4. Correct thread-scaling test (openblas_set_num_threads via ctypes)')
    print('=' * 70)
    import numpy as np
    lib = ctypes.CDLL('libopenblas.so')
    try:
        lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
    except AttributeError:
        print('  openblas_set_num_threads not exported -- cannot control thread count directly')
        lib = None

    rng = np.random.default_rng(0)
    A = np.asfortranarray(rng.random((N, N)))
    B = np.asfortranarray(rng.random((N, N)))
    C = np.zeros((N, N), order='F')

    reps = 5
    results = {}
    max_threads = max(1, int(os.environ.get('OMP_NUM_THREADS', '72')))
    for threads in (1, max_threads):
        if lib is not None:
            lib.openblas_set_num_threads(threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
        csdfg(A=A, B=B, C=C)  # warmup
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            csdfg(A=A, B=B, C=C)
            samples.append(time.perf_counter() - t0)
        samples.sort()
        median = samples[len(samples) // 2]
        results[threads] = median
        gflops = (2 * N ** 3 / 1e9) / median
        print(f'  threads={threads:3d}: median={median * 1000:.2f} ms ({gflops:.1f} GFLOP/s)')

    lo, hi = 1, max_threads
    if results.get(lo) and results.get(hi):
        speedup = results[lo] / results[hi]
        print(f'\n  {N}x{N} DGEMM speedup {lo}->{hi} threads: {speedup:.2f}x '
              f'(ideal ~{hi}x, DGEMM realistically 0.3-0.7x of ideal)')

    print()
    print('=' * 70)
    print('VERDICT')
    print('=' * 70)
    ok = bool(has_libnode) and any(i == 'OpenBLAS' for i in gemm_impls) and results.get(1, 1) / max(
        results.get(max_threads, 1), 1e-9) > 1.5
    print(f'  BLAS libnode present after canonicalize: {has_libnode}')
    print(f'  GEMM implementation selected: {gemm_impls}')
    print(f'  CANON PIPELINE CORRECTLY LIFTS GEMM TO FAST OPENBLAS: {"YES" if ok else "NO -- see checks above"}')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
