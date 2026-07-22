#!/usr/bin/env python3
"""Diagnose whether the OpenBLAS DaCe actually links/loads at runtime is the
MULTITHREADED build, and whether that threading actually engages.

Motivation: two OpenBLAS installs exist in this spack tree --
``threads=pthreads`` (h6kqood..., libopenblasp-*.so) and ``threads=none``
(kj5aqqk..., libopenblas-*.so, serial). A GEMM-heavy DaCe kernel that resolves
the serial build at link/load time silently runs single-threaded no matter
what OMP_NUM_THREADS/OPENBLAS_NUM_THREADS say -- exactly the "GEMM-heavy
kernels are slow" symptom flagged earlier. Three checks, in order:

  1. Which .so does the dynamic linker resolve for "libopenblas.so" under the
     CURRENT LD_LIBRARY_PATH (the same resolution every DaCe-compiled .so
     will do at dlopen time)? Read via /proc/self/maps after a ctypes load
     (readlink on /proc/self/maps entries, not ldconfig -- ldconfig's cache
     can lag a spack-mutated LD_LIBRARY_PATH).
  2. What does that library's OWN introspection say -- openblas_get_parallel()
     (0=serial, 1=pthreads, 2=openmp), openblas_get_config(), the number of
     procs it sees?
  3. Does a REAL DaCe-compiled GEMM kernel (native build, same pipeline as
     the perf jobs) actually get faster as threads increase? This is the
     only check that can't be fooled by an env var that the library ignores.

    sbatch slurm_openblas_check.sh
"""
import ctypes
import ctypes.util
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(sys.path[0]))

import engine  # noqa: E402  (MPI anti-hang env + configure_dace_process)

N = 2048  # square GEMM size: large enough that BLAS threading dominates over call overhead


def _fmt_size(n):
    return f'{n:,}'


def check_linker_resolution():
    print('=' * 70)
    print('1. Dynamic linker resolution of "libopenblas.so" under this env')
    print('=' * 70)
    print(f'  LD_LIBRARY_PATH={os.environ.get("LD_LIBRARY_PATH", "")}')
    print(f'  LIBRARY_PATH={os.environ.get("LIBRARY_PATH", "")}')
    print(f'  OPENBLAS_DIR={os.environ.get("OPENBLAS_DIR", "")}')
    found = ctypes.util.find_library('openblas')
    print(f'  ctypes.util.find_library("openblas") -> {found}')
    try:
        lib = ctypes.CDLL('libopenblas.so', mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f'  FAILED to dlopen libopenblas.so: {e}')
        return None
    # Resolve the actual mapped path via /proc/self/maps (authoritative -- reflects
    # what the loader actually picked, unlike ldconfig's possibly-stale cache).
    resolved = None
    with open('/proc/self/maps') as f:
        for line in f:
            if 'libopenblas' in line:
                path = line.strip().split()[-1]
                if path not in (resolved,):
                    resolved = path
    print(f'  Resolved via /proc/self/maps -> {resolved}')
    is_pthreads_build = bool(resolved and re.search(r'libopenblasp[-.]', os.path.basename(resolved)))
    print(f'  Filename indicates pthreads build (libopenblasp-*): {is_pthreads_build}')
    return lib, resolved, is_pthreads_build


def check_library_introspection(lib):
    print()
    print('=' * 70)
    print('2. OpenBLAS runtime introspection')
    print('=' * 70)
    parallel_names = {0: 'SEQUENTIAL', 1: 'THREADED (pthreads)', 2: 'THREADED (OpenMP)'}
    try:
        lib.openblas_get_parallel.restype = ctypes.c_int
        mode = lib.openblas_get_parallel()
        print(f'  openblas_get_parallel() = {mode} ({parallel_names.get(mode, "unknown")})')
    except AttributeError:
        mode = None
        print('  openblas_get_parallel() not exported (unexpected for a modern OpenBLAS)')
    try:
        lib.openblas_get_config.restype = ctypes.c_char_p
        cfg = lib.openblas_get_config()
        print(f'  openblas_get_config() = {cfg.decode() if cfg else None}')
    except AttributeError:
        pass
    try:
        lib.openblas_get_num_procs.restype = ctypes.c_int
        print(f'  openblas_get_num_procs() = {lib.openblas_get_num_procs()}')
    except AttributeError:
        pass
    try:
        lib.openblas_get_num_threads.restype = ctypes.c_int
        print(f'  openblas_get_num_threads() [current] = {lib.openblas_get_num_threads()}')
    except AttributeError:
        pass
    return mode


def build_and_compile_gemm():
    """A standalone DGEMM DaCe program, compiled NATIVELY (same direct-compile path
    engine.py uses for the perf jobs, not the CMake path -- isolates the OpenBLAS
    link/load question from CMake's own library discovery)."""
    import dace
    import numpy as np

    M, K, Nn = N, N, N

    @dace.program
    def gemm(A: dace.float64[M, K], B: dace.float64[K, Nn], C: dace.float64[M, Nn]):
        C[:] = A @ B

    sdfg = gemm.to_sdfg(simplify=True)
    sdfg.name = 'openblas_check_gemm'
    csdfg = sdfg.compile()
    return sdfg, csdfg


def find_compiled_so(sdfg):
    build_root = sdfg.build_folder
    for root, _, files in os.walk(build_root):
        for fn in files:
            if fn.startswith('lib') and fn.endswith('.so'):
                return os.path.join(root, fn)
    return None


def check_compiled_kernel_link(so_path):
    print()
    print('=' * 70)
    print('3a. What does the COMPILED GEMM kernel itself link against?')
    print('=' * 70)
    print(f'  .so = {so_path}')
    try:
        out = subprocess.run(['ldd', so_path], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        print(f'  ldd failed: {e}')
        return
    for line in out.splitlines():
        if 'openblas' in line.lower() or 'blas' in line.lower():
            print(f'  {line.strip()}')


def check_scaling(csdfg):
    print()
    print('=' * 70)
    print('3b. Does the compiled kernel actually get faster with more threads?')
    print('=' * 70)
    import numpy as np
    rng = np.random.default_rng(0)
    A = np.asfortranarray(rng.random((N, N)))
    B = np.asfortranarray(rng.random((N, N)))
    C = np.zeros((N, N), order='F')

    reps = 5
    results = {}
    for threads in (1, max(1, int(os.environ.get('OMP_NUM_THREADS', '72')))):
        os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
        os.environ['OMP_NUM_THREADS'] = str(threads)
        # warmup
        csdfg(A=A, B=B, C=C)
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            csdfg(A=A, B=B, C=C)
            samples.append(time.perf_counter() - t0)
        samples.sort()
        median = samples[len(samples) // 2]
        results[threads] = median
        print(f'  OPENBLAS_NUM_THREADS={threads:3d}: median={median * 1000:.2f} ms '
              f'(samples_ms={[round(s * 1000, 2) for s in samples]})')

    threads_list = sorted(results)
    if len(threads_list) == 2:
        lo, hi = threads_list
        speedup = results[lo] / results[hi]
        ideal = hi / lo
        print(f'\n  {_fmt_size(N)}x{_fmt_size(N)} DGEMM speedup {lo}->{hi} threads: '
              f'{speedup:.2f}x (ideal ~{ideal:.0f}x, DGEMM realistically 0.3-0.7x of ideal)')
        return speedup
    return None


def main():
    engine.configure_dace_process()
    resolution = check_linker_resolution()
    if resolution is None:
        print('\nVERDICT: FAIL -- libopenblas.so could not be loaded at all.')
        return 1
    lib, resolved_path, is_pthreads_build = resolution
    parallel_mode = check_library_introspection(lib)

    print()
    print('=' * 70)
    print(f'Compiling a {_fmt_size(N)}x{_fmt_size(N)} DGEMM DaCe kernel (native build)...')
    print('=' * 70)
    t0 = time.perf_counter()
    sdfg, csdfg = build_and_compile_gemm()
    print(f'  compiled in {time.perf_counter() - t0:.1f}s')

    so_path = find_compiled_so(sdfg)
    if so_path:
        check_compiled_kernel_link(so_path)

    speedup = check_scaling(csdfg)

    print()
    print('=' * 70)
    print('VERDICT')
    print('=' * 70)
    print(f'  Resolved libopenblas: {resolved_path}')
    print(f'  Pthreads build (filename): {is_pthreads_build}')
    print(f'  openblas_get_parallel(): {parallel_mode}')
    print(f'  Measured scaling speedup: {speedup}')
    ok = bool(is_pthreads_build) and parallel_mode in (1, 2) and speedup is not None and speedup > 1.5
    print(f'  MULTITHREADED OPENBLAS WORKING: {"YES" if ok else "NO -- see checks above"}')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
