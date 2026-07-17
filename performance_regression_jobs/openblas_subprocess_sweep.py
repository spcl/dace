#!/usr/bin/env python3
"""Re-verify the OpenBLAS thread-count collapse with a methodology immune to
ctypes/post-init ambiguity: each measurement is a FRESH subprocess with
OPENBLAS_NUM_THREADS set in its environment BEFORE the interpreter (and hence
libopenblas) ever starts -- eliminates any doubt about whether
openblas_set_num_threads() actually reached the pool the compiled kernel calls
into, or whether two ctypes.CDLL handles of "the same" library were secretly
distinct objects.

Phase 1 (this process): canonicalize + compile the GEMM SDFG ONCE, save the
finalized SDFG so each child subprocess only needs a cache-hit compile (fast).
Phase 2: for each thread count, spawn a clean `python3 -c ...` child with
OPENBLAS_NUM_THREADS baked into its env from birth, print its single measured
median.

Also introspects openblas_get_config() for NO_AFFINITY/spin-wait hints, and
runs the affinity-relevant env knobs (OPENBLAS_CORETYPE non-empty is not
needed; the real lever here is whether pinning via taskset per child changes
the picture) as a bonus A/B: unpinned vs `taskset -c 0-<threads-1>` pinned.
"""
import ctypes
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

N = 2048
THREAD_COUNTS = (1, 2, 4, 8, 16, 32, 72)
CACHE_SDFGZ = '/dev/shm/openblas_subprocess_sweep_gemm.sdfgz'

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
print("RESULT", {threads}, lib.openblas_get_num_threads(), samples[len(samples)//2])
'''


def build_compile_and_cache():
    import dace

    @dace.program
    def gemm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        C[:] = A @ B

    sdfg = gemm.to_sdfg(simplify=True)
    sdfg = engine.pipeline_canon(sdfg, device='cpu')
    sdfg.name = 'openblas_subprocess_sweep_gemm'
    sdfg.save(CACHE_SDFGZ, compress=False)
    # Warm the compile cache too, so children hit it immediately.
    sdfg.compile()
    return sdfg


def introspect():
    lib = ctypes.CDLL('libopenblas.so')
    lib.openblas_get_config.restype = ctypes.c_char_p
    cfg = lib.openblas_get_config()
    print(f'openblas_get_config() = {cfg.decode() if cfg else None}')
    lib.openblas_get_num_procs.restype = ctypes.c_int
    print(f'openblas_get_num_procs() = {lib.openblas_get_num_procs()}')


def run_child(threads, taskset_cores=None):
    perf_dir = os.path.dirname(os.path.abspath(__file__))
    script = CHILD_SCRIPT.format(perf_dir=perf_dir, cache_path=CACHE_SDFGZ, n=N, threads=threads)
    env = dict(os.environ)
    env['OPENBLAS_NUM_THREADS'] = str(threads)
    env['OMP_NUM_THREADS'] = str(threads)
    cmd = ['python3', '-c', script]
    if taskset_cores:
        cmd = ['taskset', '-c', taskset_cores] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT'):
            _, t, actual_threads, median_s = line.split()
            return int(t), int(actual_threads), float(median_s)
    print(f'  [threads={threads} taskset={taskset_cores}] no RESULT line; stdout={proc.stdout[-500:]!r} '
          f'stderr={proc.stderr[-1000:]!r}')
    return threads, None, None


def main():
    engine.configure_dace_process()
    print('=' * 70)
    print('OpenBLAS introspection')
    print('=' * 70)
    introspect()

    print()
    print('=' * 70)
    print(f'Compiling {N}x{N} DGEMM (canon pipeline) once, caching for children...')
    print('=' * 70)
    build_compile_and_cache()

    print()
    print('=' * 70)
    print('Fresh-process-per-thread-count sweep (env set before interpreter starts)')
    print('=' * 70)
    baseline = None
    results = {}
    for threads in THREAD_COUNTS:
        t, actual, median = run_child(threads)
        if median is None:
            continue
        if baseline is None:
            baseline = median
        gflops = (2 * N ** 3 / 1e9) / median
        results[threads] = median
        print(f'  threads={threads:3d} (openblas reports {actual:3d}): median={median * 1000:9.2f} ms '
              f'({gflops:7.1f} GFLOP/s) {median / baseline:6.2f}x vs 1-thread')

    print()
    print('=' * 70)
    print('CPU-pinned bonus A/B (taskset -c 0-<threads-1>) at the worst unpinned cell')
    print('=' * 70)
    if results:
        worst_threads = max(results, key=lambda k: results[k])
        if worst_threads > 1:
            cores = f'0-{worst_threads - 1}'
            t, actual, median = run_child(worst_threads, taskset_cores=cores)
            if median is not None:
                gflops = (2 * N ** 3 / 1e9) / median
                print(f'  threads={worst_threads} PINNED to cores {cores}: median={median * 1000:.2f} ms '
                      f'({gflops:.1f} GFLOP/s), vs {results[worst_threads] * 1000:.2f} ms unpinned '
                      f'({results[worst_threads] / median:.2f}x faster when pinned)'
                      if median < results[worst_threads] else
                      f'  threads={worst_threads} PINNED: {median * 1000:.2f} ms, no better than unpinned '
                      f'({results[worst_threads] * 1000:.2f} ms) -- pinning is not the fix')


if __name__ == '__main__':
    main()
