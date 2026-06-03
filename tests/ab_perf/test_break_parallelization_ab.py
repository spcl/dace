"""A/B performance comparison for the ``break`` early-exit shape across a
sweep of break-firing positions.

ONE kernel: a linear search ``for i in range(N): if A[i] > threshold:
out[0] = i; break``. The break-firing position is controlled by the input
data (``A[k] = threshold + 1`` for a specific ``k``, all other elements
below threshold); we sweep ``k in {N/8, N/4, N/2, 3N/4, 7N/8}``.

Variant A (off) -- the sequential SDFG with the literal early-exit
loop. Wall time scales with ``k`` because the loop returns as soon as
the predicate fires.

Variant B (on) -- the canonicalize-lifted form: ``EarlyExitToFindIndex``
turns the break-loop into a parallel ``Map[i]`` + reduction over the
predicate, finding the minimum index where the predicate fires. Wall
time is ~constant (proportional to ``N``, not ``k``); the constant is
the parallel scan + min-reduce.

The crossover point (break position where B starts to win) reveals
whether the lifted form is faster than the sequential form for the
typical break-fires-mid-loop case. CPU + GPU; the GPU side relies on
the lifted form being a clean Map (no Scan, no per-thread buffer).

Run with::

  pytest tests/ab_perf/test_break_parallelization_ab.py --ab-perf -s
  pytest tests/ab_perf/test_break_parallelization_ab.py --ab-perf --no-gpu -s
"""
import functools

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.ab_perf._harness import format_ab, time_cpu, time_gpu, to_gpu


N = dace.symbol('N')


@dace.program
def _find_first_above(A: dace.float64[N], threshold: dace.float64, out: dace.int32[1]):
    """Linear search: return the first ``i`` where ``A[i] > threshold``.
    The ``break`` makes Variant A's wall-time scale with the break-firing
    position; ``EarlyExitToFindIndex`` lifts this to a parallel min-of-indices
    over the predicate."""
    out[0] = -1
    for i in range(N):
        if A[i] > threshold:
            out[0] = i
            break


def _build_seq_sdfg(suffix: str) -> dace.SDFG:
    """Variant A: the unlifted (sequential break-loop) SDFG, no canonicalize."""
    sdfg = _find_first_above.to_sdfg(simplify=True)
    sdfg.name = f'{sdfg.name}_seq_{suffix}'
    sdfg.validate()
    return sdfg


def _build_parallel_sdfg(suffix: str, target: str = 'cpu') -> dace.SDFG:
    """Variant B: canonicalize lifts ``EarlyExitToFindIndex`` -> parallel
    Map + min-reduce."""
    sdfg = _find_first_above.to_sdfg(simplify=True)
    sdfg.name = f'{sdfg.name}_par_{suffix}'
    canonicalize(sdfg, validate=True, target=target)
    sdfg.validate()
    return sdfg


def _to_gpu_sdfg(sdfg: dace.SDFG, suffix: str, device_resident_data=()) -> dace.SDFG:
    import copy
    gpu = copy.deepcopy(sdfg)
    gpu.name = f'{sdfg.name}_{suffix}'
    for arr in device_resident_data:
        if arr in gpu.arrays:
            gpu.arrays[arr].storage = dace.dtypes.StorageType.GPU_Global
    gpu.apply_gpu_transformations(host_data=list(device_resident_data))
    for arr in device_resident_data:
        if arr in gpu.arrays:
            gpu.arrays[arr].storage = dace.dtypes.StorageType.GPU_Global
    gpu.validate()
    return gpu


_BREAK_NS = [1 << 18, 1 << 20, 1 << 22, 1 << 24]  # 256K, 1M, 4M, 16M


@pytest.mark.parametrize('n_param', _BREAK_NS)
def test_break_parallelization_ab(n_param, ab_iters, ab_warmup, ab_gpu_enabled, capsys):
    """A/B compare sequential break-loop vs canonicalize-lifted parallel
    find-index across several large ``N`` values, break fires at ``N/2``
    (the typical mid-loop case where the lifted-vs-sequential crossover
    is meaningful)."""
    n = n_param
    k = n // 2
    break_label = f'N/2 (k={k})'
    suffix = f'n{n}_k{k}'  # SDFG-name-safe (letters / digits / underscore only)
    threshold = 0.5
    rng = np.random.default_rng(seed=hash(break_label) & 0xFFFFFFFF)
    # Below-threshold field, then place a single above-threshold spike at k.
    A_init = rng.uniform(low=-1.0, high=threshold - 1e-3, size=n)
    A_init[k] = threshold + 1.0
    out_ref = np.array([k], dtype=np.int32)

    sdfg_a = _build_seq_sdfg(suffix)
    sdfg_b = _build_parallel_sdfg(suffix, target='cpu')

    def _make_cpu(sdfg):
        A = A_init.copy()
        out = np.full((1, ), -1, dtype=np.int32)
        fn = functools.partial(sdfg, A=A, threshold=threshold, out=out, N=n)
        fn()
        if out[0] != out_ref[0]:
            raise AssertionError(f'{sdfg.name} found out[0]={out[0]}, expected {out_ref[0]}')

        def reset_and_call():
            A[:] = A_init
            out[0] = -1
            fn()

        return reset_and_call

    stats_a_cpu = time_cpu(_make_cpu(sdfg_a), iters=ab_iters, warmup=ab_warmup)
    stats_b_cpu = time_cpu(_make_cpu(sdfg_b), iters=ab_iters, warmup=ab_warmup)

    lines = ['', f'== break_parallelization A/B  N={n}  break={break_label} (k={k})  iters={ab_iters} ==', 'CPU:',
             format_ab('A (seq break)', stats_a_cpu, 'B (parallel)', stats_b_cpu)]

    if ab_gpu_enabled:
        import cupy
        sdfg_a_gpu = _to_gpu_sdfg(sdfg_a, 'gpu_A', device_resident_data=('A', 'out'))
        sdfg_b_gpu = _to_gpu_sdfg(_build_parallel_sdfg(suffix, target='gpu'),
                                  'gpu_B', device_resident_data=('A', 'out'))

        def _make_gpu(sdfg):
            A = to_gpu(A_init)
            out = to_gpu(np.full((1, ), -1, dtype=np.int32))
            fn = functools.partial(sdfg, A=A, threshold=threshold, out=out, N=n)
            fn()
            cupy.cuda.runtime.deviceSynchronize()
            out_h = cupy.asnumpy(out)
            if out_h[0] != out_ref[0]:
                lines.append(f'  WARNING: {sdfg.name} GPU result {out_h[0]} != ref {out_ref[0]}')

            def reset_and_call():
                A[...] = to_gpu(A_init)
                out[...] = to_gpu(np.full((1, ), -1, dtype=np.int32))
                fn()

            return reset_and_call

        stats_a_gpu = time_gpu(_make_gpu(sdfg_a_gpu), iters=ab_iters, warmup=ab_warmup)
        stats_b_gpu = time_gpu(_make_gpu(sdfg_b_gpu), iters=ab_iters, warmup=ab_warmup)
        lines.append('GPU:')
        lines.append(format_ab('A (seq break)', stats_a_gpu, 'B (parallel)', stats_b_gpu))
    else:
        lines.append('GPU: SKIPPED')

    with capsys.disabled():
        print('\n'.join(lines))
