"""A/B performance comparisons for canonicalize knobs.

Covers two knobs (the third, ``interchange_carry_with_map``, is exercised
by ``test_for_1133_ab.py``):

* ``peel_limit``: 0 vs 4 vs 8 on the multi-front pattern where peeling
  off the boundary-conflict iterations is what unblocks ``LoopToMap``.
* ``break_anti_dependence``: False vs True on a pure read-ahead
  anti-dependence kernel that ``LoopToMap`` otherwise refuses.

Both runtimes (the resulting SDFG's kernel call) are timed on CPU and
GPU; canonicalize-itself time is not the subject -- the knob's effect on
the compiled kernel is.

Run with::

  pytest tests/ab_perf/test_canon_knobs_ab.py --ab-perf -s
  pytest tests/ab_perf/test_canon_knobs_ab.py --ab-perf --no-gpu -s
"""
import functools
import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.ab_perf._harness import format_ab, time_cpu, time_gpu, to_gpu


N = dace.symbol('N')


# ---------------------------------------------------------------------------
# Peel-limit A/B: multi-front conflict-write loop
# ---------------------------------------------------------------------------


@dace.program
def _multi_front_peel(A: dace.float64[N], B: dace.float64[N]):
    """``LoopToMap`` cannot lift this loop without peeling: iterations
    i=0 and i=1 each write a tail element of ``A`` (``A[N-1]`` and
    ``A[N-2]``) on top of the per-iteration ``A[i] = B[i] * 2``.
    Peeling off those two head iterations leaves a disjoint-write
    remainder that maps cleanly."""
    for i in range(N):
        A[i] = B[i] * 2.0
        if i == 0:
            A[N - 1] = A[N - 1] + 1.0
        elif i == 1:
            A[N - 2] = A[N - 2] + 1.0


def _multi_front_oracle(A_init, B):
    A = A_init.copy()
    n = A.shape[0]
    for i in range(n):
        A[i] = B[i] * 2.0
        if i == 0:
            A[n - 1] = A[n - 1] + 1.0
        elif i == 1:
            A[n - 2] = A[n - 2] + 1.0
    return A


def _build_canon_sdfg(prog, name_suffix: str, **canon_kwargs) -> dace.SDFG:
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = f'{sdfg.name}_{name_suffix}'
    canonicalize(sdfg, validate=True, **canon_kwargs)
    sdfg.validate()
    return sdfg


def _to_gpu_sdfg(sdfg: dace.SDFG, suffix: str, device_resident_data=()) -> dace.SDFG:
    """Same pattern as test_for_1133_ab._to_gpu_sdfg.

    No Scan libnode handling needed -- the kernels in this file do not
    produce Scans (they're peeling / anti-dependence patterns, not
    prefix sums).
    """
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


@pytest.mark.parametrize('peel_limits', [(0, 4), (0, 8), (4, 8)])
def test_peel_limit_ab_cpu_gpu(peel_limits, ab_iters, ab_warmup, ab_gpu_enabled, capsys):
    """A/B compare peel_limit=0 (off) vs peel_limit=4 (on) on the
    multi-front shape. Variant B should generate a peeled remainder Map
    that runs cold-cache much faster than the sequential A on big N."""
    pl_a, pl_b = peel_limits
    n = 1 << 20  # 1M elements
    rng = np.random.default_rng(seed=42)
    A_init = rng.standard_normal(n)
    B = rng.standard_normal(n)
    ref = _multi_front_oracle(A_init, B)

    sdfg_a = _build_canon_sdfg(_multi_front_peel, f'peel{pl_a}', peel_limit=pl_a)
    sdfg_b = _build_canon_sdfg(_multi_front_peel, f'peel{pl_b}', peel_limit=pl_b)

    def _make_cpu_fn(sdfg):
        A = A_init.copy()
        Bb = B.copy()
        fn = functools.partial(sdfg, A=A, B=Bb, N=n)
        fn()
        if not np.allclose(A, ref):
            raise AssertionError(f'{sdfg.name} numerical mismatch: max diff {np.abs(A - ref).max():.3e}')

        def reset_and_call():
            A[:] = A_init
            Bb[:] = B
            fn()

        return reset_and_call

    stats_a_cpu = time_cpu(_make_cpu_fn(sdfg_a), iters=ab_iters, warmup=ab_warmup)
    stats_b_cpu = time_cpu(_make_cpu_fn(sdfg_b), iters=ab_iters, warmup=ab_warmup)

    lines = ['', f'== peel_limit A/B  N={n}  iters={ab_iters} ==', 'CPU:',
             format_ab(f'peel_limit={pl_a}', stats_a_cpu, f'peel_limit={pl_b}', stats_b_cpu)]

    if ab_gpu_enabled:
        import cupy
        sdfg_a_gpu = _to_gpu_sdfg(sdfg_a, 'gpu_A', device_resident_data=('A', 'B'))
        sdfg_b_gpu = _to_gpu_sdfg(sdfg_b, 'gpu_B', device_resident_data=('A', 'B'))

        def _make_gpu_fn(sdfg):
            A = to_gpu(A_init)
            Bb = to_gpu(B)
            fn = functools.partial(sdfg, A=A, B=Bb, N=n)
            fn()
            cupy.cuda.runtime.deviceSynchronize()

            def reset_and_call():
                A[...] = to_gpu(A_init)
                Bb[...] = to_gpu(B)
                fn()

            return reset_and_call

        stats_a_gpu = time_gpu(_make_gpu_fn(sdfg_a_gpu), iters=ab_iters, warmup=ab_warmup)
        stats_b_gpu = time_gpu(_make_gpu_fn(sdfg_b_gpu), iters=ab_iters, warmup=ab_warmup)
        lines.append('GPU:')
        lines.append(format_ab(f'peel_limit={pl_a}', stats_a_gpu, f'peel_limit={pl_b}', stats_b_gpu))
    else:
        lines.append('GPU: SKIPPED')

    with capsys.disabled():
        print('\n'.join(lines))


# ---------------------------------------------------------------------------
# break_anti_dependence A/B: read-ahead anti-dep loop
# ---------------------------------------------------------------------------


@dace.program
def _anti_dep_kernel(A: dace.float64[N], B: dace.float64[N]):
    """Pure read-ahead anti-dep: ``A[i]`` depends on ``A[i+1]``. ``LoopToMap``
    refuses without the snapshot-rename preprocess; with
    ``break_anti_dependence=True`` the kernel becomes a parallel Map."""
    for i in range(N - 1):
        A[i] = A[i + 1] + B[i]


def _anti_dep_oracle(A_init, B):
    A = A_init.copy()
    n = A.shape[0]
    for i in range(n - 1):
        A[i] = A[i + 1] + B[i]
    return A


def test_break_anti_dependence_ab_cpu_gpu(ab_iters, ab_warmup, ab_gpu_enabled, capsys):
    """A/B compare ``break_anti_dependence=False`` (off, stays sequential)
    vs ``=True`` (snapshot-renames and lifts to Map). Variant B introduces
    a transient + copy but unlocks parallelism."""
    n = 1 << 20
    rng = np.random.default_rng(seed=43)
    A_init = rng.standard_normal(n)
    B = rng.standard_normal(n)
    ref = _anti_dep_oracle(A_init, B)

    sdfg_a = _build_canon_sdfg(_anti_dep_kernel, 'antidep_off', break_anti_dependence=False)
    sdfg_b = _build_canon_sdfg(_anti_dep_kernel, 'antidep_on', break_anti_dependence=True)

    def _make_cpu_fn(sdfg):
        A = A_init.copy()
        Bb = B.copy()
        fn = functools.partial(sdfg, A=A, B=Bb, N=n)
        fn()
        if not np.allclose(A, ref):
            raise AssertionError(f'{sdfg.name} numerical mismatch: max diff {np.abs(A - ref).max():.3e}')

        def reset_and_call():
            A[:] = A_init
            Bb[:] = B
            fn()

        return reset_and_call

    stats_a_cpu = time_cpu(_make_cpu_fn(sdfg_a), iters=ab_iters, warmup=ab_warmup)
    stats_b_cpu = time_cpu(_make_cpu_fn(sdfg_b), iters=ab_iters, warmup=ab_warmup)

    lines = ['', f'== break_anti_dependence A/B  N={n}  iters={ab_iters} ==', 'CPU:',
             format_ab('off', stats_a_cpu, 'on', stats_b_cpu)]

    if ab_gpu_enabled:
        import cupy
        sdfg_a_gpu = _to_gpu_sdfg(sdfg_a, 'gpu_A', device_resident_data=('A', 'B'))
        sdfg_b_gpu = _to_gpu_sdfg(sdfg_b, 'gpu_B', device_resident_data=('A', 'B'))

        def _make_gpu_fn(sdfg):
            A = to_gpu(A_init)
            Bb = to_gpu(B)
            fn = functools.partial(sdfg, A=A, B=Bb, N=n)
            fn()
            cupy.cuda.runtime.deviceSynchronize()

            def reset_and_call():
                A[...] = to_gpu(A_init)
                Bb[...] = to_gpu(B)
                fn()

            return reset_and_call

        stats_a_gpu = time_gpu(_make_gpu_fn(sdfg_a_gpu), iters=ab_iters, warmup=ab_warmup)
        stats_b_gpu = time_gpu(_make_gpu_fn(sdfg_b_gpu), iters=ab_iters, warmup=ab_warmup)
        lines.append('GPU:')
        lines.append(format_ab('off', stats_a_gpu, 'on', stats_b_gpu))
    else:
        lines.append('GPU: SKIPPED')

    with capsys.disabled():
        print('\n'.join(lines))
