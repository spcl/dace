"""A/B performance comparison for ``ScatterToGuardedMaps`` off vs on.

ONE scatter kernel::

    for i in range(N):
        out[idx[i]] = A[i] + bias

Variant A (off) -- the literal sequential SDFG; the scatter loop stays a
``LoopRegion`` because canonicalize cannot prove ``idx`` is a permutation.

Variant B (on) -- ``ScatterToGuardedMaps`` runs after canonicalize: it
inserts a sort+duplicate-count guard on ``idx`` and lifts the loop to a
parallel ``Map[i]`` (``emit_unparallelized_else_branch=False`` -- the
permutation contract is enforced by the duplicate-count check; if a
duplicate were ever observed, ``__builtin_trap()`` aborts; with a clean
permutation the parallel Map runs).

CPU + GPU; the GPU side relies on the guarded Map being a vanilla parallel
loop (no Scan, no per-thread buffer; the sort is run once on the host as
part of the guard step on first call).

Run with::

  pytest tests/ab_perf/test_scatter_ab.py --ab-perf -s
  pytest tests/ab_perf/test_scatter_ab.py --ab-perf --no-gpu -s
"""
import functools

import numpy as np

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.scatter_to_guarded_maps import ScatterToGuardedMaps

from tests.ab_perf._harness import format_ab, time_cpu, time_gpu, to_gpu

N = dace.symbol('N')


@dace.program
def _scatter(A: dace.float64[N], idx: dace.int32[N], bias: dace.float64, out: dace.float64[N]):
    """``out[idx[i]] = A[i] + bias`` -- pure write-scatter (same shape as
    TSVC ``s491`` / ``vas``). Variant A (default canonicalize) lifts the
    loop permissively to a parallel Map without proving ``idx`` is a
    permutation. Variant B adds a sort-based runtime duplicate-count
    guard before lifting -- safe even when ``idx`` collides."""
    for i in range(N):
        out[idx[i]] = A[i] + bias


def _build_variant_a(suffix: str, target: str = 'cpu') -> dace.SDFG:
    """Variant A (knob OFF): canonicalize alone -- the default pipeline
    lifts the scatter via permissive ``LoopToMap`` (assumes ``idx`` is a
    permutation without proof). NO runtime guard; correctness on a
    non-permutation ``idx`` is the caller's responsibility."""
    sdfg = _scatter.to_sdfg(simplify=True)
    sdfg.name = f'{sdfg.name}_unguarded_{suffix}'
    canonicalize(sdfg, validate=True, target=target)
    sdfg.validate()
    return sdfg


def _build_variant_b(suffix: str, target: str = 'cpu') -> dace.SDFG:
    """Variant B (knob ON): run ``ScatterToGuardedMaps`` BEFORE canonicalize
    so the scatter loop is still a ``LoopRegion`` for the pass to detect.
    The pass inserts a sort + duplicate-count guard on ``idx`` and lifts
    to a parallel Map; canonicalize then cleans up around it."""
    sdfg = _scatter.to_sdfg(simplify=True)
    sdfg.name = f'{sdfg.name}_guarded_{suffix}'
    res = ScatterToGuardedMaps(emit_unparallelized_else_branch=False).apply_pass(sdfg, {})
    assert res is not None and res >= 1, f'ScatterToGuardedMaps must lift the scatter; got res={res}'
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


def test_scatter_ab(ab_iters, ab_warmup, ab_gpu_enabled, capsys):
    """A/B compare sequential scatter vs guarded-Map scatter. ``idx`` is
    a real permutation (deterministic shuffle), so the duplicate-count
    check passes and the parallel Map dispatches at every call."""
    n = 1 << 20
    rng = np.random.default_rng(seed=0xCAFE)
    A_init = rng.standard_normal(n).astype(np.float64)
    idx = rng.permutation(n).astype(np.int32)
    bias = 0.25
    ref = np.zeros(n, dtype=np.float64)
    ref[idx] = A_init + bias

    sdfg_a = _build_variant_a('cpu', target='cpu')
    sdfg_b = _build_variant_b('cpu', target='cpu')

    def _make_cpu(sdfg):
        A = A_init.copy()
        ix = idx.copy()
        out = np.zeros(n, dtype=np.float64)
        fn = functools.partial(sdfg, A=A, idx=ix, bias=bias, out=out, N=n)
        fn()
        if not np.allclose(out, ref):
            raise AssertionError(f'{sdfg.name} numerical mismatch; max diff {np.abs(out - ref).max():.3e}')

        def reset_and_call():
            A[:] = A_init
            ix[:] = idx
            out.fill(0)
            fn()

        return reset_and_call

    stats_a_cpu = time_cpu(_make_cpu(sdfg_a), iters=ab_iters, warmup=ab_warmup)
    stats_b_cpu = time_cpu(_make_cpu(sdfg_b), iters=ab_iters, warmup=ab_warmup)

    lines = [
        '', f'== scatter A/B  N={n}  iters={ab_iters} ==', 'CPU:',
        format_ab('A (seq scatter)', stats_a_cpu, 'B (guarded Map)', stats_b_cpu)
    ]

    if ab_gpu_enabled:
        import cupy
        sdfg_a_gpu = _to_gpu_sdfg(sdfg_a, 'gpu_A', device_resident_data=('A', 'idx', 'out'))
        sdfg_b_gpu = _to_gpu_sdfg(_build_variant_b('gpu', target='gpu'),
                                  'gpu_B',
                                  device_resident_data=('A', 'idx', 'out'))

        def _make_gpu(sdfg):
            A = to_gpu(A_init)
            ix = to_gpu(idx)
            out = to_gpu(np.zeros(n, dtype=np.float64))
            fn = functools.partial(sdfg, A=A, idx=ix, bias=bias, out=out, N=n)
            fn()
            cupy.cuda.runtime.deviceSynchronize()
            out_h = cupy.asnumpy(out)
            if not np.allclose(out_h, ref):
                lines.append(f'  WARNING: {sdfg.name} GPU mismatch; max diff {np.abs(out_h - ref).max():.3e}')

            def reset_and_call():
                A[...] = to_gpu(A_init)
                ix[...] = to_gpu(idx)
                out[...] = to_gpu(np.zeros(n, dtype=np.float64))
                fn()

            return reset_and_call

        stats_a_gpu = time_gpu(_make_gpu(sdfg_a_gpu), iters=ab_iters, warmup=ab_warmup)
        stats_b_gpu = time_gpu(_make_gpu(sdfg_b_gpu), iters=ab_iters, warmup=ab_warmup)
        lines.append('GPU:')
        lines.append(format_ab('A (seq scatter)', stats_a_gpu, 'B (guarded Map)', stats_b_gpu))
    else:
        lines.append('GPU: SKIPPED')

    with capsys.disabled():
        print('\n'.join(lines))
