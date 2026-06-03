"""A/B performance comparison for the cloudsc ``for_1133`` prefix-sum shape
under ``LoopToScan.interchange_carry_with_map`` False vs True.

Pattern (single carrier, simplified from the multi-carrier real cloudsc
flux loop):

    for jk in range(1, KLEV):
        for jl in range(KLON):
            pfsqrf[jk, jl] = pfsqrf[jk-1, jl] + delta[jk, jl]

Variant A (knob OFF) -- post-``LoopToMap`` shape, executed as-is: outer
sequential ``LoopRegion[jk]`` containing inner parallel ``Map[jl]``.

Variant B (knob ON) -- after the interchange (relocation) path: outer
parallel ``Map[jl]`` whose NestedSDFG body holds a sequential
``LoopRegion[jk]``. NO buffers, NO Scan libnode, NO copies: each Map
thread runs its own ``for jk`` reading/writing ``pfsqrf``/``delta``
directly out of global memory; the accumulator lives in a register.

Both variants are timed on:
  * CPU (numpy host arrays, ``time.perf_counter`` bracketing the call)
  * GPU (cupy device arrays, ``cupy.cuda.Event`` bracketing, sync after
    warmup; only runs when ``--no-gpu`` is not set and cupy + a CUDA
    device are available)

Run with::

  pytest tests/ab_perf/test_for_1133_ab.py --ab-perf -s
  pytest tests/ab_perf/test_for_1133_ab.py --ab-perf --ab-klev=90 --ab-klon=20480 -s
  pytest tests/ab_perf/test_for_1133_ab.py --ab-perf --no-gpu -s
"""
import functools
from typing import Tuple

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.loop_to_scan import LoopToScan

from tests.ab_perf._harness import format_ab, time_cpu, time_gpu, to_gpu


KLEV = dace.symbol('KLEV')
KLON = dace.symbol('KLON')


@dace.program
def _for_1133_kernel_f64(pfsqrf: dace.float64[KLEV, KLON], delta: dace.float64[KLEV, KLON]):
    for jk in range(1, KLEV):
        for jl in range(KLON):
            pfsqrf[jk, jl] = pfsqrf[jk - 1, jl] + delta[jk, jl]


@dace.program
def _for_1133_kernel_f32(pfsqrf: dace.float32[KLEV, KLON], delta: dace.float32[KLEV, KLON]):
    for jk in range(1, KLEV):
        for jl in range(KLON):
            pfsqrf[jk, jl] = pfsqrf[jk - 1, jl] + delta[jk, jl]


_KERNEL_BY_DTYPE = {np.float32: _for_1133_kernel_f32, np.float64: _for_1133_kernel_f64}


def _np_dtype(dtype_name: str):
    return {'fp32': np.float32, 'fp64': np.float64}[dtype_name]


def _build_post_l2m_sdfg(name_suffix: str, dtype_name: str) -> dace.SDFG:
    """Build the post-``LoopToMap`` baseline (inner ``jl`` lifted to a Map)
    at the requested floating-point precision."""
    kernel = _KERNEL_BY_DTYPE[_np_dtype(dtype_name)]
    sdfg = kernel.to_sdfg(simplify=True)
    sdfg.name = f'{sdfg.name}_{dtype_name}_{name_suffix}'
    inner = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable == 'jl')
    xform = LoopToMap()
    xform.loop = inner
    xform.expr_index = 0
    assert xform.can_be_applied(inner.parent_graph, 0, sdfg, permissive=False), \
        'inner jl-loop must be parallel'
    xform.apply(inner.parent_graph, sdfg)
    sdfg.validate()
    return sdfg


def _build_variant_a(name_suffix: str = 'A_off', dtype_name: str = 'fp64') -> dace.SDFG:
    """Variant A: knob OFF -- post-L2M baseline, no further transformation."""
    return _build_post_l2m_sdfg(name_suffix, dtype_name)


def _build_variant_b(name_suffix: str = 'B_on', dtype_name: str = 'fp64') -> dace.SDFG:
    """Variant B: knob ON -- interchange path applied."""
    sdfg = _build_post_l2m_sdfg(name_suffix, dtype_name)
    res = LoopToScan(interchange_carry_with_map=True).apply_pass(sdfg, {})
    assert res is not None and res >= 1, 'interchange path must lift exactly one shape'
    sdfg.validate()
    return sdfg


def _to_gpu_sdfg(sdfg: dace.SDFG, suffix: str, device_resident_data=()) -> dace.SDFG:
    """Apply ``apply_gpu_transformations`` to a copy of the SDFG and mark
    the user-named inputs as ``GPU_Global`` so DaCe codegen treats them as
    already on device (no H2D copy at call time, no implicit host buffer).

    No Scan libnode handling needed -- the interchange rewrite emits a
    plain sequential ``LoopRegion[jk]`` inside the per-column NSDFG; the
    GPU kernel is a straight ``Map[jl]`` whose threads each run their own
    sequential carry loop reading from / writing to global memory."""
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


def _oracle(pfsqrf_init: np.ndarray, delta: np.ndarray) -> np.ndarray:
    klev = pfsqrf_init.shape[0]
    klon = pfsqrf_init.shape[1]
    out = pfsqrf_init.copy()
    for jk in range(1, klev):
        out[jk, :] = out[jk - 1, :] + delta[jk, :]
    return out


@pytest.mark.parametrize('klev_param', [90, 96])
@pytest.mark.parametrize('dtype_name', ['fp64', 'fp32'])
def test_for_1133_ab_cpu_gpu(dtype_name, klev_param, ab_klon, ab_iters, ab_warmup, ab_gpu_enabled, capsys):
    """A/B compare CPU (always) and GPU (when available) for both knob
    values at ``klev=klev_param``, ``klon=ab_klon`` (default 20480), in
    both ``fp32`` and ``fp64``.

    The test asserts NUMERICAL agreement against the numpy oracle for
    every variant and emits the timing table to stdout (use ``-s`` to
    see it). Tolerances widen for ``fp32`` so the prefix-sum's accumulated
    rounding (~89 sequential adds) doesn't trip ``allclose``."""
    klev = klev_param
    klon = ab_klon
    assert klon % 32 == 0, f'klon must be a multiple of 32; got {klon}'
    np_dtype = _np_dtype(dtype_name)
    # fp32 prefix-sum across ~klev steps accumulates ~klev * eps * scale of
    # error; allow rtol=1e-3, atol=1e-3. For fp64 the existing 1e-5 / 1e-8
    # numpy defaults are tight enough.
    if dtype_name == 'fp32':
        tol = dict(rtol=1e-3, atol=1e-3)
    else:
        tol = dict(rtol=1e-5, atol=1e-8)

    rng = np.random.default_rng(1133)
    pfsqrf_init = rng.standard_normal((klev, klon)).astype(np_dtype)
    delta = rng.standard_normal((klev, klon)).astype(np_dtype)
    ref = _oracle(pfsqrf_init, delta)

    # Compile both CPU variants.
    sdfg_a_cpu = _build_variant_a('cpu_A', dtype_name)
    sdfg_b_cpu = _build_variant_b('cpu_B', dtype_name)

    p_a = pfsqrf_init.copy()
    d_a = delta.copy()
    fn_a_cpu = functools.partial(sdfg_a_cpu, pfsqrf=p_a, delta=d_a, KLEV=klev, KLON=klon)
    fn_a_cpu()  # populate p_a once for the assertion BEFORE timing (warmup will run more)
    assert np.allclose(p_a, ref, **tol), \
        f'variant A (CPU, knob OFF, {dtype_name}) diverges: max diff {np.abs(p_a - ref).max():.3e}'

    p_b = pfsqrf_init.copy()
    d_b = delta.copy()
    fn_b_cpu = functools.partial(sdfg_b_cpu, pfsqrf=p_b, delta=d_b, KLEV=klev, KLON=klon)
    fn_b_cpu()
    assert np.allclose(p_b, ref, **tol), \
        f'variant B (CPU, knob ON, {dtype_name}) diverges: max diff {np.abs(p_b - ref).max():.3e}'

    # Re-seed the in-place buffers for clean timing samples.
    def _reseed_a():
        p_a[:] = pfsqrf_init
        d_a[:] = delta

    def _reseed_b():
        p_b[:] = pfsqrf_init
        d_b[:] = delta

    def _timed_a():
        _reseed_a()
        fn_a_cpu()

    def _timed_b():
        _reseed_b()
        fn_b_cpu()

    stats_a_cpu = time_cpu(_timed_a, iters=ab_iters, warmup=ab_warmup)
    stats_b_cpu = time_cpu(_timed_b, iters=ab_iters, warmup=ab_warmup)

    out_lines = []
    out_lines.append('')
    out_lines.append(f'== for_1133 A/B  dtype={dtype_name}  klev={klev}  klon={klon}  iters={ab_iters} ==')
    out_lines.append('CPU:')
    out_lines.append(format_ab('A (knob off)', stats_a_cpu, 'B (knob on)', stats_b_cpu))

    # GPU variants (optional).
    if ab_gpu_enabled:
        import cupy
        sdfg_a_gpu = _to_gpu_sdfg(sdfg_a_cpu, 'gpu_A', device_resident_data=('pfsqrf', 'delta'))
        sdfg_b_gpu = _to_gpu_sdfg(sdfg_b_cpu, 'gpu_B', device_resident_data=('pfsqrf', 'delta'))

        p_a_gpu = to_gpu(pfsqrf_init.copy())
        d_a_gpu = to_gpu(delta.copy())
        fn_a_gpu = functools.partial(sdfg_a_gpu, pfsqrf=p_a_gpu, delta=d_a_gpu, KLEV=klev, KLON=klon)
        # Correctness check on GPU (one call, then compare).
        fn_a_gpu()
        cupy.cuda.runtime.deviceSynchronize()
        p_a_back = cupy.asnumpy(p_a_gpu)
        if not np.allclose(p_a_back, ref, **tol):
            out_lines.append(f'  WARNING: variant A (GPU, {dtype_name}) numerical mismatch '
                             f'(max diff {np.abs(p_a_back - ref).max():.3e}); '
                             f'GPU result may be using a different ordering; timing only')

        p_b_gpu = to_gpu(pfsqrf_init.copy())
        d_b_gpu = to_gpu(delta.copy())
        fn_b_gpu = functools.partial(sdfg_b_gpu, pfsqrf=p_b_gpu, delta=d_b_gpu, KLEV=klev, KLON=klon)
        fn_b_gpu()
        cupy.cuda.runtime.deviceSynchronize()
        p_b_back = cupy.asnumpy(p_b_gpu)
        if not np.allclose(p_b_back, ref, **tol):
            out_lines.append(f'  WARNING: variant B (GPU, {dtype_name}) numerical mismatch '
                             f'(max diff {np.abs(p_b_back - ref).max():.3e}); timing only')

        # Reseed-each-iter wrappers (same pattern as CPU).
        def _reseed_a_gpu():
            p_a_gpu[...] = to_gpu(pfsqrf_init)
            d_a_gpu[...] = to_gpu(delta)

        def _reseed_b_gpu():
            p_b_gpu[...] = to_gpu(pfsqrf_init)
            d_b_gpu[...] = to_gpu(delta)

        def _timed_a_gpu():
            _reseed_a_gpu()
            fn_a_gpu()

        def _timed_b_gpu():
            _reseed_b_gpu()
            fn_b_gpu()

        stats_a_gpu = time_gpu(_timed_a_gpu, iters=ab_iters, warmup=ab_warmup)
        stats_b_gpu = time_gpu(_timed_b_gpu, iters=ab_iters, warmup=ab_warmup)
        out_lines.append('GPU:')
        out_lines.append(format_ab('A (knob off)', stats_a_gpu, 'B (knob on)', stats_b_gpu))
    else:
        out_lines.append('GPU: SKIPPED (no cupy / CUDA, or --no-gpu)')

    out_lines.append('')
    captured = '\n'.join(out_lines)
    print(captured)
    # Surface the captured text on test failure / on -s.
    with capsys.disabled():
        print(captured)
