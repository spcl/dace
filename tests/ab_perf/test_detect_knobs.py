"""Detect-knobs meta-test.

Runs each non-blocked canonicalize knob A/B on a detected target device
(``cpu`` is always available; ``gpu`` runs only when cupy + a CUDA device
are present) and emits a recommended per-target preset table. Use the
output to verify (or update) the in-source presets at
``dace.transformation.passes.canonicalize.pipeline._CPU_DEFAULTS`` /
``_GPU_DEFAULTS``.

Knob coverage:

* ``interchange_carry_with_map`` -- via cloudsc ``for_1133`` kernel
  (KLEV=96, KLON=20480, fp64). Re-uses the variant builders in
  ``tests/ab_perf/test_for_1133_ab.py``.
* ``break_anti_dependence`` -- via TSVC ``s121``-shaped anti-dep
  kernel at N=1M. Re-uses builders in ``tests/ab_perf/test_canon_knobs_ab.py``.
* ``scatter_to_guarded_maps`` -- via pure-write scatter at N=1M
  permutation idx. Re-uses builders in ``tests/ab_perf/test_scatter_ab.py``.

Knobs intentionally skipped:

* ``peel_limit`` -- fixed at 4 (TSVC corpus coverage anchor); no AB.
* break-parallelization (``EarlyExitToFindIndex``) -- blocked on a core
  ``BreakBlock`` bug in ``dace/sdfg/utils.py:1731`` (pre-existing). The
  variant-A baseline (sequential break-loop) doesn't compile.

For break (when it unblocks): assume ``N/2`` firing position per user
spec.

Run with::

  pytest tests/ab_perf/test_detect_knobs.py --ab-perf -s
  pytest tests/ab_perf/test_detect_knobs.py --ab-perf --no-gpu -s
"""
import functools
from typing import Dict

import numpy as np
import pytest

from dace.transformation.passes.canonicalize.pipeline import (_CPU_DEFAULTS, _GPU_DEFAULTS)

from tests.ab_perf._harness import time_cpu, time_gpu, to_gpu
from tests.ab_perf.test_for_1133_ab import (_build_variant_a as _for1133_a, _build_variant_b as _for1133_b, _to_gpu_sdfg
                                            as _for1133_to_gpu, _oracle as _for1133_oracle)
from tests.ab_perf.test_canon_knobs_ab import (_anti_dep_kernel, _anti_dep_oracle, _build_canon_sdfg, _to_gpu_sdfg as
                                               _canon_to_gpu)
from tests.ab_perf.test_scatter_ab import (_build_variant_a as _scatter_build_a, _build_variant_b as _scatter_build_b,
                                           _to_gpu_sdfg as _scatter_to_gpu)


def _pick(stats_a: Dict[str, float], stats_b: Dict[str, float], knob_off_label: str, knob_on_label: str) -> bool:
    """Return True if knob ON wins (B's median is lower)."""
    return stats_b['median_us'] < stats_a['median_us']


def _measure_for1133(device: str, iters: int, warmup: int) -> bool:
    """interchange_carry_with_map A/B. Returns ``True`` if ON (knob=True)
    wins on ``device``."""
    klev, klon = 96, 20480
    rng = np.random.default_rng(1133)
    p_init = rng.standard_normal((klev, klon))
    delta = rng.standard_normal((klev, klon))
    ref = _for1133_oracle(p_init, delta)

    sdfg_a = _for1133_a('detect_A', 'fp64')
    sdfg_b = _for1133_b('detect_B', 'fp64')
    if device == 'gpu':
        sdfg_a = _for1133_to_gpu(sdfg_a, 'detect_gpu_A', device_resident_data=('pfsqrf', 'delta'))
        sdfg_b = _for1133_to_gpu(sdfg_b, 'detect_gpu_B', device_resident_data=('pfsqrf', 'delta'))

    if device == 'cpu':
        p_a, d_a = p_init.copy(), delta.copy()
        p_b, d_b = p_init.copy(), delta.copy()
        fn_a = functools.partial(sdfg_a, pfsqrf=p_a, delta=d_a, KLEV=klev, KLON=klon)
        fn_b = functools.partial(sdfg_b, pfsqrf=p_b, delta=d_b, KLEV=klev, KLON=klon)
        fn_a()
        fn_b()
        assert np.allclose(p_a, ref) and np.allclose(p_b, ref)

        def _ta():
            p_a[:] = p_init
            d_a[:] = delta
            fn_a()

        def _tb():
            p_b[:] = p_init
            d_b[:] = delta
            fn_b()

        sa = time_cpu(_ta, iters, warmup)
        sb = time_cpu(_tb, iters, warmup)
    else:
        import cupy
        p_a = to_gpu(p_init.copy())
        d_a = to_gpu(delta.copy())
        p_b = to_gpu(p_init.copy())
        d_b = to_gpu(delta.copy())
        fn_a = functools.partial(sdfg_a, pfsqrf=p_a, delta=d_a, KLEV=klev, KLON=klon)
        fn_b = functools.partial(sdfg_b, pfsqrf=p_b, delta=d_b, KLEV=klev, KLON=klon)
        fn_a()
        fn_b()
        cupy.cuda.runtime.deviceSynchronize()

        def _ta():
            p_a[...] = to_gpu(p_init)
            d_a[...] = to_gpu(delta)
            fn_a()

        def _tb():
            p_b[...] = to_gpu(p_init)
            d_b[...] = to_gpu(delta)
            fn_b()

        sa = time_gpu(_ta, iters, warmup)
        sb = time_gpu(_tb, iters, warmup)
    return _pick(sa, sb, 'OFF', 'ON')


def _measure_anti_dep(device: str, iters: int, warmup: int) -> bool:
    """break_anti_dependence A/B. Returns ``True`` if ON wins."""
    n = 1 << 20
    rng = np.random.default_rng(43)
    A_init = rng.standard_normal(n)
    B = rng.standard_normal(n)
    ref = _anti_dep_oracle(A_init, B)

    sdfg_a = _build_canon_sdfg(_anti_dep_kernel, 'detect_off', break_anti_dependence=False)
    sdfg_b = _build_canon_sdfg(_anti_dep_kernel, 'detect_on', break_anti_dependence=True)
    if device == 'gpu':
        sdfg_a = _canon_to_gpu(sdfg_a, 'detect_gpu_off', device_resident_data=('A', 'B'))
        sdfg_b = _canon_to_gpu(sdfg_b, 'detect_gpu_on', device_resident_data=('A', 'B'))

    if device == 'cpu':
        A = A_init.copy()
        Bb = B.copy()
        fn_a = functools.partial(sdfg_a, A=A, B=Bb, N=n)
        fn_b = functools.partial(sdfg_b, A=A_init.copy(), B=B.copy(), N=n)
        fn_a()
        assert np.allclose(A, ref)

        def _ta():
            A[:] = A_init
            Bb[:] = B
            fn_a()

        Ab = A_init.copy()
        Bb2 = B.copy()
        fn_b = functools.partial(sdfg_b, A=Ab, B=Bb2, N=n)
        fn_b()
        assert np.allclose(Ab, ref)

        def _tb():
            Ab[:] = A_init
            Bb2[:] = B
            fn_b()

        sa = time_cpu(_ta, iters, warmup)
        sb = time_cpu(_tb, iters, warmup)
    else:
        import cupy
        A_g = to_gpu(A_init.copy())
        B_g = to_gpu(B.copy())
        Ab_g = to_gpu(A_init.copy())
        Bb_g = to_gpu(B.copy())
        fn_a = functools.partial(sdfg_a, A=A_g, B=B_g, N=n)
        fn_b = functools.partial(sdfg_b, A=Ab_g, B=Bb_g, N=n)
        fn_a()
        fn_b()
        cupy.cuda.runtime.deviceSynchronize()

        def _ta():
            A_g[...] = to_gpu(A_init)
            B_g[...] = to_gpu(B)
            fn_a()

        def _tb():
            Ab_g[...] = to_gpu(A_init)
            Bb_g[...] = to_gpu(B)
            fn_b()

        sa = time_gpu(_ta, iters, warmup)
        sb = time_gpu(_tb, iters, warmup)
    return _pick(sa, sb, 'OFF', 'ON')


def _measure_scatter(device: str, iters: int, warmup: int) -> bool:
    """scatter_to_guarded_maps A/B. Returns ``True`` if ON wins."""
    n = 1 << 20
    rng = np.random.default_rng(seed=0xCAFE)
    A_init = rng.standard_normal(n).astype(np.float64)
    idx = rng.permutation(n).astype(np.int32)
    bias = 0.25
    ref = np.zeros(n, dtype=np.float64)
    ref[idx] = A_init + bias

    sdfg_a = _scatter_build_a('detect', target=device)
    sdfg_b = _scatter_build_b('detect', target=device)
    if device == 'gpu':
        sdfg_a = _scatter_to_gpu(sdfg_a, 'detect_gpu_A', device_resident_data=('A', 'idx', 'out'))
        sdfg_b = _scatter_to_gpu(sdfg_b, 'detect_gpu_B', device_resident_data=('A', 'idx', 'out'))

    if device == 'cpu':
        A = A_init.copy()
        ix = idx.copy()
        out = np.zeros(n, dtype=np.float64)
        Ab = A_init.copy()
        ixb = idx.copy()
        outb = np.zeros(n, dtype=np.float64)
        fn_a = functools.partial(sdfg_a, A=A, idx=ix, bias=bias, out=out, N=n)
        fn_b = functools.partial(sdfg_b, A=Ab, idx=ixb, bias=bias, out=outb, N=n)
        fn_a()
        fn_b()
        assert np.allclose(out, ref) and np.allclose(outb, ref)

        def _ta():
            A[:] = A_init
            ix[:] = idx
            out.fill(0)
            fn_a()

        def _tb():
            Ab[:] = A_init
            ixb[:] = idx
            outb.fill(0)
            fn_b()

        sa = time_cpu(_ta, iters, warmup)
        sb = time_cpu(_tb, iters, warmup)
    else:
        import cupy
        A_g = to_gpu(A_init.copy())
        ix_g = to_gpu(idx.copy())
        out_g = to_gpu(np.zeros(n, dtype=np.float64))
        Ab_g = to_gpu(A_init.copy())
        ixb_g = to_gpu(idx.copy())
        outb_g = to_gpu(np.zeros(n, dtype=np.float64))
        fn_a = functools.partial(sdfg_a, A=A_g, idx=ix_g, bias=bias, out=out_g, N=n)
        fn_b = functools.partial(sdfg_b, A=Ab_g, idx=ixb_g, bias=bias, out=outb_g, N=n)
        fn_a()
        fn_b()
        cupy.cuda.runtime.deviceSynchronize()

        def _ta():
            A_g[...] = to_gpu(A_init)
            ix_g[...] = to_gpu(idx)
            out_g[...] = to_gpu(np.zeros(n, dtype=np.float64))
            fn_a()

        def _tb():
            Ab_g[...] = to_gpu(A_init)
            ixb_g[...] = to_gpu(idx)
            outb_g[...] = to_gpu(np.zeros(n, dtype=np.float64))
            fn_b()

        sa = time_gpu(_ta, iters, warmup)
        sb = time_gpu(_tb, iters, warmup)
    return _pick(sa, sb, 'OFF', 'ON')


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_detect_knobs(target, ab_iters, ab_warmup, ab_gpu_enabled, capsys):
    """Probe each knob's A/B on ``target`` and emit recommended preset."""
    if target == 'gpu' and not ab_gpu_enabled:
        pytest.skip('GPU detect requested but cupy / CUDA not available')

    iters = max(ab_iters, 3)
    warmup = max(ab_warmup, 1)

    recommended: Dict[str, bool] = {
        # peel_limit isn't AB-driven: it's set to the TSVC corpus
        # coverage anchor (4).
        'peel_limit': 4,
        'interchange_carry_with_map': _measure_for1133(target, iters, warmup),
        'break_anti_dependence': _measure_anti_dep(target, iters, warmup),
        'scatter_to_guarded_maps': _measure_scatter(target, iters, warmup),
    }

    in_source = _CPU_DEFAULTS if target == 'cpu' else _GPU_DEFAULTS

    lines = ['', f'== detect_knobs target={target}  iters={iters} ==']
    lines.append(f'  recommended:                 {recommended}')
    lines.append(f'  in-source preset:            {in_source}')
    deltas = {k: (recommended[k], in_source[k]) for k in recommended if recommended[k] != in_source.get(k)}
    if deltas:
        lines.append('  DELTAS (recommended != in-source):')
        for k, (rec, src) in deltas.items():
            lines.append(f'    {k}: AB-winner = {rec!r}  in-source = {src!r}')
    else:
        lines.append('  in-source matches AB winners for every probed knob.')

    with capsys.disabled():
        print('\n'.join(lines))
