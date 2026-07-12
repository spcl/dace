# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``PrivatizeScatterReduction`` -- the azimint-histogram scatter-reduction
privatization pass.

A data-dependent scatter accumulate ``hist[bin[i]] (+)= w[i]`` under a parallel map
must lower to an OpenMP array-section ``reduction(op:hist[0:n])`` clause (each thread
a private copy, uncontended accumulate, runtime tree-merge) instead of a contended
per-element atomic on a shared ``hist``, and must NOT lower to the per-iteration
whole-array private buffer that the generic WCR normalization would otherwise insert
(that path is both slow and -- for a straight-line scatter -- numerically wrong).

Covered:

* a float weighted histogram -- codegen has ``reduction(+:...)`` and no per-iteration
  whole-array buffer / ``Accumulate_atomic``; result is ``np.allclose`` to
  ``np.add.at`` (float reassociation across threads, so not bit-identical);
* an integer count histogram -- bit-exact (integer ``+`` is associative);
* refuse cases (non-reducible operator, self-referential accumulator, per-element
  and scalar reductions) -- the pass leaves them alone;
* the azimint_hist kernel end-to-end through ``canonicalize`` -- bit-close to numpy
  and free of the per-iteration scatter buffer.
"""
import os

import numpy as np
import pytest

import dace
from dace import nodes
from dace.transformation.passes.privatize_scatter_reduction import (PrivatizeScatterReduction, scatter_wcr_op,
                                                                    scatter_reduction_wcr_edge, SCATTER_REDUCIBLE_OPS)
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N, bins, npt = (dace.symbol(s, dtype=dace.int64) for s in ('N', 'bins', 'npt'))

# -- Kernels ------------------------------------------------------------------


@dace.program
def weighted_histogram(binidx: dace.int64[N], weights: dace.float64[N]):
    hist = np.ndarray((bins, ), dtype=np.float64)
    hist[:] = 0
    for i in dace.map[0:N]:
        hist[binidx[i]] += weights[i]
    return hist


@dace.program
def count_histogram(binidx: dace.int64[N]):
    hist = np.ndarray((bins, ), dtype=np.int64)
    hist[:] = 0
    for i in dace.map[0:N]:
        hist[binidx[i]] += 1
    return hist


def _codegen_text(sdfg: dace.SDFG) -> str:
    return "\n".join(c.code for c in sdfg.generate_code())


def _unique_build(sdfg: dace.SDFG, tag: str) -> None:
    """Give the SDFG a unique build folder so parallel test workers do not race on
    the same ``.dacecache`` directory."""
    sdfg.build_folder = os.path.join(sdfg.build_folder + f'_{tag}_{os.getpid()}')


# -- Detection unit tests -----------------------------------------------------


def test_scatter_wcr_op_recognizes_reducible_ops():
    assert scatter_wcr_op('lambda x, y: (x + y)') == '+'
    assert scatter_wcr_op('lambda x, y: (x * y)') == '*'
    assert scatter_wcr_op('lambda x, y: min(x, y)') == 'min'
    assert scatter_wcr_op('lambda x, y: max(x, y)') == 'max'
    # Non-reducible / non-associative reducers are refused.
    assert scatter_wcr_op('lambda x, y: (x - y)') is None
    assert scatter_wcr_op('lambda x, y: (x / y)') is None
    assert scatter_wcr_op('lambda x, y: (2 * x + y)') is None
    assert set(SCATTER_REDUCIBLE_OPS) == {'+', '*', 'min', 'max'}


def test_pass_fires_on_weighted_histogram():
    """On the raw frontend histogram the pass surfaces exactly one scatter reduction,
    and the surfaced ``NestedSDFG -> MapExit -> accumulator`` edge chain now carries the
    ``+`` WCR (idempotent: a second run finds nothing)."""
    sdfg = weighted_histogram.to_sdfg(simplify=True)
    n = PrivatizeScatterReduction().apply_pass(sdfg, {})
    assert n == 1
    # The accumulator's map-exit edge chain now carries a WCR.
    wcr_into_mapexit = [
        e for _, st in [(sd, s) for sd in sdfg.all_sdfgs_recursive() for s in sd.states()] for e in st.edges()
        if e.data is not None and e.data.wcr is not None and isinstance(e.dst, nodes.MapExit)
    ]
    assert wcr_into_mapexit, "expected a WCR edge surfaced onto the MapExit"
    # Idempotent second run.
    assert PrivatizeScatterReduction().apply_pass(sdfg, {}) is None


# -- Refuse cases -------------------------------------------------------------


def _build_scatter_nsdfg(wcr: str, read_accumulator: bool = False) -> dace.SDFG:
    """A minimal map-body-NestedSDFG scatter ``acc[idx[i]] (wcr)= w[i]`` for refuse
    tests. ``wcr`` is the reducer lambda; ``read_accumulator`` also wires ``acc`` as a
    map input (self-referential)."""
    sdfg = dace.SDFG('scatter_refuse')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('bins', dace.int64)
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_array('w', [N], dace.float64)
    sdfg.add_array('acc', [bins], dace.float64)
    st = sdfg.add_state('main')

    nsdfg = dace.SDFG('body')
    nsdfg.add_scalar('b_in', dace.int64)
    nsdfg.add_scalar('w_in', dace.float64)
    nsdfg.add_array('oc', [bins], dace.float64)
    nsdfg.add_symbol('bsym', dace.int64)
    nsdfg.add_scalar('b_scal', dace.int64, transient=True)
    s0 = nsdfg.add_state('s0', is_start_block=True)
    r = s0.add_read('b_in')
    t0 = s0.add_tasklet('rd', {'__b'}, {'__o'}, '__o = __b')
    a0 = s0.add_access('b_scal')
    s0.add_edge(r, None, t0, '__b', dace.Memlet(data='b_in', subset='0'))
    s0.add_edge(t0, '__o', a0, None, dace.Memlet(data='b_scal', subset='0'))
    s1 = nsdfg.add_state('s1')
    nsdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'bsym': 'b_scal'}))
    tw = s1.add_read('w_in')
    t1 = s1.add_tasklet('acc', {'__w'}, {'__o'}, '__o = __w')
    ow = s1.add_write('oc')
    s1.add_edge(tw, None, t1, '__w', dace.Memlet(data='w_in', subset='0'))
    s1.add_edge(t1, '__o', ow, None, dace.Memlet(data='oc', subset='bsym', wcr=wcr))

    me, mx = st.add_map('scatter', {'i': '0:N'}, schedule=dace.ScheduleType.CPU_Multicore)
    in_conns = {'b_in', 'w_in'}
    if read_accumulator:
        in_conns.add('acc_in')
        nsdfg.add_array('acc_in', [bins], dace.float64)
    node = st.add_nested_sdfg(nsdfg, in_conns, {'oc'}, symbol_mapping={'bins': 'bins'})
    st.add_memlet_path(st.add_read('idx'), me, node, dst_conn='b_in', memlet=dace.Memlet(data='idx', subset='i'))
    st.add_memlet_path(st.add_read('w'), me, node, dst_conn='w_in', memlet=dace.Memlet(data='w', subset='i'))
    if read_accumulator:
        st.add_memlet_path(st.add_read('acc'),
                           me,
                           node,
                           dst_conn='acc_in',
                           memlet=dace.Memlet(data='acc', subset='0:bins'))
    mx.add_in_connector('IN_acc')
    mx.add_out_connector('OUT_acc')
    st.add_edge(node, 'oc', mx, 'IN_acc', dace.Memlet(data='acc', subset='0:bins'))
    st.add_edge(mx, 'OUT_acc', st.add_write('acc'), None, dace.Memlet(data='acc', subset='0:bins'))
    return sdfg


def test_refuse_non_reducible_op():
    """A data-dependent scatter with a non-associative reducer (``-``) is left alone."""
    sdfg = _build_scatter_nsdfg('lambda x, y: (x - y)')
    assert PrivatizeScatterReduction().apply_pass(sdfg, {}) is None
    # No WCR was surfaced onto the map-exit edge chain.
    st = sdfg.node(0)
    assert all(e.data is None or e.data.wcr is None for e in st.edges() if isinstance(e.dst, nodes.MapExit))


def test_refuse_self_referential_accumulator():
    """A scatter whose map also READS the accumulator array is refused (a whole-buffer
    privatization would make those reads see the private identity copy)."""
    sdfg = _build_scatter_nsdfg('lambda x, y: (x + y)', read_accumulator=True)
    assert PrivatizeScatterReduction().apply_pass(sdfg, {}) is None


def test_refuse_scalar_reduction():
    """A plain scalar accumulator (``s += a[i]``, not a scatter into a bounded array)
    is NormalizeWCR's job, not this pass's."""

    @dace.program
    def dot(a: dace.float64[N], b: dace.float64[N]):
        s = np.float64(0)
        for i in dace.map[0:N]:
            s += a[i] * b[i]
        return s

    sdfg = dot.to_sdfg(simplify=True)
    assert PrivatizeScatterReduction().apply_pass(sdfg, {}) is None


# -- End-to-end: codegen shape + numerical correctness ------------------------


def test_weighted_histogram_codegen_and_values():
    """Float histogram: codegen privatizes via ``reduction(+:...)`` with no per-iteration
    whole-array buffer / ``Accumulate_atomic``; result matches ``np.add.at``."""
    sdfg = weighted_histogram.to_sdfg(simplify=True)
    _unique_build(sdfg, 'whist')
    canonicalize(sdfg, validate=True, target='cpu')
    code = _codegen_text(sdfg)
    assert 'reduction(+:' in code, "expected an OpenMP array-section reduction clause"
    # No whole-array atomic copy and no per-iteration whole-array reduction buffer
    # (both are the markers of the generic-normalization mangling this pass avoids;
    # the harmless scalar ``_wcr_priv_assign_*`` temp for the inner WCR may remain).
    assert 'Accumulate_atomic' not in code, "scatter must not use a whole-array atomic copy"
    assert '_nnr_out' not in code, "scatter must not be wrapped in a per-iteration whole-array buffer"

    rng = np.random.default_rng(0)
    n, nb = 20000, 37
    binidx = rng.integers(0, nb, n).astype(np.int64)
    weights = rng.random(n)
    res = sdfg(binidx=binidx, weights=weights, N=n, bins=nb)
    ref = np.zeros(nb)
    np.add.at(ref, binidx, weights)
    assert np.allclose(res, ref), f"maxerr={np.max(np.abs(res - ref))}"


def test_count_histogram_bit_exact():
    """Integer count histogram: integer ``+`` is associative, so the privatized reduction
    is bit-exact vs ``np.add.at``."""
    sdfg = count_histogram.to_sdfg(simplify=True)
    _unique_build(sdfg, 'chist')
    canonicalize(sdfg, validate=True, target='cpu')
    code = _codegen_text(sdfg)
    assert 'reduction(+:' in code

    rng = np.random.default_rng(1)
    n, nb = 20000, 29
    binidx = rng.integers(0, nb, n).astype(np.int64)
    res = sdfg(binidx=binidx, N=n, bins=nb)
    ref = np.zeros(nb, dtype=np.int64)
    np.add.at(ref, binidx, 1)
    assert np.array_equal(res, ref)


def test_knob_off_leaves_scatter_unprivatized():
    """With the knob off the array-section reduction is not surfaced for the scatter (the
    pass is what introduces it). The fail-safe refuse-guard in ``NormalizeWCR`` means the
    scatter is ALSO not mangled into the unsound per-iteration whole-array ``_nnr_out``
    buffer -- it falls back to the correct per-element atomic, so the count histogram stays
    bit-exact (this path used to silently miscompile via the whole-buffer rewrite)."""
    sdfg = count_histogram.to_sdfg(simplify=True)
    _unique_build(sdfg, 'knoboff')
    canonicalize(sdfg, validate=True, target='cpu', privatize_scatter_reductions=False)
    code = _codegen_text(sdfg)
    assert 'reduction(+:' not in code, 'knob off must not surface the array-section reduction'
    assert '_nnr_out' not in code, 'the scatter must not be mangled into a whole-array _nnr_out buffer'

    rng = np.random.default_rng(4)
    n, nb = 20000, 29
    binidx = rng.integers(0, nb, n).astype(np.int64)
    res = sdfg(binidx=binidx, N=n, bins=nb)
    ref = np.zeros(nb, dtype=np.int64)
    np.add.at(ref, binidx, 1)
    assert np.array_equal(res, ref)


# -- azimint_hist end-to-end --------------------------------------------------


@dace.program
def _get_bin_edges(a: dace.float64[N], bin_edges: dace.float64[bins + 1]):
    a_min = np.amin(a)
    a_max = np.amax(a)
    delta = (a_max - a_min) / bins
    for i in dace.map[0:bins]:
        bin_edges[i] = a_min + i * delta
    bin_edges[bins] = a_max


@dace.program
def _compute_bin(x: dace.float64, bin_edges: dace.float64[bins + 1]):
    a_min = bin_edges[0]
    a_max = bin_edges[bins]
    return dace.int64(bins * (x - a_min) / (a_max - a_min))


@dace.program
def _histogram(a: dace.float64[N], bin_edges: dace.float64[bins + 1]):
    hist = np.ndarray((bins, ), dtype=np.int64)
    hist[:] = 0
    _get_bin_edges(a, bin_edges)
    for i in dace.map[0:N]:
        b = min(_compute_bin(a[i], bin_edges), bins - 1)
        hist[b] += 1
    return hist


@dace.program
def _histogram_weights(a: dace.float64[N], bin_edges: dace.float64[bins + 1], weights: dace.float64[N]):
    hist = np.ndarray((bins, ), dtype=weights.dtype)
    hist[:] = 0
    _get_bin_edges(a, bin_edges)
    for i in dace.map[0:N]:
        b = min(_compute_bin(a[i], bin_edges), bins - 1)
        hist[b] += weights[i]
    return hist


@dace.program
def azimint_hist(data: dace.float64[N], radius: dace.float64[N]):
    bin_edges_u = np.ndarray((npt + 1, ), dtype=np.float64)
    histu = _histogram(radius, bin_edges_u)
    bin_edges_w = np.ndarray((npt + 1, ), dtype=np.float64)
    histw = _histogram_weights(radius, bin_edges_w, data)
    return histw / histu


def test_azimint_hist_end_to_end():
    """azimint_hist through ``canonicalize`` (knob on): both scatter histograms are
    privatized (array-section reductions, no per-iteration buffers) and the result is
    bit-close to numpy's ``np.histogram``."""
    nv, nptv = 30000, 80
    rng = np.random.default_rng(42)
    data, radius = rng.random((nv, )), rng.random((nv, ))

    sdfg = azimint_hist.to_sdfg(simplify=True)
    _unique_build(sdfg, 'azimint')
    canonicalize(sdfg, validate=True, target='cpu')
    code = _codegen_text(sdfg)
    assert 'reduction(+:' in code
    assert 'Accumulate_atomic' not in code, "no per-iteration whole-array scatter accumulate"

    val = sdfg(data=data, radius=radius, N=nv, npt=nptv)
    histu = np.histogram(radius, nptv)[0]
    histw = np.histogram(radius, nptv, weights=data)[0]
    ref = histw / histu
    assert np.allclose(val, ref), f"maxrelerr={np.max(np.abs(val - ref)) / (np.max(np.abs(ref)) + 1e-30)}"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
