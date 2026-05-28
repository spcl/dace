# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.normalize_wcr_source.NormalizeWCRSource`.

The pass maintains the invariant that every WCR-bearing edge in the SDFG sources from
an :class:`~dace.nodes.AccessNode` (the canonical reduction shape DaCe codegen
recognises). Tasklet- and NestedSDFG-sourced WCR get a freshly inserted private
transient between producer and consumer.
"""
import copy
import functools

import numpy as np
import pytest

import dace
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource


def _wcr_source_classes(sdfg: dace.SDFG):
    return [type(e.src).__name__
            for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states()
            for e in st.edges() if e.data is not None and e.data.wcr is not None]


def _build_nsdfg_wcr_sum(n: int) -> dace.SDFG:
    """``acc[0] = sum(src)`` with the per-iteration body wrapped in a NestedSDFG (the
    vectorization-pipeline shape) and the WCR placed directly on the NSDFG output edge.

    Codegen drops the reduction on this shape; the pass normalises it.
    """
    sdfg = dace.SDFG(f'nsdfg_wcr_sum_n{n}')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    state = sdfg.add_state('m')
    src_read = state.add_read('src')
    acc_write = state.add_write('acc')

    body = dace.SDFG('body')
    body.add_array('_in', [1], dace.float64)
    body.add_array('_out', [1], dace.float64)
    bstate = body.add_state('b')
    bin = bstate.add_read('_in')
    bout = bstate.add_write('_out')
    bstate.add_nedge(bin, bout, dace.Memlet(data='_in', subset='0', other_subset='0'))

    me, mx = state.add_map('m', {'i': f'0:{n}'})
    nnode = state.add_nested_sdfg(body, {'_in'}, {'_out'})
    state.add_memlet_path(src_read, me, nnode, dst_conn='_in',
                          memlet=dace.Memlet(data='src', subset='i'))
    state.add_memlet_path(nnode, mx, acc_write, src_conn='_out',
                          memlet=dace.Memlet(data='acc', subset='0',
                                              wcr='lambda a, b: a + b'))
    sdfg.validate()
    return sdfg


def test_pass_rewrites_nsdfg_source_to_access_node():
    """After the pass, no WCR-bearing edge has a NestedSDFG source; the producer-to-
    private-AN edge is plain (no WCR), and the new AN-to-consumer edge carries the
    reduction.
    """
    sdfg = _build_nsdfg_wcr_sum(64)
    pre = _wcr_source_classes(sdfg)
    assert 'NestedSDFG' in pre, f'fixture should have NSDFG-source WCR; got {pre}'

    NormalizeWCRSource().apply_pass(sdfg, {})

    post = _wcr_source_classes(sdfg)
    assert 'NestedSDFG' not in post, (
        f'pass should remove all NSDFG-source WCR edges; got {post}')
    assert post, 'pass should preserve at least one WCR edge per reduction'
    bad = [c for c in post if c not in ('AccessNode', 'MapExit')]
    assert not bad, f'WCR sources must be AccessNode/MapExit only; got {bad}'


def test_pass_rewrite_is_numerically_correct():
    """The post-pass SDFG produces the correct sum (the bug manifests as roughly the
    last thread's value, orders of magnitude off)."""
    n = 64
    sdfg = _build_nsdfg_wcr_sum(n)
    NormalizeWCRSource().apply_pass(sdfg, {})

    rng = np.random.default_rng(0)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1, dtype=np.float64)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum()), f'got {acc[0]}, expected {src.sum()}'


def test_pass_preserves_initial_accumulator_value():
    """The pre-Map value of ``acc[0]`` is preserved (WCR accumulates into it)."""
    n = 32
    sdfg = _build_nsdfg_wcr_sum(n)
    NormalizeWCRSource().apply_pass(sdfg, {})

    rng = np.random.default_rng(1)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.array([10.0])
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], 10.0 + src.sum())


@pytest.mark.parametrize(
    'wcr_str,binop,init,domain',
    [
        ('lambda a, b: a + b', lambda a, b: a + b, 0.0, (-1.0, 1.0)),
        ('lambda a, b: a * b', lambda a, b: a * b, 1.0, (0.95, 1.05)),
        ('lambda a, b: max(a, b)', max, 0.0, (-1.0, 1.0)),
        ('lambda a, b: min(a, b)', min, 0.0, (-1.0, 1.0)),
    ],
    ids=['sum', 'product', 'max', 'min'],
)
def test_pass_handles_every_associative_wcr(wcr_str, binop, init, domain):
    """The rewrite preserves the WCR string verbatim; sum / product / max / min all
    produce the correct sequential-reduction result."""
    n = 32
    sdfg = _build_nsdfg_wcr_sum(n)
    # Swap the WCR on every WCR-bearing edge before normalisation.
    for st in sdfg.all_states():
        for e in st.edges():
            if e.data is not None and e.data.wcr is not None:
                e.data.wcr = wcr_str
    NormalizeWCRSource().apply_pass(sdfg, {})

    rng = np.random.default_rng(hash(wcr_str) & 0xFFFFFFFF)
    src = rng.uniform(*domain, size=n)
    expected = functools.reduce(binop, src.tolist(), init)
    acc = np.array([init])
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], expected), (
        f'WCR ``{wcr_str}`` returned {acc[0]}, expected {expected}')


def test_pass_is_idempotent():
    """Re-running the pass finds no WCR edges sourced from a CodeNode -- it is a no-op
    on already-normalised SDFGs."""
    sdfg = _build_nsdfg_wcr_sum(16)
    first = NormalizeWCRSource().apply_pass(sdfg, {})
    assert first is not None, 'first apply should report a rewrite'
    second = NormalizeWCRSource().apply_pass(sdfg, {})
    assert second is None, f'second apply should be a no-op; got {second}'


def test_pass_rewrites_tasklet_source_wcr_too():
    """The pass also rewrites Tasklet-source WCR (the existing Tasklet path already
    works because the codegen handles scalar-typed CodeNode outputs, but the
    normalisation invariant must apply uniformly)."""
    n = 32
    sdfg = dace.SDFG(f'tasklet_wcr_n{n}')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    st = sdfg.add_state('m')
    src_read = st.add_read('src')
    acc_write = st.add_write('acc')
    me, mx = st.add_map('m', {'i': f'0:{n}'})
    t = st.add_tasklet('id', {'_in'}, {'_out'}, '_out = _in')
    st.add_memlet_path(src_read, me, t, dst_conn='_in', memlet=dace.Memlet(data='src', subset='i'))
    st.add_memlet_path(t, mx, acc_write, src_conn='_out',
                       memlet=dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
    sdfg.validate()

    NormalizeWCRSource().apply_pass(sdfg, {})

    post = _wcr_source_classes(sdfg)
    assert 'Tasklet' not in post, f'pass should remove Tasklet-source WCR; got {post}'

    rng = np.random.default_rng(7)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1, dtype=np.float64)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum())


def test_pass_no_op_on_canonical_access_node_source_wcr():
    """An SDFG that already places WCR on AccessNode -> MapExit edges is left
    unchanged (the pass only rewrites CodeNode-source WCR)."""
    n = 16
    sdfg = dace.SDFG(f'canonical_wcr_n{n}')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    sdfg.add_scalar('_priv', dace.float64, transient=True)
    st = sdfg.add_state('m')
    src_read = st.add_read('src')
    acc_write = st.add_write('acc')
    priv = st.add_access('_priv')
    me, mx = st.add_map('m', {'i': f'0:{n}'})
    t = st.add_tasklet('id', {'_in'}, {'_out'}, '_out = _in')
    st.add_memlet_path(src_read, me, t, dst_conn='_in', memlet=dace.Memlet(data='src', subset='i'))
    st.add_edge(t, '_out', priv, None, dace.Memlet(data='_priv', subset='0'))
    st.add_memlet_path(priv, mx, acc_write,
                       memlet=dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
    sdfg.validate()

    before = copy.deepcopy(sdfg.to_json())
    res = NormalizeWCRSource().apply_pass(sdfg, {})
    assert res is None, 'canonical-shape SDFG should be left alone'

    rng = np.random.default_rng(4)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1, dtype=np.float64)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum())


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
