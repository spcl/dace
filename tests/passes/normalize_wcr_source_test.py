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
    return [
        type(e.src).__name__ for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for e in st.edges()
        if e.data is not None and e.data.wcr is not None
    ]


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
    state.add_memlet_path(src_read, me, nnode, dst_conn='_in', memlet=dace.Memlet(data='src', subset='i'))
    state.add_memlet_path(nnode,
                          mx,
                          acc_write,
                          src_conn='_out',
                          memlet=dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
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
    assert 'NestedSDFG' not in post, (f'pass should remove all NSDFG-source WCR edges; got {post}')
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
    assert np.isclose(acc[0], expected), (f'WCR ``{wcr_str}`` returned {acc[0]}, expected {expected}')


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
    st.add_memlet_path(t,
                       mx,
                       acc_write,
                       src_conn='_out',
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
    st.add_memlet_path(priv, mx, acc_write, memlet=dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
    sdfg.validate()

    before = copy.deepcopy(sdfg.to_json())
    res = NormalizeWCRSource().apply_pass(sdfg, {})
    assert res is None, 'canonical-shape SDFG should be left alone'

    rng = np.random.default_rng(4)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1, dtype=np.float64)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum())


def _build_nsdfg_inout_wcr_sdfg() -> dace.SDFG:
    """Hand-crafted outer SDFG whose body has a NestedSDFG with an InOut
    connector ``out`` and a WCR-bearing edge leaving on that same connector.

    The shape is the post-canonicalize residue of a per-iteration RMW
    (``out[i] += src[i]``) under a per-iteration guard: the body becomes a
    NestedSDFG, the accumulator ``out`` is read in on one connector AND
    written out on the SAME connector (InOut), and AugAssignToWCR has tagged
    the outgoing edge with WCR. The shape stands on its own without depending
    on any other pass producing it.
    """
    sdfg = dace.SDFG('nsdfg_inout_wcr')
    sdfg.add_array('out', [4], dace.float64)
    sdfg.add_array('src', [4], dace.float64)

    inner = dace.SDFG('loop_body')
    inner.add_array('out', [1], dace.float64)
    inner.add_array('src', [1], dace.float64)
    istate = inner.add_state('s', is_start_block=True)
    t = istate.add_tasklet('add', {'_o', '_s'}, {'_r'}, '_r = _o + _s')
    or_in = istate.add_read('out')
    sr = istate.add_read('src')
    ow = istate.add_write('out')
    istate.add_edge(or_in, None, t, '_o', dace.Memlet('out[0]'))
    istate.add_edge(sr, None, t, '_s', dace.Memlet('src[0]'))
    istate.add_edge(t, '_r', ow, None, dace.Memlet('out[0]'))

    state = sdfg.add_state('s', is_start_block=True)
    me, mx = state.add_map('m', dict(i='0:4'))
    n = state.add_nested_sdfg(inner, {'out', 'src'}, {'out'})
    out_in = state.add_read('out')
    out_out = state.add_write('out')
    src_in = state.add_read('src')
    state.add_memlet_path(out_in, me, n, dst_conn='out', memlet=dace.Memlet('out[i]'))
    state.add_memlet_path(src_in, me, n, dst_conn='src', memlet=dace.Memlet('src[i]'))
    state.add_memlet_path(n, mx, out_out, src_conn='out', memlet=dace.Memlet('out[i]', wcr='lambda a, b: a + b'))
    sdfg.validate()
    return sdfg


def test_skips_rewrite_when_nsdfg_output_is_also_inout_connector():
    """``NormalizeWCRSource`` must not rewrite a WCR-bearing edge whose source
    is a ``NestedSDFG`` and whose ``src_conn`` is ALSO an in-connector on that
    NestedSDFG (i.e. the connector is InOut).

    Pre-fix the pass inserted ``_wcr_priv_<nsdfg>_<conn>`` only on the OUT side
    of the connector, leaving the IN side feeding the original AccessNode. That
    broke the InOut invariant (validation: ``Inout connector X is connected to
    different input ({X}) and output ({_wcr_priv_..._X}) arrays``) and made the
    surrounding pipeline crash.

    With the skip predicate the WCR stays on the direct
    ``NestedSDFG -> AccessNode`` edge; codegen falls back to its atomic-add
    path. The SDFG validates, compiles, and computes the right value."""
    from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource

    sdfg = _build_nsdfg_inout_wcr_sdfg()

    # Sanity: the input really has a NestedSDFG with an InOut connector on a
    # WCR-bearing outgoing edge. If the SDFG-API stopped producing this shape
    # the test would silently degenerate to a no-op without this guard.
    inout_pairs = []
    for n_node, _ in sdfg.all_nodes_recursive():
        if isinstance(n_node, dace.nodes.NestedSDFG):
            shared = set(n_node.in_connectors) & set(n_node.out_connectors)
            if shared:
                inout_pairs.append(shared)
    assert inout_pairs, 'expected at least one NestedSDFG with InOut connectors'

    # Pre-fix this raises InvalidSDFGNodeError("Inout connector out is
    # connected to different input ({'out'}) and output
    # ({'_wcr_priv_loop_body_out'}) arrays"). Post-fix the pass leaves the
    # InOut edge intact.
    NormalizeWCRSource().apply_pass(sdfg, {})
    sdfg.validate()

    rng = np.random.default_rng(42)
    src = rng.standard_normal(4)
    out_buf = np.zeros(4)
    sdfg(out=out_buf, src=src)
    assert np.allclose(out_buf, src), f'got {out_buf}, expected {src}'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
