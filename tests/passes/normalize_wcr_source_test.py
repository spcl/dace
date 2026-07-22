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


def _seed_states(sdfg: dace.SDFG):
    return [st.label for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() if '_wcr_seed' in st.label]


def test_seed_fresh_write_once_accumulator():
    """A write-once WCR (a distinct accumulator slot per map iteration -- a spurious
    per-element assignment expressed as a WCR) into a *fresh transient* with no plain
    initializer is identity-seeded, so codegen's ``acc = acc OP val`` read-back
    (``0 + val``) starts defined instead of reading uninitialized memory."""
    n = 16
    sdfg = dace.SDFG('seed_write_once')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('out', [n], dace.float64)
    sdfg.add_transient('acc', [n], dace.float64)
    st = sdfg.add_state('m')
    sr = st.add_read('src')
    aw = st.add_access('acc')
    me, mx = st.add_map('m', {'i': f'0:{n}'})
    t = st.add_tasklet('id', {'_in'}, {'_out'}, '_out = _in')
    st.add_memlet_path(sr, me, t, dst_conn='_in', memlet=dace.Memlet('src[i]'))
    st.add_memlet_path(t, mx, aw, src_conn='_out', memlet=dace.Memlet(data='acc', subset='i', wcr='lambda a, b: a + b'))
    st2 = sdfg.add_state_after(st, 'c')
    st2.add_nedge(st2.add_read('acc'), st2.add_write('out'), dace.Memlet(f'acc[0:{n}]'))
    sdfg.validate()

    assert not _seed_states(sdfg)
    NormalizeWCRSource().apply_pass(sdfg, {})
    assert _seed_states(sdfg), 'fresh write-once WCR accumulator must be identity-seeded'

    rng = np.random.default_rng(0)
    src = rng.uniform(-1.0, 1.0, size=n)
    out = np.empty(n)
    sdfg(src=src, out=out)
    assert np.allclose(out, src), f'write-once acc[i] = 0 + src[i]; got {out}'


def test_seed_spares_top_level_argument():
    """An in-place write-once WCR onto a top-level (non-transient) argument -- e.g.
    gramschmidt's ``A[:,j] -= ...`` -- must NOT be seeded: the argument carries the
    caller's value, which an identity fill would destroy."""
    n = 16
    sdfg = dace.SDFG('inplace_arg')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [n], dace.float64)  # non-transient argument, updated in place
    st = sdfg.add_state('m')
    sr = st.add_read('src')
    aw = st.add_write('acc')
    me, mx = st.add_map('m', {'i': f'0:{n}'})
    t = st.add_tasklet('id', {'_in'}, {'_out'}, '_out = _in')
    st.add_memlet_path(sr, me, t, dst_conn='_in', memlet=dace.Memlet('src[i]'))
    st.add_memlet_path(t, mx, aw, src_conn='_out', memlet=dace.Memlet(data='acc', subset='i', wcr='lambda a, b: a + b'))
    sdfg.validate()

    NormalizeWCRSource().apply_pass(sdfg, {})
    assert not _seed_states(sdfg), 'in-place WCR onto a top-level argument must not be seeded'

    rng = np.random.default_rng(3)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.full(n, 5.0)
    sdfg(src=src, acc=acc)
    assert np.allclose(acc, 5.0 + src), 'the caller-supplied prior must be preserved'


def test_seed_spares_same_slot_fold():
    """A same-slot fold (a constant, non-map-indexed subset -- a genuine reduction that
    continues a prior value, like nussinov's ``_priv_table`` seeded through a separate
    ``table`` input) must NOT be blindly reset. The pass leaves it unseeded so the prior
    is preserved."""
    n = 16
    sdfg = dace.SDFG('fold_prior')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)  # non-transient: caller supplies the prior
    st = sdfg.add_state('m')
    sr = st.add_read('src')
    aw = st.add_write('acc')
    me, mx = st.add_map('m', {'i': f'0:{n}'})
    t = st.add_tasklet('id', {'_in'}, {'_out'}, '_out = _in')
    st.add_memlet_path(sr, me, t, dst_conn='_in', memlet=dace.Memlet('src[i]'))
    st.add_memlet_path(t, mx, aw, src_conn='_out', memlet=dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
    sdfg.validate()

    NormalizeWCRSource().apply_pass(sdfg, {})
    assert not _seed_states(sdfg), 'same-slot fold onto a live accumulator must not be seeded'

    src = np.ones(n)
    acc = np.array([10.0])
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], 10.0 + n), f'prior + sum(src); got {acc[0]}'


def test_seed_spares_nested_out_only_aliasing_live_array():
    """A nested-SDFG out-only connector bound (plain outer edge) to a LIVE non-transient array
    is not fresh storage: the in-place reduction accumulates onto the caller's data, so seeding
    it to identity would erase the live value. The pass must leave it unseeded."""
    n = 8
    sdfg = dace.SDFG('inplace_nested')
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('val', [n], dace.float64)
    st = sdfg.add_state('s')
    body = dace.SDFG('body')
    body.add_array('out', [n], dace.float64)
    body.add_array('bval', [n], dace.float64)
    bst = body.add_state('b')
    me, mx = bst.add_map('m', {'i': f'0:{n}'})
    t = bst.add_tasklet('f', {'__v'}, {'__o'}, '__o = __v')
    bst.add_memlet_path(bst.add_read('bval'), me, t, dst_conn='__v', memlet=dace.Memlet('bval[i]'))
    bst.add_memlet_path(t,
                        mx,
                        bst.add_write('out'),
                        src_conn='__o',
                        memlet=dace.Memlet(data='out', subset='i', wcr='lambda a, b: a + b'))
    nsdfg = st.add_nested_sdfg(body, {'bval'}, {'out'})
    st.add_edge(st.add_read('val'), None, nsdfg, 'bval', dace.Memlet(f'val[0:{n}]'))
    st.add_edge(nsdfg, 'out', st.add_write('A'), None, dace.Memlet(f'A[0:{n}]'))  # plain outer edge: out -> live A
    sdfg.validate()

    NormalizeWCRSource().apply_pass(sdfg, {})
    assert not _seed_states(sdfg), 'an out-only connector aliasing a live array must not be seeded'

    A = np.full(n, 10.0)
    val = np.arange(1, n + 1, dtype=np.float64)
    sdfg(A=A, val=val)
    assert np.allclose(A, 10.0 + val), f'A must be accumulated in place (10 + val), not overwritten; got {A}'


def test_seed_spares_source_oriented_plain_init():
    """A plain initializer written SOURCE-oriented -- a ``read(bias) -> write(acc)`` copy whose
    memlet data is ``bias`` -- still initializes ``acc``. The pass must detect it (keying the
    plain-writer scan on the edge DESTINATION, not the memlet data) and leave the accumulator
    unseeded, so the init survives."""
    n = 8
    sdfg = dace.SDFG('src_oriented_init')
    sdfg.add_array('bias', [n], dace.float64)
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('out', [n], dace.float64)
    sdfg.add_transient('acc', [n], dace.float64)
    s1 = sdfg.add_state('init')  # acc <- bias, source-oriented (memlet.data == 'bias')
    s1.add_edge(s1.add_read('bias'), None, s1.add_write('acc'), None, dace.Memlet(f'bias[0:{n}]'))
    s2 = sdfg.add_state_after(s1, 'accum')  # acc[i] (wcr+)= src[i]
    me, mx = s2.add_map('m', {'i': f'0:{n}'})
    t = s2.add_tasklet('f', {'__s'}, {'__o'}, '__o = __s')
    s2.add_memlet_path(s2.add_read('src'), me, t, dst_conn='__s', memlet=dace.Memlet('src[i]'))
    s2.add_memlet_path(t,
                       mx,
                       s2.add_access('acc'),
                       src_conn='__o',
                       memlet=dace.Memlet(data='acc', subset='i', wcr='lambda a, b: a + b'))
    s3 = sdfg.add_state_after(s2, 'copyout')
    s3.add_edge(s3.add_read('acc'), None, s3.add_write('out'), None, dace.Memlet(f'acc[0:{n}]'))
    sdfg.validate()

    NormalizeWCRSource().apply_pass(sdfg, {})
    assert not _seed_states(sdfg), 'acc already has a (source-oriented) plain init; must not be re-seeded'

    rng = np.random.default_rng(2)
    bias = rng.uniform(-1.0, 1.0, size=n)
    src = rng.uniform(-1.0, 1.0, size=n)
    out = np.empty(n)
    sdfg(bias=bias, src=src, out=out)
    assert np.allclose(out, bias + src), f'seeding would have dropped bias; got {out}'


def _build_nsdfg_accumulating_connector_sdfg(n: int) -> dace.SDFG:
    """Hand-crafted outer SDFG whose NestedSDFG body WCR-*accumulates* into its own
    write-only output connector over a per-iteration DYNAMIC subrange.

    The shape is the post-canonicalize residue of the ICON-style
    ``for i: for k in range(beg, end): out[i, k] += 1.0`` (a nest whose inner bounds
    derive from the outer iterator): the ``i`` map body becomes a NestedSDFG whose
    ``acc`` output connector is an ``[n]`` array bound to ``out[i, 0:n]``, and the
    inner per-element ``acc[k] (+)= 1.0`` only ever touches ``k in [i, i+2)``.

    The shape stands on its own without depending on any other pass producing it.
    """
    inner = dace.SDFG('acc_body')
    inner.add_array('acc', [n], dace.float64)
    istate = inner.add_state('s', is_start_block=True)
    ime, imx = istate.add_map('k', dict(k='i:i+2'))
    t = istate.add_tasklet('one', {}, {'_r'}, '_r = 1.0')
    aw = istate.add_write('acc')
    istate.add_edge(ime, None, t, None, dace.Memlet())
    istate.add_memlet_path(t, imx, aw, src_conn='_r', memlet=dace.Memlet(data='acc', subset='k',
                                                                        wcr='lambda a, b: a + b'))

    sdfg = dace.SDFG(f'nsdfg_accumulating_connector_n{n}')
    sdfg.add_array('out', [n, n], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    me, mx = state.add_map('m', dict(i=f'0:{n - 1}'))
    nnode = state.add_nested_sdfg(inner, {}, {'acc'}, symbol_mapping=dict(i='i'))
    out_w = state.add_write('out')
    state.add_edge(me, None, nnode, None, dace.Memlet())
    state.add_memlet_path(nnode,
                          mx,
                          out_w,
                          src_conn='acc',
                          memlet=dace.Memlet(data='out', subset=f'i, 0:{n}', wcr='lambda a, b: a + b'))
    sdfg.validate()
    return sdfg


def _accumulating_connector_oracle(n: int) -> np.ndarray:
    exp = np.zeros((n, n))
    for i in range(n - 1):
        for k in range(i, i + 2):
            exp[i, k] += 1.0
    return exp


def test_skips_rewrite_when_nsdfg_wcr_accumulates_into_its_output_connector():
    """``NormalizeWCRSource`` must not re-home an output connector that the body
    WCR-*accumulates* into onto a whole-array ``_wcr_priv`` buffer.

    The rewrite copies the WHOLE private buffer out to the consumer, which is only
    sound when the producer PLAIN-writes every element of it. Here the body instead
    folds into the connector (``acc[k] (+)= 1.0``) over a per-iteration dynamic
    subrange, so the buffer IS an accumulator -- and the pass seeds only the WCR
    *destination*, never the private buffer. Codegen emits it as a bare per-iteration
    ``new double[...]`` (uninitialized), so the body accumulates into garbage and the
    whole-buffer copy folds every slot -- including the ones no iteration wrote --
    into the target.

    In the ICON compound nest this silently miscompiled ``for k in range(beg, end):
    out[i, k, 0] += 1.0`` into an ``out[i] += sum(out[:i])`` prefix sum, because
    ``delete[]``/``new[]`` of the same size hands the same block back with the previous
    iteration's total still in it. That numeric symptom is allocator-dependent, so this
    test pins the deterministic *structural* contract instead (the end-to-end value gate
    is ``tests/canonicalize/canonicalize_compound_nest_test.py``): the accumulating
    connector keeps its direct ``NestedSDFG -> MapExit`` WCR edge, so the inner
    per-element WCR accumulates straight onto the target -- which is already correct.
    """
    from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource

    n = 4
    sdfg = _build_nsdfg_accumulating_connector_sdfg(n)
    assert 'NestedSDFG' in _wcr_source_classes(sdfg), 'fixture should have NSDFG-source WCR'

    NormalizeWCRSource().apply_pass(sdfg, {})
    sdfg.validate()

    # Structural: the accumulating connector must keep its direct NestedSDFG-sourced WCR
    # edge, and no WHOLE-ARRAY private buffer may be minted for it -- that buffer is
    # precisely the miscompile. The body's own inner ``acc[k] (+)= 1.0`` tasklet edge is
    # still normalized onto a scalar ``_wcr_priv`` (the rewrite this pass exists for), so
    # only multi-element buffers are disqualifying.
    assert 'NestedSDFG' in _wcr_source_classes(sdfg), 'accumulating connector must keep its direct WCR edge'
    wide = [
        nm for sd in sdfg.all_sdfgs_recursive() for nm, d in sd.arrays.items()
        if nm.startswith('_wcr_priv') and d.total_size != 1
    ]
    assert not wide, f'accumulating output connector must not be re-homed onto a whole-array buffer; got {wide}'

    # Value-preservation: the refused shape still computes the right answer, bit-exact.
    out = np.zeros((n, n))
    sdfg(out=out)
    exp = _accumulating_connector_oracle(n)
    assert np.array_equal(out, exp), f'refused shape must stay value-preserving; got\n{out}\nwant\n{exp}'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
