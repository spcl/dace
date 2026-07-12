# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Focused unit tests for the :class:`~dace.transformation.passes.vectorization.widen_accesses.WidenAccesses`
reduction-copyback lowering (Step 0) and the scalar-target ``other_subset`` guard.

A masked map reduction (``if c: acc op= x``) reaches ``WidenAccesses`` as ``NormalizeWCR``'s seeded
body-local accumulator plus a PLAIN copyback ``priv[0] -> oc[0]`` into a write-only output connector
whose boundary edge carries the reduction WCR. Left as a plain copy, widening ``priv`` to a tile
would over-widen the scalar sink ``oc`` (``other_subset`` -> ``[0:W]`` on a shape-``(1,)`` array).
Step 0 rewrites the copyback into a ``reduce_accum`` fold so it lowers to a horizontal ``TileReduce``
and the scalar sink stays un-widened."""
import dace
from dace import subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses

N = dace.symbol('N')


def _make_seeded_reduction_body() -> "tuple[dace.SDFG, dace.SDFGState, nodes.NestedSDFG, dace.SDFG]":
    """A body NSDFG holding a seeded reduction: transient ``priv`` copied (plain) into the
    write-only output connector ``oc``, whose boundary is ``NSDFG[oc] -> acc -[wcr:+]-> acc2`` (the
    ``_boundary_reduction_wcr`` walk needs only the WCR one hop past the connector -- no Map)."""
    inner = dace.SDFG('body')
    inner.add_scalar('priv', dace.float64, transient=True)
    inner.add_scalar('oc', dace.float64, transient=False)  # write-only output connector
    st = inner.add_state('copyback', is_start_block=True)
    p = st.add_access('priv')
    o = st.add_access('oc')
    st.add_nedge(p, o, Memlet(data='priv', subset=subsets.Range([(0, 0, 1)]),
                              other_subset=subsets.Range([(0, 0, 1)])))

    outer = dace.SDFG('outer')
    outer.add_scalar('acc', dace.float64, transient=True)
    outer.add_array('res', [1], dace.float64)
    state = outer.add_state('s', is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {}, {'oc'})
    acc = state.add_access('acc')
    res = state.add_access('res')
    # NSDFG[oc] -> acc -[wcr:+]-> res: the reduction op rides the edge one hop past the connector.
    state.add_edge(nsdfg, 'oc', acc, None, Memlet('acc[0]'))
    wr = Memlet('res[0]')
    wr.wcr = 'lambda x, y: x + y'
    state.add_edge(acc, None, res, None, wr)
    return outer, state, nsdfg, inner


def test_reduction_copyback_lowered_to_fold():
    """The plain ``priv -> oc`` copyback becomes a ``reduce_accum`` tasklet ``oc = oc + priv``."""
    outer, state, nsdfg, inner = _make_seeded_reduction_body()
    n = WidenAccesses(widths=(8, ))._lower_reduction_copybacks(state, nsdfg, inner)
    assert n == 1, 'the seeded reduction copyback should be rewritten to a fold'
    body_state = next(s for s in inner.states() if s.label == 'copyback')
    tasklets = [t for t in body_state.nodes() if isinstance(t, nodes.Tasklet)]
    assert len(tasklets) == 1 and tasklets[0].label == 'reduce_accum'
    t = tasklets[0]
    assert t.in_connectors.keys() == {'__in1', '__in2'} and t.out_connectors.keys() == {'__out'}
    assert '__in1 + __in2' in t.code.as_string
    # ``oc`` is written by the fold; the plain AN -> AN copy is gone.
    assert any(isinstance(e.dst, nodes.AccessNode) and e.dst.data == 'oc' and e.src is t
               for e in body_state.edges())
    assert not any(isinstance(e.src, nodes.AccessNode) and e.src.data == 'priv'
                   and isinstance(e.dst, nodes.AccessNode) for e in body_state.edges())


def test_no_boundary_wcr_leaves_copyback_untouched():
    """Without a boundary reduction WCR the copyback is NOT a reduction fold -- leave it alone."""
    outer, state, nsdfg, inner = _make_seeded_reduction_body()
    # Strip the WCR from the boundary edge.
    for e in state.edges():
        if e.data is not None and e.data.wcr is not None:
            e.data.wcr = None
    n = WidenAccesses(widths=(8, ))._lower_reduction_copybacks(state, nsdfg, inner)
    assert n == 0, 'a non-reduction copyback must not be rewritten'


def test_other_endpoint_widens_guard():
    """``_other_endpoint_widens`` keeps a single-element scalar sink un-widened but widens a
    tile-bound endpoint."""
    sd = dace.SDFG('g')
    sd.add_array('priv', [8], dace.float64, transient=True)
    sd.add_scalar('scalar_sink', dace.float64, transient=True)
    sd.add_array('tileB', [8], dace.float64, transient=True)
    st = sd.add_state()
    p = st.add_access('priv')
    s = st.add_access('scalar_sink')
    b = st.add_access('tileB')
    e_scalar = st.add_nedge(p, s, Memlet(data='priv', subset=subsets.Range([(0, 7, 1)]),
                                         other_subset=subsets.Range([(0, 0, 1)])))
    e_tile = st.add_nedge(p, b, Memlet(data='priv', subset=subsets.Range([(0, 7, 1)]),
                                       other_subset=subsets.Range([(0, 7, 1)])))
    # scalar_sink stays single-element (not in the widen set) -> do NOT widen its other_subset.
    assert WidenAccesses._other_endpoint_widens(e_scalar, 'priv', sd, to_widen={'priv'}) is False
    # tileB is itself being widened -> widen its other_subset symmetrically.
    assert WidenAccesses._other_endpoint_widens(e_tile, 'priv', sd, to_widen={'priv', 'tileB'}) is True


if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main([__file__, '-v']))
