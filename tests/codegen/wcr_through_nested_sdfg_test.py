# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""WCR codegen must work when the WCR edge sources from a NestedSDFG, not a Tasklet.

Some transformation pipelines (e.g. vectorization's "normalize the map body into a
NestedSDFG") move the per-iteration computation inside a NestedSDFG before the Map
exits. The WCR edge that previously ran ``tasklet -> MapExit -> acc[c]`` now runs
``NSDFG -> MapExit -> acc[c]`` -- same conflict resolution semantics, different
upstream node class. The codegen must still emit the reduction (atomic or scalar)
correctly in that case; if it special-cases ``Tasklet`` as the only WCR source,
the NSDFG path silently degenerates to a plain store and the parallel result is
wrong.

The shape exercised here mirrors what ``AugAssignToWCR`` + a normalising pass
would leave behind: a parallel Map whose body is a small NestedSDFG that writes
to a scalar carry, and the carry value is funneled out via a WCR-sum edge into a
1-element accumulator in the outer SDFG.
"""
import numpy as np
import pytest

import dace


def _build_wcr_nsdfg_sdfg(n: int) -> dace.SDFG:
    """Build ``acc[0] = sum(src[i] for i in range(n))`` as a Map + WCR write whose
    source is a NestedSDFG (one scalar in, one scalar out per Map iteration)."""
    sdfg = dace.SDFG(f'wcr_through_nsdfg_n{n}')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    state = sdfg.add_state('map')
    src_read = state.add_read('src')
    acc_write = state.add_write('acc')

    # The body NestedSDFG: read ``_in`` (a scalar slice of ``src``) and produce ``_out``
    # (a scalar). The body's only computation is ``_out = _in``; the actual reduction
    # semantics live on the WCR edge from this NSDFG's output to the outer ``acc``.
    nsdfg = dace.SDFG('body')
    nsdfg.add_array('_in', [1], dace.float64)
    nsdfg.add_array('_out', [1], dace.float64)
    nstate = nsdfg.add_state('s0')
    nin = nstate.add_read('_in')
    nout = nstate.add_write('_out')
    nstate.add_nedge(nin, nout, dace.Memlet(data='_in', subset='0', other_subset='0'))

    map_entry, map_exit = state.add_map('outer', {'i': f'0:{n}'})
    nnode = state.add_nested_sdfg(nsdfg, {'_in'}, {'_out'})

    state.add_memlet_path(src_read, map_entry, nnode, dst_conn='_in', memlet=dace.Memlet(data='src', subset='i'))
    # The WCR edges: from the NSDFG output, through MapExit, to the outer ``acc[0]``.
    # ``wcr='lambda a, b: a + b'`` should make the codegen emit a sum reduction --
    # set on BOTH the inner (NSDFG -> MapExit) and outer (MapExit -> AccessNode) edges
    # so propagation and codegen both see the WCR semantics regardless of where the
    # analysis picks the edge up.
    state.add_memlet_path(nnode,
                          map_exit,
                          acc_write,
                          src_conn='_out',
                          memlet=dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))

    sdfg.validate()
    return sdfg


@pytest.mark.xfail(
    reason='Codegen bug: WCR sum through a NestedSDFG source is silently dropped; the '
    'result is approximately the last iteration value, not the running sum. The '
    'WCR codegen path special-cases ``Tasklet`` as the upstream node class; when '
    'the upstream is a ``NestedSDFG`` (vectorization-pipeline-style normalised map '
    'body), the reduction is not emitted and the parallel result is wrong. This '
    'test is the regression target -- fixing the codegen should flip it to PASS.',
    strict=True,
)
def test_wcr_through_nested_sdfg_sum_reduction():
    """A WCR-sum edge whose source is a NestedSDFG (one Map iteration per body
    invocation) should accumulate correctly: ``acc[0] += sum(src)``.
    """
    n = 64
    rng = np.random.default_rng(0)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1, dtype=np.float64)

    sdfg = _build_wcr_nsdfg_sdfg(n)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum()), (f'WCR sum through NSDFG returned {acc[0]}, expected {src.sum()}.')


@pytest.mark.xfail(
    reason='Same WCR-through-NSDFG codegen bug as above; documents that the bug also '
    'breaks the live-in case (``acc[0] = 10.0`` before the Map runs).',
    strict=True,
)
def test_wcr_through_nested_sdfg_with_initial_value():
    """The pre-loop ``acc[0]`` value should be preserved (WCR accumulates *into* it)."""
    n = 32
    rng = np.random.default_rng(1)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.array([10.0])

    sdfg = _build_wcr_nsdfg_sdfg(n)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], 10.0 + src.sum())


def _build_wcr_via_private_scalar_sdfg(n: int) -> dace.SDFG:
    """Workable shape: the NSDFG writes a per-iteration value into a private *scalar
    AccessNode* inside the Map scope; the WCR sits on the AccessNode -> MapExit edge.

    The codegen's WCR analysis recognises ``AccessNode -[wcr]-> MapExit`` as a
    reduction (the canonical ``acc += val`` shape ``AugAssignToWCR`` produces), so
    the parallel sum runs correctly regardless of whether the upstream node is a
    Tasklet or a NestedSDFG -- the NSDFG's own output edge has no WCR, it just
    writes its scalar result into the private AccessNode.

    Structure::

        src --> MapEntry --> NSDFG_in --[nsdfg: _out = _in]--> NSDFG_out
            --> AccessNode(_priv)  [no WCR]
            --[wcr=+]--> MapExit --[wcr=+]--> AccessNode(acc)
    """
    sdfg = dace.SDFG(f'wcr_via_private_scalar_n{n}')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    # The private per-iteration scalar lives inside the Map scope. Marked transient
    # so its lifetime is per-iteration (Scope-lifetime by default for a scalar
    # declared in a Map body).
    sdfg.add_scalar('_priv', dace.float64, transient=True)
    state = sdfg.add_state('map')
    src_read = state.add_read('src')
    acc_write = state.add_write('acc')
    priv_write = state.add_access('_priv')

    nsdfg = dace.SDFG('body')
    nsdfg.add_array('_in', [1], dace.float64)
    nsdfg.add_array('_out', [1], dace.float64)
    nstate = nsdfg.add_state('s0')
    nin = nstate.add_read('_in')
    nout = nstate.add_write('_out')
    nstate.add_nedge(nin, nout, dace.Memlet(data='_in', subset='0', other_subset='0'))

    map_entry, map_exit = state.add_map('outer', {'i': f'0:{n}'})
    nnode = state.add_nested_sdfg(nsdfg, {'_in'}, {'_out'})

    state.add_memlet_path(src_read, map_entry, nnode, dst_conn='_in', memlet=dace.Memlet(data='src', subset='i'))
    # NSDFG output -> private scalar (no WCR; just a plain write of this iter's value).
    state.add_edge(nnode, '_out', priv_write, None, dace.Memlet(data='_priv', subset='0'))
    # Private scalar -> MapExit -> acc (WCR on the AccessNode->MapExit edge; the
    # codegen sees this as the canonical reduction shape).
    state.add_memlet_path(priv_write,
                          map_exit,
                          acc_write,
                          memlet=dace.Memlet(data='acc', subset='0', wcr='lambda a, b: a + b'))
    sdfg.validate()
    return sdfg


def _wcr_edge_sources(sdfg: dace.SDFG):
    """Return the list of upstream node *classes* for every WCR-carrying edge in ``sdfg``."""
    out = []
    for sd in sdfg.all_sdfgs_recursive():
        for st in sd.all_states():
            for e in st.edges():
                if e.data is not None and e.data.wcr is not None:
                    out.append(type(e.src).__name__)
    return out


def test_wcr_via_private_scalar_numerically_correct():
    """The private-scalar shape sums correctly: ``acc[0] += sum(src)``."""
    n = 64
    rng = np.random.default_rng(2)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1, dtype=np.float64)

    sdfg = _build_wcr_via_private_scalar_sdfg(n)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum()), f'got {acc[0]}, expected {src.sum()}'


def test_wcr_via_private_scalar_preserves_initial_value():
    """The pre-Map ``acc[0]`` value is preserved (WCR accumulates into it)."""
    n = 32
    rng = np.random.default_rng(3)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.array([10.0])

    sdfg = _build_wcr_via_private_scalar_sdfg(n)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], 10.0 + src.sum())


def test_wcr_via_private_scalar_edge_sourced_from_access_node():
    """Structural: every WCR-carrying edge originates at an AccessNode (or a MapExit
    propagating it outward), never at the NSDFG. This is the property that lets the
    codegen recognise the reduction; the buggy variant above puts WCR directly on a
    NSDFG-out edge and the codegen drops it."""
    n = 8
    sdfg = _build_wcr_via_private_scalar_sdfg(n)
    sources = _wcr_edge_sources(sdfg)
    assert sources, 'Expected at least one WCR-carrying edge in the SDFG.'
    bad = [c for c in sources if c not in ('AccessNode', 'MapExit')]
    assert not bad, f'WCR edges must source from AccessNode/MapExit only; got {bad}.'


@pytest.mark.parametrize(
    'wcr_str,binop',
    [
        ('lambda a, b: a + b', lambda a, b: a + b),
        ('lambda a, b: a * b', lambda a, b: a * b),
        ('lambda a, b: max(a, b)', max),
        ('lambda a, b: min(a, b)', min),
    ],
    ids=['sum', 'product', 'max', 'min'],
)
@pytest.mark.xfail(
    reason='Same root cause as ``test_nest_state_subgraph_wcr_placement``: the core '
    'helper does not normalise WCR placement, and the codegen drops the reduction. '
    'A separate transformation pass (not a core change) will close this xfail by '
    'rewriting NSDFG-sourced WCR edges to the canonical AccessNode-sourced shape.',
    strict=True,
)
def test_nest_state_subgraph_emits_detectable_wcr_shape(wcr_str, binop):
    """The pipeline should turn ``nest_state_subgraph`` output into
    ``NestedSDFG -> AccessNode -[wcr]-> MapExit -> AN(acc)`` (the canonical shape the
    codegen recognises), for every associative WCR op. Without the normalisation pass
    this xfails. ``functools.reduce`` over ``src`` is the reference.
    """
    import functools

    from dace.sdfg.graph import SubgraphView
    from dace.transformation.helpers import nest_state_subgraph

    n = 32
    sdfg = dace.SDFG(f'helper_wcr_shape_{wcr_str.split(":")[-1].strip().replace(" ", "_")}_n{n}')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    st = sdfg.add_state('m')
    src_read = st.add_read('src')
    acc_write = st.add_write('acc')
    me, mx = st.add_map('m', {'i': f'0:{n}'})
    t = st.add_tasklet('id', {'_in'}, {'_out'}, '_out = _in')
    st.add_memlet_path(src_read, me, t, dst_conn='_in', memlet=dace.Memlet(data='src', subset='i'))
    st.add_memlet_path(t, mx, acc_write, src_conn='_out', memlet=dace.Memlet(data='acc', subset='0', wcr=wcr_str))

    nest_state_subgraph(sdfg, st, SubgraphView(st, {t}), name='wrapped_body')
    sdfg.validate()

    # Structural: no NestedSDFG-source WCR edge survives anywhere in the SDFG.
    wcr_sources = [
        type(e.src).__name__ for sd in sdfg.all_sdfgs_recursive() for state in sd.all_states() for e in state.edges()
        if e.data is not None and e.data.wcr is not None
    ]
    assert wcr_sources, 'Expected at least one WCR edge.'
    assert 'NestedSDFG' not in wcr_sources, (
        f'WCR sources after helper wrapping: {wcr_sources}. None should be NestedSDFG; '
        'the helper must emit ``NSDFG -> AN -[wcr]-> MapExit -> AN(acc)`` so the '
        "codegen's reduction analysis recognises it.")

    # Numerical: matches a sequential reduction over the same data.
    rng = np.random.default_rng(hash(wcr_str) & 0xFFFFFFFF)
    if binop is min or binop is max:
        src = rng.uniform(-1.0, 1.0, size=n)
        init = 0.0
    elif wcr_str.endswith('a * b'):
        src = rng.uniform(0.95, 1.05, size=n)
        init = 1.0
    else:
        src = rng.uniform(-1.0, 1.0, size=n)
        init = 0.0
    expected = functools.reduce(binop, src.tolist(), init)
    acc = np.array([init])
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], expected), (f'helper-wrapped WCR ({wcr_str}) returned {acc[0]}, expected {expected}. '
                                          'If wildly off, the codegen has stopped detecting the helper-emitted shape.')


@pytest.mark.xfail(
    reason='Documents that ``nest_state_subgraph`` (core helper) leaves the WCR edge '
    'on the NestedSDFG output, producing the buggy shape codegen drops. The fix is '
    'NOT in the core helper -- a separate normalising transformation pass will rewrite '
    "``NSDFG -[wcr]-> MapExit`` to ``NSDFG -> AN -[wcr]-> MapExit`` pre-codegen. "
    'Flips to PASS once that pass runs in the default pipeline.',
    strict=True,
)
def test_nest_state_subgraph_wcr_placement():
    """``nest_state_subgraph`` (the helper underlying ``NestInnermostMapBodyIntoNSDFG``)
    wraps a Map body in a NestedSDFG. When the wrapped body has a WCR output, the
    helper currently emits the WCR on the NSDFG -> MapExit edge (codegen drops it).
    The canonical shape is ``NSDFG -> AN -[wcr]-> MapExit``; a pre-codegen normalising
    transformation pass will rewrite to it without changing core helpers.
    """
    from dace.sdfg.graph import SubgraphView
    from dace.transformation.helpers import nest_state_subgraph

    n = 32
    sdfg = dace.SDFG(f'nest_helper_wcr_n{n}')
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

    # Wrap the tasklet (and just the tasklet) in a NestedSDFG via the helper.
    body_nodes = {t}
    nest_state_subgraph(sdfg, st, SubgraphView(st, body_nodes), name='wrapped_body')
    sdfg.validate()

    # Structural: WCR sources must be AccessNode or MapExit, not NestedSDFG.
    nsdfg_wcr_sources = [type(e.src).__name__ for e in st.edges() if e.data is not None and e.data.wcr is not None]
    assert nsdfg_wcr_sources, 'Expected at least one WCR edge after wrapping.'
    assert 'NestedSDFG' not in nsdfg_wcr_sources, (
        'After ``nest_state_subgraph``, no WCR edge should originate at a NestedSDFG; '
        f'got sources {nsdfg_wcr_sources}. The fix in ``helpers.py`` puts the WCR on '
        'the new private-AccessNode -> MapExit edge instead, which is what the codegen '
        "expects (the canonical ``AugAssignToWCR`` shape).")

    # Numerical: the wrapped SDFG still sums correctly.
    rng = np.random.default_rng(42)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum()), (f'WCR through helper-wrapped body returned {acc[0]}, expected {src.sum()}.')


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
