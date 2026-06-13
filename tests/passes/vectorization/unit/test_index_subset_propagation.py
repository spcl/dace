# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Index-subset propagation: undo frontend index-symbol promotion in memlet subsets.

The frontend promotes a computed index ``i + offset`` to a scalar (``i_plus_offset``,
written by a tasklet) then to a symbol (``__sym = i_plus_offset`` on an interstate edge)
used in a memlet subset ``A[__sym]``. ``SymbolPropagation`` folds the ``__sym`` layer to the
scalar ``A[i_plus_offset]`` but cannot reach ``A[i+offset]`` (the scalar depends on the map
parameter ``i``). :func:`propagate_subset` (resolver last hop) inlines it to the direct
``A[i+offset]`` so the access widens to a **dense load** — while leaving genuine
data-dependent gather indices (``A[idx[i]]``) untouched.

These tests build the exact pattern via the SDFG API (the `@dace.program` frontend does not
reliably reproduce the promotion) and pin both the contiguous-inline and gather-stop behaviors.
"""
import dace
import pytest

from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.vectorization.utils.tile_access import propagate_subset

N = dace.symbol('N')


def _build(gather: bool) -> dace.SDFG:
    """Inner-NSDFG promotion pattern. ``gather=False``: ``i_plus_offset = i + offset`` →
    ``A[__sym]``. ``gather=True``: ``g = idx[i]`` (data-dependent) → ``A[__sym]``."""
    sdfg = dace.SDFG('repro_gather' if gather else 'repro')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_symbol('offset', dace.int64)
    inputs = {'A'}
    if gather:
        sdfg.add_array('idx', [N], dace.int64)
        inputs = {'A', 'idx'}
    st = sdfg.add_state('main', is_start_block=True)
    me, mx = st.add_map('m', {'i': '0:N'})

    inner = dace.SDFG('inner')
    inner.add_array('A', [N], dace.float64)
    inner.add_array('out', [N], dace.float64)
    inner.add_scalar('idxval', dace.int64, transient=True)
    inner.add_symbol('i', dace.int64)
    inner.add_symbol('__sym', dace.int64)
    if gather:
        inner.add_array('idx', [N], dace.int64)
    else:
        inner.add_symbol('offset', dace.int64)
    s0 = inner.add_state('s0', is_start_block=True)
    if gather:
        rd_idx = s0.add_access('idx')
        t = s0.add_tasklet('gidx', {'a'}, {'o'}, 'o = a')
        s0.add_edge(rd_idx, None, t, 'a', dace.Memlet('idx[i]'))
        s0.add_edge(t, 'o', s0.add_access('idxval'), None, dace.Memlet('idxval[0]'))
    else:
        t = s0.add_tasklet('cidx', {}, {'o'}, 'o = i + offset')
        s0.add_edge(t, 'o', s0.add_access('idxval'), None, dace.Memlet('idxval[0]'))
    s1 = inner.add_state('s1')
    inner.add_edge(s0, s1, dace.InterstateEdge(assignments={'__sym': 'idxval'}))
    rd = s1.add_access('A')
    wr = s1.add_access('out')
    tt = s1.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    s1.add_edge(rd, None, tt, 'a', dace.Memlet('A[__sym]'))
    s1.add_edge(tt, 'o', wr, None, dace.Memlet('out[i]'))

    symmap = {'i': 'i', 'N': 'N'}
    if not gather:
        symmap['offset'] = 'offset'
    nsdfg = st.add_nested_sdfg(inner, inputs, {'out'}, symmap)
    for arr in inputs:
        st.add_edge(st.add_access(arr), None, me, f'IN_{arr}', dace.Memlet(f'{arr}[0:N]'))
        st.add_edge(me, f'OUT_{arr}', nsdfg, arr, dace.Memlet(f'{arr}[0:N]'))
        me.add_in_connector(f'IN_{arr}')
        me.add_out_connector(f'OUT_{arr}')
    st.add_edge(nsdfg, 'out', mx, 'IN_out', dace.Memlet('out[0:N]'))
    st.add_edge(mx, 'OUT_out', st.add_access('out'), None, dace.Memlet('out[0:N]'))
    mx.add_in_connector('IN_out')
    mx.add_out_connector('OUT_out')
    sdfg.validate()
    return sdfg


def _a_subsets(sdfg):
    out = []
    for sd in sdfg.all_sdfgs_recursive():
        for s in sd.states():
            for e in s.edges():
                if e.data and e.data.data == 'A' and e.data.subset is not None:
                    out.append(str(e.data.subset))
    return out


def _run_chain_and_rewrite(sdfg):
    Pipeline([ScalarToSymbolPromotion(), SymbolPropagation(), RemoveUnusedSymbols()]).apply_pass(sdfg, {})
    for sd in sdfg.all_sdfgs_recursive():
        for s in sd.states():
            for e in list(s.edges()):
                if e.data and e.data.data == 'A' and e.data.subset is not None:
                    new = propagate_subset(e.data.subset, sd, s)
                    if new is not None:
                        e.data.subset = new


def test_contiguous_index_inlined_to_dense():
    sdfg = _build(gather=False)
    assert any('__sym' in s for s in _a_subsets(sdfg)), "fixture should start with A[__sym]"
    _run_chain_and_rewrite(sdfg)
    subs = _a_subsets(sdfg)
    # The point access is now the direct i+offset form; no __sym, no scalar index.
    assert any('i + offset' in s or 'offset + i' in s for s in subs), subs
    assert not any('__sym' in s for s in subs), subs
    assert not any('idxval' in s for s in subs), subs


def test_gather_index_not_inlined():
    sdfg = _build(gather=True)
    _run_chain_and_rewrite(sdfg)
    subs = _a_subsets(sdfg)
    # Data-dependent index must be left as a gather (NOT inlined to an array read in the subset).
    assert not any('idx[' in s for s in subs), f"gather index wrongly inlined into subset: {subs}"


def test_iplusoffset_kernel_emits_no_gather():
    """End-to-end: the `a[i+offset1, j+offset2]` contiguous pattern must vectorize to a
    DENSE load — no gather. Asserts the multi-dim pipeline emits no TileLoad/TileStore
    with gather_dims and mints no per-lane index tile (`_idx_*`)."""
    import copy
    from tests.passes.vectorization.kernels.test_nested import tasklet_in_nested_sdfg
    from tests.passes.vectorization.helpers.harness import _auto_tile_widths
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
    from dace.libraries.tileops import TileLoad, TileStore

    sdfg = tasklet_in_nested_sdfg.to_sdfg(simplify=True)
    widths = _auto_tile_widths(sdfg, 8)
    cs = copy.deepcopy(sdfg)
    cs.name = "iplusoffset_nogather"
    VectorizeCPUMultiDim(widths=widths, expand_tile_nodes=False).apply_pass(cs, {})

    gather_nodes = []
    idx_tiles = []
    for sd in cs.all_sdfgs_recursive():
        idx_tiles += [n for n in sd.arrays if n.startswith('_idx_')]
        for st in sd.states():
            for n in st.nodes():
                if isinstance(n, (TileLoad, TileStore)) and getattr(n, 'gather_dims', ()):
                    gather_nodes.append((type(n).__name__, n.gather_dims))
    assert not gather_nodes, f"contiguous a[i+offset] emitted a gather: {gather_nodes}"
    assert not idx_tiles, f"contiguous a[i+offset] minted per-lane index tiles: {idx_tiles}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
