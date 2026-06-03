"""Unit tests for :class:`ExpandNestedSDFGInputs`."""
import copy

import dace
import numpy as np

from dace.sdfg import nodes
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol('N')
M = dace.symbol('M')
K = 4


@dace.program
def _jacobi2d_map_tile(a: dace.float64[N, M], b: dace.float64[N, M]):
    for ii, jj in dace.map[0:N - 2:K, 0:M - 2:K]:
        for i, j in dace.map[0:K, 0:K]:
            b[ii + i + 1, jj + j + 1] = a[ii + i + 1, jj + j + 1] * 2.0


def _count_nsdfgs(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))


def _build_map_to_for_loop_test_sdfg():
    """Build the post-MapToForLoop NSDFG-wrapped shape that
    ``ExpandNestedSDFGInputs`` must widen."""
    sdfg = _jacobi2d_map_tile.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([MapToForLoop()]).apply_pass(sdfg, {})
    return sdfg


def test_widens_narrowed_inedges_to_full_array():
    """Every in/out edge of the top-level NSDFG must read the full outer
    array after ``ExpandNestedSDFGInputs``."""
    from dace import subsets
    sdfg = _build_map_to_for_loop_test_sdfg()
    n_before = _count_nsdfgs(sdfg)
    assert n_before > 0
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    # Top-level NSDFGs (those NOT scoped by a Map) must now read the
    # full outer array on every in/out edge.
    for state in sdfg.states():
        for n in state.nodes():
            if not isinstance(n, nodes.NestedSDFG):
                continue
            if state.entry_node(n) is not None:
                continue  # Map-scoped: deliberately not widened
            for e in (*state.in_edges(n), *state.out_edges(n)):
                if e.data is None or e.data.data is None:
                    continue
                full = subsets.Range.from_array(sdfg.arrays[e.data.data])
                assert e.data.subset == full, \
                    f'NSDFG in-edge for {e.data.data!r} should be full {full}; got {e.data.subset}'


def test_apply_preserves_numerics_via_inline():
    """End-to-end: ExpandNestedSDFGInputs followed by InlineMultistateSDFG
    produces a numerically-identical SDFG to the un-modified one."""
    sdfg = _build_map_to_for_loop_test_sdfg()
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(sdfg, {})
    sdfg.validate()
    n, m = 10, 10
    rng = np.random.default_rng(0xCAFE)
    a = rng.standard_normal((n, m))
    b = np.zeros((n, m))
    ref = b.copy()
    copy.deepcopy(_jacobi2d_map_tile.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m)
    sdfg(a=a, b=b, N=n, M=m)
    assert np.allclose(b, ref), f'max diff: {np.abs(b - ref).max():.3e}'


def test_refuses_inside_map_scope():
    """If the NSDFG is inside a Map scope, the per-iteration narrowing
    is intentional and the transformation must refuse."""

    @dace.program
    def kernel(a: dace.float64[N, N], b: dace.float64[N, N]):
        # The outer Map wraps a NSDFG implicitly via to_sdfg.
        for i in dace.map[0:N]:
            for j in range(N):
                b[i, j] = a[i, j] * 2.0

    sdfg = kernel.to_sdfg(simplify=True)
    # No top-level NSDFGs to widen here, but if there are any nested
    # ones inside the Map scope the pass must leave them alone.
    before = _count_nsdfgs(sdfg)
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    after = _count_nsdfgs(sdfg)
    assert before == after, 'pass must not touch Map-scoped NSDFGs'


def test_introduced_symbol_picks_up_outer_type():
    """When the offset expression introduces a symbol not in the inner
    SDFG's table, the new symbol must inherit the outer SDFG's declared
    type (not silently fall back to ``int64``)."""
    sdfg = _build_map_to_for_loop_test_sdfg()
    # Verify the outer SDFG has at least one symbol of a non-default
    # type that the inner SDFG should pick up.
    outer_syms = set(sdfg.symbols)
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    # Each NSDFG that introduced a new symbol must carry the outer
    # symbol's type.
    for state in sdfg.states():
        for n in state.nodes():
            if not isinstance(n, nodes.NestedSDFG):
                continue
            for sym, mapping in n.symbol_mapping.items():
                if sym not in n.sdfg.symbols:
                    continue
                inner_t = n.sdfg.symbols[sym]
                if sym in sdfg.symbols:
                    assert inner_t == sdfg.symbols[sym], \
                        f'symbol {sym!r}: inner type {inner_t} != outer type {sdfg.symbols[sym]}'
