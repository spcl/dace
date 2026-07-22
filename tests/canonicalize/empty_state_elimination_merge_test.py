# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Assignment merging in ``EmptyStateElimination``."""
import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination


def _chain(first_assign, second_assign, second_cond=None):
    """``head -[first]-> empty -[second]-> tail`` with a symbol read in ``tail``."""
    sdfg = dace.SDFG('chain')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('m', dace.int64)
    sdfg.add_array('A', [2], dace.int64)

    head = sdfg.add_state('head', is_start_block=True)
    empty = sdfg.add_state('empty')
    tail = sdfg.add_state('tail')
    sdfg.add_edge(head, empty, InterstateEdge(assignments=first_assign))
    edge = InterstateEdge(assignments=second_assign)
    if second_cond is not None:
        edge = InterstateEdge(condition=second_cond, assignments=second_assign)
    sdfg.add_edge(empty, tail, edge)

    t = tail.add_tasklet('w', {}, {'o'}, 'o = m')
    tail.add_edge(t, 'o', tail.add_access('A'), None, dace.Memlet('A[0]'))
    return sdfg, empty


def test_independent_assignments_merge():
    """Disjoint reads/writes: both edges collapse onto one bypass edge."""
    sdfg, empty = _chain({'k': '1'}, {'m': '2'})
    assert EmptyStateElimination().apply_pass(sdfg, {}) == 1
    assert empty not in sdfg.nodes()
    edges = sdfg.edges()
    assert len(edges) == 1
    assert edges[0].data.assignments == {'k': '1', 'm': '2'}


def test_dependent_assignment_blocks_merge():
    """``m := k + 1`` reads what the first edge writes; unordered on one edge."""
    sdfg, empty = _chain({'k': '1'}, {'m': 'k + 1'})
    assert EmptyStateElimination().apply_pass(sdfg, {}) is None
    assert empty in sdfg.nodes()


def test_lhs_collision_merges_to_second():
    """Both edges write ``m``; the later write wins either way."""
    sdfg, empty = _chain({'m': '1'}, {'m': '2'})
    assert EmptyStateElimination().apply_pass(sdfg, {}) == 1
    assert sdfg.edges()[0].data.assignments == {'m': '2'}


def test_conditional_successor_edge_is_kept():
    """A condition on the outgoing edge cannot be pushed onto predecessors."""
    sdfg, empty = _chain({'k': '1'}, {'m': '2'}, second_cond='k > 0')
    assert EmptyStateElimination().apply_pass(sdfg, {}) is None
    assert empty in sdfg.nodes()


def test_start_state_assignment_is_kept():
    """No predecessor edge exists to carry the outgoing assignments."""
    sdfg = dace.SDFG('start')
    sdfg.add_symbol('m', dace.int64)
    sdfg.add_array('A', [2], dace.int64)
    empty = sdfg.add_state('empty', is_start_block=True)
    tail = sdfg.add_state('tail')
    sdfg.add_edge(empty, tail, InterstateEdge(assignments={'m': '2'}))
    t = tail.add_tasklet('w', {}, {'o'}, 'o = m')
    tail.add_edge(t, 'o', tail.add_access('A'), None, dace.Memlet('A[0]'))

    assert EmptyStateElimination().apply_pass(sdfg, {}) is None
    assert empty in sdfg.nodes()


def test_merged_chain_is_value_preserving():
    """The merged edge computes the same value the two-edge chain did."""
    import numpy as np
    sdfg, _ = _chain({'k': '3'}, {'m': '7'})
    EmptyStateElimination().apply_pass(sdfg, {})
    sdfg.validate()
    a = np.zeros([2], dtype=np.int64)
    sdfg(A=a)
    assert a[0] == 7


if __name__ == '__main__':
    test_independent_assignments_merge()
    test_dependent_assignment_blocks_merge()
    test_lhs_collision_merges_to_second()
    test_conditional_successor_edge_is_kept()
    test_start_state_assignment_is_kept()
    test_merged_chain_is_value_preserving()
