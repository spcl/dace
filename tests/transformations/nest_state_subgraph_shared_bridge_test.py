# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``nest_state_subgraph`` must not delete a transient that something else still names.

A transient bridging two tasklets is named by nothing but the memlet on the edge between them --
there is no access node for it. ``nest_state_subgraph`` decided a transient was subgraph-local by
looking at access nodes alone, so a second user of such a bridge was invisible: nesting one body
moved the descriptor into the nested SDFG and deleted it from the parent, leaving the other user
pointing at nothing (``KeyError`` the next time anything looked the descriptor up -- e.g. nesting
that other body in turn, which is exactly what the tile remainder tail does after
``SplitMapForTileRemainder`` copies a body).
"""
import dace
import pytest
from dace.sdfg.graph import SubgraphView
from dace.transformation.helpers import nest_state_subgraph


def _two_bodies_sharing_a_bridge_name():
    """Two independent tasklet pairs, each bridging through its own element of ``bridge``.

    Mirrors a split map: the main body and the remainder tail are separate node sets that both
    name the same bridge transient on a tasklet-to-tasklet edge.
    """
    sdfg = dace.SDFG('shared_bridge')
    sdfg.add_array('a', [2], dace.float64)
    sdfg.add_array('b', [2], dace.float64)
    sdfg.add_transient('bridge', [1], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)

    made = []
    for i in (0, 1):
        producer = state.add_tasklet(f'produce_{i}', {'v'}, {'o'}, 'o = v * 2')
        consumer = state.add_tasklet(f'consume_{i}', {'v'}, {'o'}, 'o = v + 1')
        state.add_edge(state.add_access('a'), None, producer, 'v', dace.Memlet(f'a[{i}]'))
        state.add_edge(producer, 'o', consumer, 'v', dace.Memlet('bridge[0]'))
        state.add_edge(consumer, 'o', state.add_access('b'), None, dace.Memlet(f'b[{i}]'))
        made.append((producer, consumer))
    return sdfg, state, made


def test_bridge_transient_survives_when_another_body_still_names_it():
    sdfg, state, made = _two_bodies_sharing_a_bridge_name()

    nest_state_subgraph(sdfg, state, SubgraphView(state, list(made[0])), name='first_body')

    assert 'bridge' in sdfg.arrays, 'the second body still names bridge -- it must not be deleted'
    # And the nested SDFG got its own copy, so the moved body is self-contained.
    nested = [n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(nested) == 1
    assert 'bridge' in nested[0].sdfg.arrays

    # The second body nests too, which is what used to raise KeyError.
    nest_state_subgraph(sdfg, state, SubgraphView(state, list(made[1])), name='second_body')
    sdfg.validate()


def test_an_exclusive_bridge_transient_is_still_moved_out():
    """The no-other-user case is unchanged: a bridge only this subgraph names moves in."""
    sdfg, state, made = _two_bodies_sharing_a_bridge_name()
    # Drop the second body so the bridge has exactly one user.
    for node in made[1]:
        state.remove_node(node)

    nest_state_subgraph(sdfg, state, SubgraphView(state, list(made[0])), name='only_body')
    assert 'bridge' not in sdfg.arrays, 'a subgraph-local transient still moves into the nested SDFG'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
