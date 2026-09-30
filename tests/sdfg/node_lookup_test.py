# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests positional node lookups of ordered graphs against a reference scan. """
import pytest

import dace
from dace.sdfg import graph as gr, nodes


def _reference_node(graph, id):
    return next(n for i, n in enumerate(graph.nodes()) if i == id)


def _reference_node_id(graph, node):
    return next(i for i, n in enumerate(graph.nodes()) if n is node)


def _state_with_nodes(count: int) -> dace.SDFGState:
    sdfg = dace.SDFG('node_lookup')
    sdfg.add_array('A', [10], dace.float64)
    state = sdfg.add_state()
    for i in range(count):
        if i % 2:
            state.add_access('A')
        else:
            state.add_tasklet(f't{i}', {}, {}, '')
    return state


@pytest.mark.parametrize('count', [1, 2, 7, 50])
def test_state_node_lookup(count):
    state = _state_with_nodes(count)
    for i, node in enumerate(state.nodes()):
        assert state.node(i) is _reference_node(state, i)
        assert state.node_id(node) == _reference_node_id(state, node) == i
    if count > 1:
        assert state.node(True) is _reference_node(state, True)


def test_control_flow_region_node_lookup():
    sdfg = dace.SDFG('node_lookup_cfg')
    states = [sdfg.add_state(f's{i}') for i in range(6)]
    for i, state in enumerate(states):
        assert sdfg.node(i) is state
        assert sdfg.node_id(state) == i


def test_node_lookup_not_found():
    state = _state_with_nodes(5)
    with pytest.raises(gr.NodeNotFoundError):
        state.node(5)
    with pytest.raises(gr.NodeNotFoundError):
        state.node(-1)
    with pytest.raises(gr.NodeNotFoundError):
        state.node_id(nodes.Tasklet('not_in_state'))


def test_node_ids_are_positions_in_collapsed_graphs():
    """ Pattern matching takes the node ID from the position in `collapse_multigraph_to_nx`. """
    from dace.transformation.passes.pattern_matching import collapse_multigraph_to_nx

    @dace.program
    def branches_in_loop(A: dace.float64[10], B: dace.float64[10]):
        for i in range(10):
            if A[i] > 0:
                B[i] = A[i] + 1
            else:
                B[i] = A[i] - 1

    sdfg = branches_in_loop.to_sdfg(simplify=False)
    graphs = list(sdfg.all_control_flow_regions(recursive=True)) + list(sdfg.all_states())
    assert any(isinstance(g, dace.sdfg.state.ConditionalBlock) for g in graphs)
    for graph in graphs:
        collapsed = collapse_multigraph_to_nx(graph)
        for i, node in enumerate(graph.nodes()):
            assert collapsed.nodes[i]['node'] is node
            assert graph.node_id(node) == i


class _EqualToEveryNode:

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


def test_node_id_is_identity_based():
    graph = gr.OrderedDiGraph()
    graph.add_node('a')
    graph.add_node('b')
    with pytest.raises(gr.NodeNotFoundError):
        graph.node_id(_EqualToEveryNode())


if __name__ == '__main__':
    for count in [1, 2, 7, 50]:
        test_state_node_lookup(count)
    test_control_flow_region_node_lookup()
    test_node_lookup_not_found()
    test_node_ids_are_positions_in_collapsed_graphs()
    test_node_id_is_identity_based()
