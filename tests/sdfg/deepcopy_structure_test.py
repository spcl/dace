# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A deep copy of an SDFG reproduces every graph container of the original, in order, and shares none."""
import copy
from typing import Any, Dict, List, Tuple

import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


def build_sdfg() -> Tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG('deepcopy_structure')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    entry = sdfg.add_state('entry', is_start_block=True)
    loop = LoopRegion('loop', 'i < 8', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(entry, loop, dace.InterstateEdge(assignments={'k': '1'}))
    body = loop.add_state('body', is_start_block=True)
    read = body.add_read('A')
    write = body.add_write('B')
    tasklet = body.add_tasklet('combine', {'lhs': None, 'rhs': None}, {'out': None}, 'out = lhs + rhs')
    # Two parallel edges between the same pair of nodes: distinct multigraph keys.
    body.add_edge(read, None, tasklet, 'lhs', dace.Memlet('A[i]'))
    body.add_edge(read, None, tasklet, 'rhs', dace.Memlet('A[i]'))
    body.add_edge(tasklet, 'out', write, None, dace.Memlet('B[i]'))
    branch = ConditionalBlock('branch', sdfg=sdfg)
    loop.add_node(branch)
    loop.add_edge(body, branch, dace.InterstateEdge())
    arm = ControlFlowRegion('arm', sdfg=sdfg)
    branch.add_branch(dace.properties.CodeBlock('i > 2'), arm)
    arm.add_state('arm_state', is_start_block=True)
    # Node order no longer matches the networkx node order, which the copy has to keep independently.
    body.reorder_nodes([write, tasklet, read])
    return sdfg, body


def container_orders(sdfg: dace.SDFG) -> List[Tuple[Any, ...]]:
    positions: Dict[int, int] = {}

    def position(obj: Any) -> int:
        return positions.setdefault(id(obj), len(positions))

    orders = []
    for graph in [sdfg, *sdfg.all_control_flow_blocks()]:
        if not isinstance(graph, dace.sdfg.graph.OrderedDiGraph):
            continue
        multigraph = graph.nx.is_multigraph()
        orders.append((
            graph.label,
            [position(node) for node in graph.nodes()],
            [(position(e.src), position(e.dst)) for e in graph.edges()],
            [[(position(e.src), position(e.dst)) for e in graph.in_edges(node)] for node in graph.nodes()],
            [position(node) for node in graph.nx.nodes],
            [(position(u), [(position(v), list(entry) if multigraph else None) for v, entry in neighbors.items()])
             for u, neighbors in graph.nx.succ.items()],
            [(position(v), [position(u) for u in neighbors]) for v, neighbors in graph.nx.pred.items()],
        ))
    return orders


def test_a_deep_copy_keeps_every_node_edge_and_adjacency_order():
    sdfg, _ = build_sdfg()

    clone = copy.deepcopy(sdfg)

    assert container_orders(clone) == container_orders(sdfg)
    clone.validate()


def test_removing_one_of_two_parallel_edges_from_a_copy_updates_both_adjacency_directions():
    sdfg, _ = build_sdfg()
    clone = copy.deepcopy(sdfg)
    body = next(state for state in clone.all_states() if state.label == 'body')
    tasklet = next(node for node in body.nodes() if isinstance(node, dace.nodes.Tasklet))
    read = next(node for node in body.nodes() if isinstance(node, dace.nodes.AccessNode) and node.data == 'A')

    body.remove_edge(body.in_edges(tasklet)[0])

    assert body.nx.number_of_edges(read, tasklet) == 1
    assert len(body.nx.pred[tasklet][read]) == 1
    assert body.in_edges(tasklet)[0].data is next(iter(body.nx.succ[read][tasklet].values()))['data']


def test_a_deep_copy_shares_no_node_edge_or_memlet_with_the_original():
    sdfg, body = build_sdfg()

    clone = copy.deepcopy(sdfg)

    cloned_body = next(state for state in clone.all_states() if state.label == 'body')
    originals = {id(obj) for obj in (*body.nodes(), *body.edges(), *(e.data for e in body.edges()))}
    copies = {id(obj) for obj in (*cloned_body.nodes(), *cloned_body.edges(), *(e.data for e in cloned_body.edges()))}
    assert originals.isdisjoint(copies)
    cloned_body.remove_edge(cloned_body.edges()[0])
    assert len(body.edges()) == 3


if __name__ == '__main__':
    test_a_deep_copy_keeps_every_node_edge_and_adjacency_order()
    test_removing_one_of_two_parallel_edges_from_a_copy_updates_both_adjacency_directions()
    test_a_deep_copy_shares_no_node_edge_or_memlet_with_the_original()
