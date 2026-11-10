# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``node_id`` answered from the per-graph index matches the position of the node in insertion order."""

import copy
import random

import pytest

import dace
from dace.sdfg import graph as gr
from dace.transformation.dataflow import MapTiling


def scanned_id(graph, node) -> int:
    return next(i for i, n in enumerate(graph.nodes()) if n is node)


def assert_ids_match_the_scan(graph):
    for i, node in enumerate(graph.nodes()):
        assert graph.node_id(node) == scanned_id(graph, node) == i
        assert graph.node(graph.node_id(node)) is node


def chain(length: int) -> dace.SDFGState:
    sdfg = dace.SDFG("chain")
    sdfg.add_array("A", [length + 1], dace.float64)
    state = sdfg.add_state()
    previous = state.add_access("A")
    for i in range(length):
        tasklet = state.add_tasklet(f"t{i}", {"x"}, {"y"}, "y = x + 1")
        state.add_edge(previous, None, tasklet, "x", dace.Memlet(f"A[{i}]"))
        previous = state.add_access("A")
        state.add_edge(tasklet, "y", previous, None, dace.Memlet(f"A[{i + 1}]"))
    return state


def mutate_randomly(graph, new_node, steps: int, seed: int):
    """Interleaves adds, removes and lookups; the index must equal the scan after every step."""
    rng = random.Random(seed)
    for _ in range(steps):
        nodes = graph.nodes()
        if nodes and rng.random() < 0.4:
            graph.remove_node(rng.choice(nodes))
        else:
            graph.add_node(new_node())
        if graph.number_of_nodes() and rng.random() < 0.7:
            graph.node_id(rng.choice(graph.nodes()))
        assert_ids_match_the_scan(graph)


def test_random_mutations_of_a_state():
    sdfg = dace.SDFG("random_state")
    sdfg.add_array("A", [1], dace.float64)
    state = sdfg.add_state()
    mutate_randomly(state, lambda: dace.nodes.AccessNode("A"), steps=500, seed=0)


def test_random_mutations_of_an_sdfg():
    sdfg = dace.SDFG("random_blocks")
    sdfg.add_state("start", is_start_block=True)
    names = iter(range(10**6))

    def new_block():
        return dace.SDFGState(f"s{next(names)}")

    mutate_randomly(sdfg, new_block, steps=500, seed=1)


def test_random_mutations_of_a_multigraph():
    graph = gr.OrderedMultiDiGraph()
    mutate_randomly(graph, object, steps=500, seed=2)


def test_removing_first_middle_and_last():
    for position in (0, 5, -1):
        state = chain(5)
        assert_ids_match_the_scan(state)
        state.remove_node(state.nodes()[position])
        assert_ids_match_the_scan(state)
        state.add_access("A")
        assert_ids_match_the_scan(state)


def test_a_foreign_node_is_not_found():
    state = chain(1)
    state.node_id(state.nodes()[0])
    foreign = chain(1).nodes()[0]
    with pytest.raises(gr.NodeNotFoundError):
        state.node_id(foreign)


def test_deepcopy_keeps_its_own_index():
    state = chain(4)
    sdfg = state.sdfg
    assert_ids_match_the_scan(state)
    copied = copy.deepcopy(sdfg)
    copied_state = copied.nodes()[0]
    assert_ids_match_the_scan(copied_state)
    state.remove_node(state.nodes()[0])
    state.add_access("A")
    assert_ids_match_the_scan(state)
    assert_ids_match_the_scan(copied_state)
    assert all(n not in state.nodes() for n in copied_state.nodes())


def test_json_round_trip():
    state = chain(4)
    # Drop the last tasklet and its output so the ids of the saved graph come from a rebuilt index
    state.remove_node(state.nodes()[-1])
    state.remove_node(state.nodes()[-1])
    state.sdfg.validate()
    loaded = dace.SDFG.from_json(state.sdfg.to_json())
    loaded_state = loaded.nodes()[0]
    assert_ids_match_the_scan(loaded_state)
    edges = sorted((state.node_id(e.src), state.node_id(e.dst)) for e in state.edges())
    assert sorted((loaded_state.node_id(e.src), loaded_state.node_id(e.dst)) for e in loaded_state.edges()) == edges


def test_transformation_ids_resolve_after_nodes_are_added():
    """A transformation stores node ids at match time and resolves them with ``node(id)`` after the graph grew."""

    @dace.program
    def scale(A: dace.float64[64]):
        for i in dace.map[0:64]:
            A[i] = 2 * A[i]

    sdfg = scale.to_sdfg(simplify=True)
    state = sdfg.nodes()[0]
    assert_ids_match_the_scan(state)
    assert sdfg.apply_transformations(MapTiling, options={"tile_sizes": (8,)}) == 1
    assert_ids_match_the_scan(state)
    sdfg.validate()


if __name__ == "__main__":
    test_random_mutations_of_a_state()
    test_random_mutations_of_an_sdfg()
    test_random_mutations_of_a_multigraph()
    test_removing_first_middle_and_last()
    test_a_foreign_node_is_not_found()
    test_deepcopy_keeps_its_own_index()
    test_json_round_trip()
    test_transformation_ids_resolve_after_nodes_are_added()
