# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.transformation.estimator.enumeration.brute_force_enumerator import BruteForceEnumerator
from dace.transformation.estimator.enumeration.connected_enumerator import ConnectedEnumerator
import dace
import numpy as np
import pytest

from dace.transformation.estimator import GreedyEnumerator
from dace.transformation.subgraph.composite import CompositeFusion
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph.reduce_expansion import ReduceExpansion

W = dace.symbol('W')
H = dace.symbol('H')
B = dace.symbol('B')


@dace.program
def p1(in1: dace.float32[W, H, B], in2: dace.float32[W, H], out: dace.float32[W, H]):
    tmp1 = np.ndarray([W, H, B], dtype=dace.float32)
    for i, j, k in dace.map[0:W, 0:H, 0:B]:
        with dace.tasklet:
            a << in1[i, j, k]
            b << in2[i, j]
            c >> tmp1[i, j, k]
            c = a + b * 2

    tmp2 = np.ndarray([W, H, B], dtype=dace.float32)
    #tmp3 = np.ndarray([W, H], dtype=dace.float32)

    for i, j, k in dace.map[0:W, 0:H, 0:B]:
        with dace.tasklet:
            a << tmp1[i, j, k]
            c >> tmp2[i, j, k]
            c = 3 * a

    tmp3 = dace.reduce(lambda x, y: x + y, tmp1, axis=2, identity=0)
    tmp4 = dace.reduce(lambda x, y: x + y, tmp2, axis=2, identity=0)

    for i, j in dace.map[0:W, 0:H]:
        with dace.tasklet:
            a << tmp3[i, j]
            b << tmp4[i, j]
            c >> out[i, j]

            c = a * 2 + b * 3 + 1


@pytest.mark.parametrize(["map_splits"], [[True], [False]])
def test_greedy(map_splits):
    # Test diamond graph structure and ensure topologically correct enumeration
    w = 30
    h = 30
    b = 20
    A1 = np.random.rand(w, h, b).astype(np.float32)
    A2 = np.random.rand(w, h).astype(np.float32)
    ret = np.zeros([w, h], dtype=np.float32)

    sdfg = p1.to_sdfg()
    sdfg.simplify()
    graph = sdfg.nodes()[0]

    sdfg.apply_transformations_repeated(ReduceExpansion)

    subgraph = SubgraphView(graph, graph.nodes())
    composite = CompositeFusion(subgraph)
    composite.expansion_split = map_splits
    cf = lambda sdfg, subgraph: composite.can_be_applied(sdfg, subgraph)
    enum = GreedyEnumerator(sdfg, graph, subgraph, cf)
    result = enum.list()
    if map_splits:
        assert len(result) == 1
    else:
        assert len(result) == 2


@pytest.mark.parametrize(["map_splits"], [[True], [False]])
def test_connected(map_splits):
    # Test diamond graph structure and ensure topologically correct enumeration
    w = 30
    h = 30
    b = 20
    A1 = np.random.rand(w, h, b).astype(np.float32)
    A2 = np.random.rand(w, h).astype(np.float32)
    ret = np.zeros([w, h], dtype=np.float32)

    sdfg = p1.to_sdfg()
    sdfg.simplify()
    graph = sdfg.nodes()[0]

    sdfg.apply_transformations_repeated(ReduceExpansion)

    subgraph = SubgraphView(graph, graph.nodes())
    composite = CompositeFusion(subgraph)
    composite.expansion_split = map_splits
    cf = lambda sdfg, subgraph: composite.can_be_applied(sdfg, subgraph)
    enum = ConnectedEnumerator(sdfg, graph, subgraph, cf)
    result = enum.list()

    if map_splits:
        assert len(result) == 14
    else:
        assert len(result) == 4


@pytest.mark.parametrize(["map_splits"], [[True], [False]])
def test_brute_force(map_splits):
    # Test diamond graph structure and ensure topologically correct enumeration
    w = 30
    h = 30
    b = 20
    A1 = np.random.rand(w, h, b).astype(np.float32)
    A2 = np.random.rand(w, h).astype(np.float32)
    ret = np.zeros([w, h], dtype=np.float32)

    sdfg = p1.to_sdfg()
    sdfg.simplify()
    graph = sdfg.nodes()[0]

    sdfg.apply_transformations_repeated(ReduceExpansion)

    subgraph = SubgraphView(graph, graph.nodes())
    composite = CompositeFusion(subgraph)
    composite.expansion_split = map_splits
    cf = lambda sdfg, subgraph: composite.can_be_applied(sdfg, subgraph)
    enum = BruteForceEnumerator(sdfg, graph, subgraph, cf)
    result = enum.list()
    if map_splits:
        assert len(result) == 15
    else:
        assert len(result) == 5


if __name__ == "__main__":
    test_greedy(True)
    test_greedy(False)

    test_connected(True)
    test_connected(False)

    test_brute_force(True)
    test_brute_force(False)
