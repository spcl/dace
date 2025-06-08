# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import MapExpansion


def test_expand_with_inputs():

    @dace.program
    def toexpand(A: dace.float64[4, 2], B: dace.float64[2, 2]):
        for i, j in dace.map[1:3, 0:2]:
            with dace.tasklet:
                a1 << A[i, j]
                a2 << A[i + 1, j]
                a3 << A[i - 1, j]
                b >> B[i - 1, j]
                b = a1 + a2 + a3

    sdfg = toexpand.to_sdfg()
    sdfg.simplify()

    # Init conditions
    sdfg.validate()
    assert len([node for node in sdfg.start_state.nodes() if isinstance(node, dace.nodes.MapEntry)]) == 1
    assert len([node for node in sdfg.start_state.nodes() if isinstance(node, dace.nodes.MapExit)]) == 1

    # Expansion
    assert sdfg.apply_transformations_repeated(MapExpansion) == 1
    sdfg.validate()

    map_entries = set()
    state = sdfg.start_state
    for node in state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue

        # (Fast) MapExpansion should not add memlet paths for each memlet to a tasklet
        if sdfg.start_state.entry_node(node) is None:
            assert state.in_degree(node) == 1
            assert state.out_degree(node) == 1
            assert len(node.out_connectors) == 1
        else:
            assert state.in_degree(node) == 1
            assert state.out_degree(node) == 3
            assert len(node.out_connectors) == 1

        map_entries.add(node)

    assert len(map_entries) == 2


def test_expand_without_inputs():

    @dace.program
    def toexpand(B: dace.float64[4, 4]):
        for i, j in dace.map[0:4, 0:4]:
            with dace.tasklet:
                b >> B[i, j]
                b = 0

    sdfg = toexpand.to_sdfg()
    sdfg.simplify()

    # Init conditions
    sdfg.validate()
    assert len([node for node in sdfg.start_state.nodes() if isinstance(node, dace.nodes.MapEntry)]) == 1
    assert len([node for node in sdfg.start_state.nodes() if isinstance(node, dace.nodes.MapExit)]) == 1

    # Expansion
    assert sdfg.apply_transformations_repeated(MapExpansion) == 1
    sdfg.validate()

    map_entries = set()
    state = sdfg.start_state
    for node in state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue

        # (Fast) MapExpansion should not add memlet paths for each memlet to a tasklet
        if state.entry_node(node) is None:
            assert state.in_degree(node) == 0
            assert state.out_degree(node) == 1
            assert len(node.out_connectors) == 0
        else:
            assert state.in_degree(node) == 1
            assert state.out_degree(node) == 1
            assert len(node.out_connectors) == 0

        map_entries.add(node)

    assert len(map_entries) == 2


def test_expand_without_dynamic_inputs():

    @dace.program
    def expansion(A: dace.float32[20, 30, 5], rng: dace.int32[2]):

        @dace.map
        def mymap(i: _[0:20], j: _[rng[0]:rng[1]], k: _[0:5]):
            a << A[i, j, k]
            b >> A[i, j, k]
            b = a * 2

    A = np.random.rand(20, 30, 5).astype(np.float32)
    b = np.array([5, 10], dtype=np.int32)
    expected = A.copy()
    expected[:, 5:10, :] *= 2

    sdfg = expansion.to_sdfg()
    sdfg(A=A, rng=b)
    diff = np.linalg.norm(A - expected)
    print('Difference (before transformation):', diff)

    sdfg.apply_transformations(MapExpansion)

    sdfg(A=A, rng=b)
    expected[:, 5:10, :] *= 2
    diff2 = np.linalg.norm(A - expected)
    print('Difference:', diff2)
    assert (diff <= 1e-5) and (diff2 <= 1e-5)


def test_expand_with_limits():

    @dace.program
    def expansion(A: dace.float32[20, 30, 5]):

        @dace.map
        def mymap(i: _[0:20], j: _[0:30], k: _[0:5]):
            a << A[i, j, k]
            b >> A[i, j, k]
            b = a * 2

    A = np.random.rand(20, 30, 5).astype(np.float32)
    expected = A.copy()
    expected *= 2

    sdfg = expansion.to_sdfg()
    sdfg.simplify()
    sdfg(A=A)
    diff = np.linalg.norm(A - expected)
    print('Difference (before transformation):', diff)

    sdfg.apply_transformations(MapExpansion, options=dict(expansion_limit=1))

    map_entries = set()
    state = sdfg.start_state
    for node in state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue

        if state.entry_node(node) is None:
            assert state.in_degree(node) == 1
            assert state.out_degree(node) == 1
            assert len(node.out_connectors) == 1
            assert len(node.map.range.ranges) == 1
            assert node.map.range.ranges[0][1] - node.map.range.ranges[0][0] + 1 == 20
        else:
            assert state.in_degree(node) == 1
            assert state.out_degree(node) == 1
            assert len(node.out_connectors) == 1
            assert len(node.map.range.ranges) == 2
            assert list(map(lambda x: x[1] - x[0] + 1, node.map.range.ranges)) == [30, 5]

        map_entries.add(node)

    sdfg(A=A)
    expected *= 2
    diff2 = np.linalg.norm(A - expected)
    print('Difference:', diff2)
    assert (diff <= 1e-5) and (diff2 <= 1e-5)
    assert len(map_entries) == 2


def test_expand_with_dependency_edges():

    @dace.program
    def expansion(A: dace.float32[2], B: dace.float32[2, 2, 2]):
        for i in dace.map[0:2]:
            A[i] = i

            for j, k in dace.map[0:2, 0:2]:
                B[i, j, k] = i * j + k

    sdfg = expansion.to_sdfg()
    sdfg.simplify()
    sdfg.validate()

    # If dependency edges are handled correctly, this should not raise an exception
    try:
        num_app = sdfg.apply_transformations_repeated(MapExpansion)
    except Exception as e:
        assert False, f"MapExpansion failed: {str(e)}"
    assert num_app == 1
    sdfg.validate()

    A = np.random.rand(2).astype(np.float32)
    B = np.random.rand(2, 2, 2).astype(np.float32)
    sdfg(A=A, B=B)

    A_expected = np.array([0, 1], dtype=np.float32)
    B_expected = np.array([[[0, 1], [0, 1]], [[0, 1], [1, 2]]], dtype=np.float32)
    assert np.all(A == A_expected)
    assert np.all(B == B_expected)


if __name__ == '__main__':
    test_expand_with_inputs()
    test_expand_without_inputs()
    test_expand_without_dynamic_inputs()
    test_expand_with_limits()
    test_expand_with_dependency_edges()
