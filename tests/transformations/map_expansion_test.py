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
                b >> B[i-1, j]
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
        if sdfg.start_state.entry_node(node) is None:
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

if __name__ == '__main__':
    test_expand_with_inputs()
    test_expand_without_inputs()
    test_expand_without_dynamic_inputs()
