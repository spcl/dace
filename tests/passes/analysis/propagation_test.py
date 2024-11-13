# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.passes.analysis.propagation import MemletPropagation


def test_nested_conditional_in_map():
    """
    Thanks to the else branch, propagation should correctly identify that only A[0, 0] is being read with volume 1 (in
    the condition check of the branch), and A[i, :] (volume N) is written for each map iteration.
    NOTE: Due to view-based NSDFGs, currently the read is actually A[0, :] with volume N because of the way the nested
          SDFG is constructed. The entire A[0] slice is passed to the nested SDFG and then the read to A[0, 0] happens
          on an interstate edge inside the nested SDFG. This analysis correctly identifies the subset being read, but
          the volume is technically wrong for now. This will be resolved when no-view-NSDFGs are introduced.
    TODO: Revisit when no-view-NSDFGs are introduced.
    """
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def nested_conditional_in_map(A: dace.int32[M, N]):
        for i in dace.map[0:M]:
            if A[0][0]:
                A[i, :] = 1
            else:
                A[i, :] = 2

    sdfg = nested_conditional_in_map.to_sdfg(simplify=True)

    MemletPropagation().apply_pass(sdfg, {})

    assert 'A' in sdfg._possible_reads
    assert str(sdfg._possible_reads['A'].subset) == '0, 0'
    assert sdfg._possible_reads['A'].dynamic == False
    assert sdfg._possible_reads['A'].volume == M
    assert 'A' in sdfg._certain_reads
    assert str(sdfg._certain_reads['A'].subset) == '0, 0'
    assert sdfg._certain_reads['A'].dynamic == False
    assert sdfg._certain_reads['A'].volume == M
    assert 'A' in sdfg._possible_writes
    assert str(sdfg._possible_writes['A'].subset) == '0:M, 0:N'
    assert sdfg._possible_writes['A'].dynamic == False
    assert sdfg._possible_writes['A'].volume == M * N
    assert 'A' in sdfg._certain_writes
    assert str(sdfg._certain_writes['A'].subset) == '0:M, 0:N'
    assert sdfg._certain_writes['A'].dynamic == False
    assert sdfg._certain_writes['A'].volume == M * N

def test_nested_conditional_in_loop_in_map():
    """
    Write in nested SDFG in two-dimensional map nest. 
    Nested map does not iterate over shape of second array dimension.
    --> should approximate write-set of map nest precisely."""
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def nested_conditional_in_loop_in_map(A: dace.int32[M, N]):
        for i in dace.map[0:M]:
            for j in range(0, N - 2, 1):
                if A[0][0]:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
                A[i, j] = A[i, j] * A[i, j]

    sdfg = nested_conditional_in_loop_in_map.to_sdfg(simplify=True)

    MemletPropagation().apply_pass(sdfg, {})

    assert 'A' in sdfg._possible_reads
    assert str(sdfg._possible_reads['A'].subset) == '0, 0'
    assert sdfg._possible_reads['A'].dynamic == False
    assert sdfg._possible_reads['A'].volume == M * (N - 2)
    assert 'A' in sdfg._certain_reads
    assert str(sdfg._certain_reads['A'].subset) == '0, 0'
    assert sdfg._certain_reads['A'].dynamic == False
    assert sdfg._certain_reads['A'].volume == M * (N - 2)
    assert 'A' in sdfg._possible_writes
    assert str(sdfg._possible_writes['A'].subset) == '0:M, 0:N - 2'
    assert sdfg._possible_writes['A'].dynamic == False
    assert sdfg._possible_writes['A'].volume == M * (N - 2)
    assert 'A' in sdfg._certain_writes
    assert str(sdfg._certain_writes['A'].subset) == '0:M, 0:N - 2'
    assert sdfg._certain_writes['A'].dynamic == False
    assert sdfg._certain_writes['A'].volume == M * (N - 2)

def test_2D_map_added_indices():
    """
    2-dimensional array that writes to two-dimensional array with 
    subscript expression that adds two indices 
    --> Approximated write-set of Map is empty
    """

    N = dace.symbol('N')
    M = dace.symbol('M')

    sdfg = dace.SDFG("twoD_map")
    sdfg.add_array("B", (M, N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1",
                                 map_ranges={
                                     '_i': '0:N:1',
                                     '_j': '0:M:1'
                                 },
                                 inputs={},
                                 code="b = 5",
                                 outputs={"b": dace.Memlet("B[_j,_i + _j]")},
                                 output_nodes={"B": a1},
                                 external_edges=True)

    print(sdfg)


if __name__ == '__main__':
    test_nested_conditional_in_map()
    test_nested_conditional_in_loop_in_map()
    #test_2D_map_added_indices()
