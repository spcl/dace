# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.passes.analysis.propagation import MemletPropagation


def test_nested_conditional_in_map():
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

    assert 'A' in sdfg.possible_reads
    assert str(sdfg.possible_reads['A'].subset) == '0, 0'
    assert sdfg.possible_reads['A'].dynamic == False
    assert sdfg.possible_reads['A'].volume == M
    assert 'A' in sdfg.certain_reads
    assert str(sdfg.certain_reads['A'].subset) == '0, 0'
    assert sdfg.certain_reads['A'].dynamic == False
    assert sdfg.certain_reads['A'].volume == M
    assert 'A' in sdfg.possible_writes
    assert str(sdfg.possible_writes['A'].subset) == '0:M, 0:N'
    assert sdfg.possible_writes['A'].dynamic == False
    assert sdfg.possible_writes['A'].volume == M * N
    assert 'A' in sdfg.certain_writes
    assert str(sdfg.certain_writes['A'].subset) == '0:M, 0:N'
    assert sdfg.certain_writes['A'].dynamic == False
    assert sdfg.certain_writes['A'].volume == M * N

def test_nested_conditional_in_loop_in_map():
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

    assert 'A' in sdfg.possible_reads
    assert str(sdfg.possible_reads['A'].subset) == '0, 0'
    assert sdfg.possible_reads['A'].dynamic == False
    assert sdfg.possible_reads['A'].volume == M * (N - 2)
    assert 'A' in sdfg.certain_reads
    assert str(sdfg.certain_reads['A'].subset) == '0, 0'
    assert sdfg.certain_reads['A'].dynamic == False
    assert sdfg.certain_reads['A'].volume == M * (N - 2)
    assert 'A' in sdfg.possible_writes
    assert str(sdfg.possible_writes['A'].subset) == '0:M, 0:N - 2'
    assert sdfg.possible_writes['A'].dynamic == False
    assert sdfg.possible_writes['A'].volume == M * (N - 2)
    assert 'A' in sdfg.certain_writes
    assert str(sdfg.certain_writes['A'].subset) == '0:M, 0:N - 2'
    assert sdfg.certain_writes['A'].dynamic == False
    assert sdfg.certain_writes['A'].volume == M * (N - 2)

def test_runtime_conditional():
    @dace.program
    def rconditional(in1: dace.float64[10], out: dace.float64[10], mask: dace.int32[10]):
        for i in dace.map[1:10]:
            if mask[i] > 0:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    sdfg = rconditional.to_sdfg(simplify=True)

    MemletPropagation().apply_pass(sdfg, {})

    assert 'mask' in sdfg.possible_reads
    assert str(sdfg.possible_reads['mask'].subset) == '1:10'
    assert sdfg.possible_reads['mask'].dynamic == False
    assert sdfg.possible_reads['mask'].volume == 9
    assert 'in1' in sdfg.possible_reads
    assert str(sdfg.possible_reads['in1'].subset) == '0:10'
    assert sdfg.possible_reads['in1'].dynamic == False
    assert sdfg.possible_reads['in1'].volume == 18

    assert 'mask' in sdfg.certain_reads
    assert str(sdfg.certain_reads['mask'].subset) == '1:10'
    assert sdfg.certain_reads['mask'].dynamic == False
    assert sdfg.certain_reads['mask'].volume == 9
    assert 'in1' in sdfg.certain_reads
    assert str(sdfg.certain_reads['in1'].subset) == '0:10'
    assert sdfg.certain_reads['in1'].dynamic == False
    assert sdfg.certain_reads['in1'].volume == 18

    assert 'out' in sdfg.possible_writes
    assert str(sdfg.possible_writes['out'].subset) == '1:10'
    assert sdfg.possible_writes['out'].dynamic == False
    assert sdfg.possible_writes['out'].volume == 9

    assert 'out' in sdfg.certain_writes
    assert str(sdfg.certain_writes['out'].subset) == '1:10'
    assert sdfg.certain_writes['out'].dynamic == False
    assert sdfg.certain_writes['out'].volume == 9

def test_nsdfg_memlet_propagation_with_one_sparse_dimension():
    N = dace.symbol('N')
    M = dace.symbol('M')
    @dace.program
    def sparse(A: dace.float32[M, N], ind: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, ind[i, j]] += 1

    sdfg = sparse.to_sdfg(simplify=False)

    MemletPropagation().apply_pass(sdfg, {})

    assert 'ind' in sdfg.possible_reads
    assert str(sdfg.possible_reads['ind'].subset) == '0:M, 0:N'
    assert sdfg.possible_reads['ind'].dynamic == False
    assert sdfg.possible_reads['ind'].volume == N * M

    assert 'ind' in sdfg.certain_reads
    assert str(sdfg.certain_reads['ind'].subset) == '0:M, 0:N'
    assert sdfg.certain_reads['ind'].dynamic == False
    assert sdfg.certain_reads['ind'].volume == N * M

    assert 'A' in sdfg.possible_writes
    assert str(sdfg.possible_writes['A'].subset) == '0:M, 0:N'
    assert sdfg.possible_writes['A'].dynamic == False
    assert sdfg.possible_writes['A'].volume == N * M

    assert 'A' in sdfg.certain_writes
    assert str(sdfg.certain_writes['A'].subset) == '0:M, 0:N'
    assert sdfg.certain_writes['A'].dynamic == False
    assert sdfg.certain_writes['A'].volume == N * M

def test_nested_loop_in_map():
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def nested_loop_in_map(A: dace.float64[N, M]):
        for i in dace.map[0:N]:
            for j in range(M):
                A[i, j] = 0

    sdfg = nested_loop_in_map.to_sdfg(simplify=True)

    MemletPropagation().apply_pass(sdfg, {})

    assert sdfg.possible_reads == {}
    assert sdfg.certain_reads == {}

    assert 'A' in sdfg.possible_writes
    assert str(sdfg.possible_writes['A'].subset) == '0:N, 0:M'
    assert sdfg.possible_writes['A'].dynamic == False
    assert sdfg.possible_writes['A'].volume == N * M

    assert 'A' in sdfg.certain_writes
    assert str(sdfg.certain_writes['A'].subset) == '0:N, 0:M'
    assert sdfg.certain_writes['A'].dynamic == False
    assert sdfg.certain_writes['A'].volume == N * M


if __name__ == '__main__':
    test_nested_conditional_in_map()
    test_nested_conditional_in_loop_in_map()
    test_runtime_conditional()
    test_nsdfg_memlet_propagation_with_one_sparse_dimension()
    test_nested_loop_in_map()
