# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the RefineNestedAccess transformation. """

import dace
import numpy as np

from dace.transformation.interstate import RefineNestedAccess


def test_refine_dataflow():

    i = dace.symbol('i')
    j = dace.symbol('j')

    @dace.program
    def inner_sdfg(A: dace.int32[5, 5], B: dace.int32[5, 5]):
        B[i, j] = A[j, i]

    sdfg = dace.SDFG('refine_dataflow')
    sdfg.add_array('A', [5, 5], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    me, mx = state.add_map('m', dict(i='0:5', j='0:5'))
    nsdfg = state.add_nested_sdfg(inner_sdfg.to_sdfg(), sdfg, {'A'}, {'B'}, {'i': 'i', 'j': 'j'})
    state.add_memlet_path(A, me, nsdfg, dst_conn='A', memlet=dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_memlet_path(nsdfg, mx, B, src_conn='B', memlet=dace.Memlet.from_array('B', sdfg.arrays['B']))

    num = sdfg.apply_transformations_repeated(RefineNestedAccess)
    assert num == 1

    for edge in state.out_edges(me):
        assert edge.data.subset == dace.subsets.Range([(j, j, 1), (i, i, 1)])
    for edge in state.in_edges(mx):
        assert edge.data.subset == dace.subsets.Range([(i, i, 1), (j, j, 1)])

    A = np.arange(25, dtype=np.int32).reshape(5, 5).copy()
    B = np.empty((5, 5), dtype=np.int32)
    sdfg(A=A, B=B)
    assert np.allclose(B, A.T)


def test_refine_interstate():

    i = dace.symbol('i')
    j = dace.symbol('j')

    @dace.program
    def inner_sdfg(A: dace.int32[5, 5], B: dace.int32[5, 5], select: dace.bool[5, 5]):
        if select[i, j]:
            B[i, j] = A[j, i]
        else:
            B[i, j] = A[i, j]

    sdfg = dace.SDFG('refine_dataflow')
    sdfg.add_array('A', [5, 5], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)
    sdfg.add_array('select', [5, 5], dace.bool)

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    select = state.add_access('select')
    me, mx = state.add_map('m', dict(i='0:5', j='0:5'))
    nsdfg = state.add_nested_sdfg(inner_sdfg.to_sdfg(), sdfg, {'A', 'select'}, {'B'}, {'i': 'i', 'j': 'j'})
    state.add_memlet_path(A, me, nsdfg, dst_conn='A', memlet=dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_memlet_path(select,
                          me,
                          nsdfg,
                          dst_conn='select',
                          memlet=dace.Memlet.from_array('select', sdfg.arrays['select']))
    state.add_memlet_path(nsdfg, mx, B, src_conn='B', memlet=dace.Memlet.from_array('B', sdfg.arrays['B']))

    num = sdfg.apply_transformations_repeated(RefineNestedAccess)
    assert num == 1

    for edge in state.out_edges(me):
        if edge.data.data == 'A':
            expr = dace.symbolic.pystr_to_symbolic('Max(i, j)')
            assert edge.data.subset == dace.subsets.Range([(0, expr, 1), (0, expr, 1)])
        else:
            assert edge.data.subset == dace.subsets.Range([(i, i, 1), (j, j, 1)])
    for edge in state.in_edges(mx):
        assert edge.data.subset == dace.subsets.Range([(i, i, 1), (j, j, 1)])

    A = np.arange(25, dtype=np.int32).reshape(5, 5).copy()
    B = np.empty((5, 5), dtype=np.int32)
    select = np.empty((5, 5), dtype=np.bool_)
    select[:] = True
    upper = np.triu(select, k=0)
    sdfg(A=A, B=B, select=upper)
    lower = np.tril(A, k=0)
    diag = np.diag(np.diag(A))
    assert np.allclose(B, lower.T + lower - diag)


if __name__ == '__main__':
    test_refine_dataflow()
    test_refine_interstate()
