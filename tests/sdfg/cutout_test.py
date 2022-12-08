# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.sdfg.analysis import cutout
import pytest


def test_cutout_onenode():
    """ Tests cutout on a single node in a state. """

    @dace.program
    def simple_matmul(A: dace.float64[20, 20], B: dace.float64[20, 20]):
        return A @ B + 5

    sdfg = simple_matmul.to_sdfg(simplify=True)
    assert sdfg.number_of_nodes() == 1
    state = sdfg.node(0)
    assert state.number_of_nodes() == 8
    node = next(n for n in state if isinstance(n, dace.nodes.LibraryNode))

    cut_sdfg = cutout.cutout_state(state, node)
    assert cut_sdfg.number_of_nodes() == 1
    assert cut_sdfg.node(0).number_of_nodes() == 4
    assert len(cut_sdfg.arrays) == 3
    assert all(not a.transient for a in cut_sdfg.arrays.values())


def test_cutout_multinode():
    """ Tests cutout on multiple nodes in a state. """

    @dace.program
    def simple_matmul(A: dace.float64[20, 20], B: dace.float64[20, 20]):
        return A @ B + 5

    sdfg = simple_matmul.to_sdfg(simplify=True)
    assert sdfg.number_of_nodes() == 1
    state = sdfg.node(0)
    assert state.number_of_nodes() == 8
    nodes = [n for n in state if isinstance(n, (dace.nodes.LibraryNode, dace.nodes.Tasklet))]
    assert len(nodes) == 2

    cut_sdfg = cutout.cutout_state(state, *nodes)
    assert cut_sdfg.number_of_nodes() == 1
    assert cut_sdfg.node(0).number_of_nodes() == 8
    assert len(cut_sdfg.arrays) == 4
    assert sum([1 if a.transient else 0 for a in cut_sdfg.arrays.values()]) == 1


def test_cutout_complex_case():
    """ Tests cutout on a map with dynamic inputs and two tasklets, which would need two out of three input arrays. """
    # Prepare graph
    sdfg = dace.SDFG('complex')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_array('ind', [2], dace.int32)
    sdfg.add_array('C', [20], dace.float64)
    sdfg.add_array('D', [20], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    b = state.add_read('B')
    i = state.add_read('ind')
    c = state.add_write('C')
    d = state.add_write('D')

    # Map with dynamic range
    me, mx = state.add_map('somemap', dict(i='b:e'))
    me.add_in_connector('b')
    me.add_in_connector('e')
    state.add_edge(i, None, me, 'b', dace.Memlet('ind[0]'))
    state.add_edge(i, None, me, 'e', dace.Memlet('ind[1]'))

    # Two tasklets, one that reads from A and another from B
    t1 = state.add_tasklet('doit1', {'a'}, {'o'}, 'o = a + 1')
    t2 = state.add_tasklet('doit2', {'a'}, {'o'}, 'o = a + 2')
    state.add_memlet_path(a, me, t1, memlet=dace.Memlet('A[i]'), dst_conn='a')
    state.add_memlet_path(b, me, t2, memlet=dace.Memlet('B[i]'), dst_conn='a')
    state.add_memlet_path(t1, mx, c, memlet=dace.Memlet('C[i]'), src_conn='o')
    state.add_memlet_path(t2, mx, d, memlet=dace.Memlet('D[i]'), src_conn='o')

    # Cutout
    cut_sdfg = cutout.cutout_state(state, t2)
    cut_sdfg.validate()
    assert cut_sdfg.arrays.keys() == {'B', 'ind', 'D'}

    # Functionality
    B = np.random.rand(20)
    D = np.random.rand(20)
    ind = np.array([5, 10], dtype=np.int32)
    cut_sdfg(B=B, D=D, ind=ind)
    assert not np.allclose(D, B + 2) and np.allclose(D[5:10], B[5:10] + 2)


def test_cutout_scope_fail():
    """ Tests a case in which implicit cutout expansion should fail due to scope mismatch. """
    # Prepare graph
    sdfg = dace.SDFG('complex')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_transient('local', [2], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    l = state.add_access('local')
    b = state.add_write('B')

    # Tiled map
    ome, omx = state.add_map('somemap', dict(ti='0:20:2'))
    ime, imx = state.add_map('somemap', dict(i='0:2'))

    # A tasklet that reads from local memory
    t = state.add_tasklet('doit', {'a'}, {'o'}, 'o = a + 1')
    state.add_memlet_path(a, ome, l, memlet=dace.Memlet('A[ti:ti+2]'))
    state.add_memlet_path(l, ime, t, memlet=dace.Memlet('local[i]'), dst_conn='a')
    state.add_memlet_path(t, imx, omx, b, memlet=dace.Memlet('B[ti + i]'), src_conn='o')

    # Cutout (should fail)
    with pytest.raises(ValueError):
        cutout.cutout_state(state, t)


def test_cutout_implicit_array():
    N = dace.symbol("N")
    C = dace.symbol("C")
    nnz = dace.symbol("nnz")

    @dace.program
    def spmm(
        A_row: dace.int32[C + 1],
        A_col: dace.int32[nnz],
        A_val: dace.float32[nnz],
        B: dace.float32[C, N],
    ):
        out = dace.define_local((C, N), dtype=B.dtype)

        for i in dace.map[0:C]:
            for j in dace.map[A_row[i]:A_row[i + 1]]:
                for k in dace.map[0:N]:
                    b_col = B[:, k]
                    with dace.tasklet:
                        w << A_val[j]
                        b << b_col[A_col[j]]
                        o >> out(0, lambda x, y: x + y)[i, k]
                        o = w * b

        return out

    sdfg = spmm.to_sdfg()
    c = cutout.cutout_state(sdfg.start_state, *sdfg.start_state.nodes())
    c.validate()


def test_cutout_init_map():
    N = dace.symbol("N")

    @dace.program
    def init(A: dace.int32[N]):
        A[:] = 0

    sdfg = init.to_sdfg()
    c = cutout.cutout_state(sdfg.start_state, *sdfg.start_state.nodes())
    c.validate()


if __name__ == '__main__':
    test_cutout_onenode()
    test_cutout_multinode()
    test_cutout_complex_case()
    test_cutout_scope_fail()
    test_cutout_implicit_array()
    test_cutout_init_map()
