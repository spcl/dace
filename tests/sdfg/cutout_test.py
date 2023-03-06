# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.sdfg.analysis.cutout import SDFGCutout
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

    cut_sdfg = SDFGCutout.singlestate_cutout(state, node)
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

    cut_sdfg = SDFGCutout.singlestate_cutout(state, *nodes)
    assert cut_sdfg.number_of_nodes() == 1
    assert cut_sdfg.node(0).number_of_nodes() == 7
    assert len(cut_sdfg.arrays) == 5
    assert (not any(a.transient for a in cut_sdfg.arrays.values()))


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
    cut_sdfg = SDFGCutout.singlestate_cutout(state, t2, me, mx)
    cut_sdfg.validate()
    assert cut_sdfg.arrays.keys() == {'B', 'ind', 'D'}

    # Functionality
    B = np.random.rand(20)
    D = np.random.rand(20)
    ind = np.array([5, 10], dtype=np.int32)
    cut_sdfg(B=B, D=D, ind=ind)
    assert not np.allclose(D, B + 2) and np.allclose(D[5:10], B[5:10] + 2)


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
    c = SDFGCutout.singlestate_cutout(sdfg.start_state, *sdfg.start_state.nodes())
    c.validate()


def test_cutout_init_map():
    N = dace.symbol("N")

    @dace.program
    def init(A: dace.int32[N]):
        A[:] = 0

    sdfg = init.to_sdfg()
    c = SDFGCutout.singlestate_cutout(sdfg.start_state, *sdfg.start_state.nodes())
    c.validate()


def test_cutout_alibi_nodes():
    sdfg = dace.SDFG('alibi')

    N = dace.symbol('N')
    M = dace.symbol('M')

    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [M], dace.float64)
    sdfg.add_array('C', [N, M], dace.float64)
    sdfg.add_scalar('tmp1', dace.float64, transient=True)
    sdfg.add_scalar('tmp2', dace.float64, transient=True)
    sdfg.add_scalar('tmp3', dace.float64, transient=True)
    sdfg.add_scalar('tmp4', dace.float64, transient=True)
    sdfg.add_scalar('tmp5', dace.float64, transient=True)
    sdfg.add_scalar('tmp6', dace.float64, transient=True)

    state = sdfg.add_state('state')

    read_a = state.add_read('A')
    read_b = state.add_read('B')
    write_c = state.add_write('C')
    acc_tmp1 = state.add_access('tmp1')
    acc_tmp2 = state.add_access('tmp2')
    acc_tmp3 = state.add_access('tmp3')
    acc_tmp4 = state.add_access('tmp4')
    acc_tmp5 = state.add_access('tmp5')
    acc_tmp6 = state.add_access('tmp6')

    t1 = state.add_tasklet('t1', {'a', 'b'}, {'t'}, 't = a * b')
    t2 = state.add_tasklet('t2', {'i1'}, {'o1', 'o2', 'o3', 'o4'}, 'o1 = i1 * i1\no2 = i1 * 4\no3 = i1 / 2\no4 = i1')
    t3 = state.add_tasklet('t3', {'i1', 'i2'}, {'o'}, 'o = i1 * i2')
    t4 = state.add_tasklet('t4', {'i1', 'i2', 'i3'}, {'o'}, 'o = i1 + i2 + i3')

    map_entry, map_exit = state.add_map('map', dict(i='0:N', j='0:M'))

    state.add_memlet_path(read_a, map_entry, t1, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(read_b, map_entry, t1, dst_conn='b', memlet=dace.Memlet('B[j]'))
    state.add_edge(t1, 't', acc_tmp1, None, dace.Memlet('tmp1'))
    state.add_edge(acc_tmp1, None, t2, 'i1', dace.Memlet('tmp1'))
    state.add_edge(t2, 'o1', acc_tmp2, None, dace.Memlet('tmp2'))
    state.add_edge(t2, 'o2', acc_tmp3, None, dace.Memlet('tmp3'))
    state.add_edge(t2, 'o3', acc_tmp4, None, dace.Memlet('tmp4'))
    state.add_edge(t2, 'o4', acc_tmp5, None, dace.Memlet('tmp5'))
    state.add_edge(acc_tmp3, None, t3, 'i1', dace.Memlet('tmp3'))
    state.add_edge(acc_tmp4, None, t3, 'i2', dace.Memlet('tmp4'))
    state.add_edge(t3, 'o', acc_tmp6, None, dace.Memlet('tmp6'))
    state.add_edge(acc_tmp2, None, t4, 'i1', dace.Memlet('tmp2'))
    state.add_edge(acc_tmp5, None, t4, 'i2', dace.Memlet('tmp5'))
    state.add_edge(acc_tmp6, None, t4, 'i3', dace.Memlet('tmp6'))
    state.add_memlet_path(t4, map_exit, write_c, src_conn='o', memlet=dace.Memlet('C[i, j]'))

    ct = SDFGCutout.singlestate_cutout(state, t4)

    assert ('__cutout_C' in ct.arrays)
    assert ('tmp2' in ct.arrays)
    assert ('tmp5' in ct.arrays)
    assert ('tmp6' in ct.arrays)
    assert ('C' not in ct.arrays)


def test_multistate_cutout_simple_expand():
    sdfg = dace.SDFG('multistate')
    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    s3 = sdfg.add_state('s3')
    s4 = sdfg.add_state('s4')
    s5 = sdfg.add_state('s5')
    s6 = sdfg.add_state('s6')
    s7 = sdfg.add_state('s7')
    s8 = sdfg.add_state('s8')
    s9 = sdfg.add_state('s9')

    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_edge(s1, s3, dace.InterstateEdge())
    sdfg.add_edge(s2, s4, dace.InterstateEdge())
    sdfg.add_edge(s3, s4, dace.InterstateEdge())
    sdfg.add_edge(s4, s5, dace.InterstateEdge())
    sdfg.add_edge(s5, s6, dace.InterstateEdge())
    sdfg.add_edge(s5, s7, dace.InterstateEdge())
    sdfg.add_edge(s6, s8, dace.InterstateEdge())
    sdfg.add_edge(s7, s8, dace.InterstateEdge())
    sdfg.add_edge(s8, s9, dace.InterstateEdge())

    in_translation = dict()
    ct: SDFGCutout = SDFGCutout.multistate_cutout(s6, s7, in_translation=in_translation)
    assert len(ct.nodes()) == 3
    assert (in_translation[s5] in ct.nodes())
    assert (in_translation[s6] in ct.nodes())
    assert (in_translation[s7] in ct.nodes())


def test_multistate_cutout_complex_expand():
    sdfg = dace.SDFG('multistate')
    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    s3 = sdfg.add_state('s3')
    s4 = sdfg.add_state('s4')
    s5 = sdfg.add_state('s5')
    s6 = sdfg.add_state('s6')
    s7 = sdfg.add_state('s7')
    s8 = sdfg.add_state('s8')
    s9 = sdfg.add_state('s9')

    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_edge(s1, s3, dace.InterstateEdge())
    sdfg.add_edge(s2, s4, dace.InterstateEdge())
    sdfg.add_edge(s3, s4, dace.InterstateEdge())
    sdfg.add_edge(s4, s5, dace.InterstateEdge())
    sdfg.add_edge(s5, s6, dace.InterstateEdge())
    sdfg.add_edge(s5, s7, dace.InterstateEdge())
    sdfg.add_edge(s6, s8, dace.InterstateEdge())
    sdfg.add_edge(s7, s8, dace.InterstateEdge())
    sdfg.add_edge(s8, s9, dace.InterstateEdge())

    in_translation = dict()
    ct: SDFGCutout = SDFGCutout.multistate_cutout(s4, s5, s6, s7, in_translation=in_translation)
    assert len(ct.nodes()) == 7
    assert (in_translation[s1] in ct.nodes())
    assert (in_translation[s2] in ct.nodes())
    assert (in_translation[s3] in ct.nodes())
    assert (in_translation[s4] in ct.nodes())
    assert (in_translation[s5] in ct.nodes())
    assert (in_translation[s6] in ct.nodes())
    assert (in_translation[s7] in ct.nodes())


def test_input_output_configuration():
    sdfg = dace.SDFG('silly')
    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    s3 = sdfg.add_state('s3')
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_edge(s2, s3, dace.InterstateEdge())

    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_array('tmp1', [1], dace.float64, transient=True)
    sdfg.add_array('tmp2', [1], dace.float64, transient=True)
    sdfg.add_array('tmp3', [1], dace.float64, transient=True)
    sdfg.add_array('tmp4', [1], dace.float64, transient=True)

    read_a = s1.add_read('A')
    w_tmp1 = s1.add_write('tmp1')
    t1 = s1.add_tasklet('t1', {'a'}, {'b'}, 'b = a')
    s1.add_edge(read_a, None, t1, 'a', dace.Memlet('A[0]'))
    s1.add_edge(t1, 'b', w_tmp1, None, dace.Memlet('tmp1[0]'))

    read_tmp1 = s2.add_read('tmp1')
    w_tmp2 = s2.add_write('tmp2')
    t2 = s2.add_tasklet('t2', {'a'}, {'b'}, 'b = a')
    s2.add_edge(read_tmp1, None, t2, 'a', dace.Memlet('tmp1[0]'))
    s2.add_edge(t2, 'b', w_tmp2, None, dace.Memlet('tmp2[0]'))
    a_tmp3 = s2.add_access('tmp3')
    w_tmp4 = s2.add_write('tmp4')
    t3 = s2.add_tasklet('t3', {}, {'b'}, 'b = 1')
    t4 = s2.add_tasklet('t4', {'a'}, {'b'}, 'b = a')
    s2.add_edge(t3, 'b', a_tmp3, None, dace.Memlet('tmp3[0]'))
    s2.add_edge(a_tmp3, None, t4, 'a', dace.Memlet('tmp3[0]'))
    s2.add_edge(t4, 'b', w_tmp4, None, dace.Memlet('tmp4[0]'))

    read_tmp2 = s3.add_read('tmp2')
    write_b = s3.add_write('B')
    w_tmp3 = s3.add_write('tmp3')
    t5 = s3.add_tasklet('t5', {'a'}, {'b'}, 'b = a')
    t6 = s3.add_tasklet('t6', {}, {'b'}, 'b = 1')
    s3.add_edge(read_tmp2, None, t5, 'a', dace.Memlet('tmp2[0]'))
    s3.add_edge(t5, 'b', write_b, None, dace.Memlet('B[0]'))
    s3.add_edge(t6, 'b', w_tmp3, None, dace.Memlet('tmp3[0]'))

    ct = SDFGCutout.singlestate_cutout(s2, t2, t3, t4)

    assert ct.arrays['tmp1'].transient == False
    assert ct.arrays['tmp2'].transient == False
    assert ct.arrays['tmp3'].transient == True
    assert ct.arrays['tmp4'].transient == True
    assert len(ct.arrays) == 4


if __name__ == '__main__':
    test_cutout_onenode()
    test_cutout_multinode()
    test_cutout_complex_case()
    test_cutout_implicit_array()
    test_cutout_init_map()
    test_cutout_alibi_nodes()
    test_multistate_cutout_simple_expand()
    test_multistate_cutout_complex_expand()
    test_input_output_configuration()
