# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree
import numpy as np


def test_for_in_map_in_for():

    @dace.program
    def matmul(A: dace.float32[10, 10], B: dace.float32[10, 10], C: dace.float32[10, 10]):
        for i in range(10):
            for j in dace.map[0:10]:
                atile = dace.define_local([10], dace.float32)
                atile[:] = A[i]
                for k in range(10):
                    with dace.tasklet:
                        a << atile[k]
                        b << B[k, j]
                        cin << C[i, j]
                        c >> C[i, j]
                        c = cin + a * b

    sdfg = matmul.to_sdfg()
    stree = as_schedule_tree(sdfg)

    assert len(stree.children) == 1  # for
    fornode = stree.children[0]
    assert isinstance(fornode, tn.ForScope)
    assert len(fornode.children) == 1  # map
    mapnode = fornode.children[0]
    assert isinstance(mapnode, tn.MapScope)
    assert len(mapnode.children) == 2  # copy, for
    copynode, fornode = mapnode.children
    assert isinstance(copynode, tn.CopyNode)
    assert isinstance(fornode, tn.ForScope)
    assert len(fornode.children) == 1  # tasklet
    tasklet = fornode.children[0]
    assert isinstance(tasklet, tn.TaskletNode)


def test_libnode():
    M, N, K = (dace.symbol(s) for s in 'MNK')

    @dace.program
    def matmul_lib(a: dace.float64[M, K], b: dace.float64[K, N]):
        return a @ b

    sdfg = matmul_lib.to_sdfg()
    stree = as_schedule_tree(sdfg)
    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.LibraryCall)
    assert (stree.children[0].as_string() ==
            '__return[0:M, 0:N] = library MatMul[alpha=1, beta=0](a[0:M, 0:K], b[0:K, 0:N])')


def test_nesting():

    @dace.program
    def nest2(a: dace.float64[10]):
        a += 1

    @dace.program
    def nest1(a: dace.float64[5, 10]):
        for i in range(5):
            nest2(a[:, i])

    @dace.program
    def main(a: dace.float64[20, 10]):
        nest1(a[:5])
        nest1(a[5:10])
        nest1(a[10:15])
        nest1(a[15:])

    sdfg = main.to_sdfg(simplify=True)
    stree = as_schedule_tree(sdfg)

    # Despite two levels of nesting, immediate children are the 4 for loops
    assert len(stree.children) == 4
    offsets = ['', '5', '10', '15']
    for fornode, offset in zip(stree.children, offsets):
        assert isinstance(fornode, tn.ForScope)
        assert len(fornode.children) == 1  # map
        mapnode = fornode.children[0]
        assert isinstance(mapnode, tn.MapScope)
        assert len(mapnode.children) == 1  # tasklet
        tasklet = mapnode.children[0]
        assert isinstance(tasklet, tn.TaskletNode)
        assert offset in str(next(iter(tasklet.in_memlets.values())))


def test_nesting_view():

    @dace.program
    def nest2(a: dace.float64[40]):
        a += 1

    @dace.program
    def nest1(a):
        for i in range(5):
            subset = a[:, i, :]
            nest2(subset.reshape((40, )))

    @dace.program
    def main(a: dace.float64[20, 10]):
        nest1(a.reshape((4, 5, 10)))

    sdfg = main.to_sdfg()
    stree = as_schedule_tree(sdfg)
    assert any(isinstance(node, tn.ViewNode) for node in stree.children)


def test_nesting_nview():

    @dace.program
    def nest2(a: dace.float64[40]):
        a += 1

    @dace.program
    def nest1(a: dace.float64[4, 5, 10]):
        for i in range(5):
            nest2(a[:, i, :])

    @dace.program
    def main(a: dace.float64[20, 10]):
        nest1(a)

    sdfg = main.to_sdfg()
    stree = as_schedule_tree(sdfg)
    assert any(isinstance(node, tn.NView) for node in stree.children)


def test_irreducible_sub_sdfg():
    sdfg = dace.SDFG('irreducible')
    # Add a simple chain
    s = sdfg.add_state_after(sdfg.add_state_after(sdfg.add_state()))
    # Add an irreducible CFG
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()

    sdfg.add_edge(s, s1, dace.InterstateEdge('a < b'))
    # sdfg.add_edge(s, s2, dace.InterstateEdge('a >= b'))
    sdfg.add_edge(s1, s2, dace.InterstateEdge('b > 9'))
    sdfg.add_edge(s2, s1, dace.InterstateEdge('b < 19'))
    e = sdfg.add_state()
    sdfg.add_edge(s1, e, dace.InterstateEdge('a < 0'))
    sdfg.add_edge(s2, e, dace.InterstateEdge('b < 0'))

    # Add a loop following general block
    sdfg.add_loop(e, sdfg.add_state(), None, 'i', '0', 'i < 10', 'i + 1')

    stree = as_schedule_tree(sdfg)
    node_types = [type(n) for n in stree.preorder_traversal()]
    assert node_types.count(tn.GBlock) == 1  # Only one gblock
    assert node_types[-1] == tn.ForScope  # Check that loop was detected


def test_irreducible_in_loops():
    sdfg = dace.SDFG('irreducible')
    # Add a simple chain of two for loops with goto from second to first's body
    s1 = sdfg.add_state_after(sdfg.add_state_after(sdfg.add_state()))
    s2 = sdfg.add_state()
    e = sdfg.add_state()

    # Add a loop
    l1 = sdfg.add_state()
    l2 = sdfg.add_state_after(l1)
    sdfg.add_loop(s1, l1, s2, 'i', '0', 'i < 10', 'i + 1', loop_end_state=l2)

    l3 = sdfg.add_state()
    l4 = sdfg.add_state_after(l3)
    sdfg.add_loop(s2, l3, e, 'i', '0', 'i < 10', 'i + 1', loop_end_state=l4)

    # Irreducible part
    sdfg.add_edge(l3, l1, dace.InterstateEdge('i < 5'))

    # Avoiding undefined behavior
    sdfg.edges_between(l3, l4)[0].data.condition.as_string = 'i >= 5'

    stree = as_schedule_tree(sdfg)
    node_types = [type(n) for n in stree.preorder_traversal()]
    assert node_types.count(tn.GBlock) == 1
    assert node_types.count(tn.ForScope) == 2


def test_reference():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('n', dace.int32)
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_array('C', [20], dace.float64)
    sdfg.add_reference('ref', [20], dace.float64)

    init = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    end = sdfg.add_state()
    sdfg.add_edge(init, s1, dace.InterstateEdge('n > 0'))
    sdfg.add_edge(init, s2, dace.InterstateEdge('n <= 0'))
    sdfg.add_edge(s1, end, dace.InterstateEdge())
    sdfg.add_edge(s2, end, dace.InterstateEdge())

    s1.add_edge(s1.add_access('A'), None, s1.add_access('ref'), 'set', dace.Memlet('A[0:20]'))
    s2.add_edge(s2.add_access('B'), None, s2.add_access('ref'), 'set', dace.Memlet('B[0:20]'))
    end.add_nedge(end.add_access('ref'), end.add_access('C'), dace.Memlet('ref[0:20]'))

    stree = as_schedule_tree(sdfg)
    nodes = list(stree.preorder_traversal())[1:]
    assert [type(n) for n in nodes] == [tn.IfScope, tn.RefSetNode, tn.ElseScope, tn.RefSetNode, tn.CopyNode]
    assert nodes[1].as_string() == 'ref = refset to A[0:20]'
    assert nodes[3].as_string() == 'ref = refset to B[0:20]'


def test_code_to_code():
    sdfg = dace.SDFG('tester')
    sdfg.add_scalar('scal', dace.int32, transient=True)
    state = sdfg.add_state()
    t1 = state.add_tasklet('a', {}, {'out'}, 'out = 5')
    t2 = state.add_tasklet('b', {'inp'}, {}, 'print(inp)', side_effects=True)
    state.add_edge(t1, 'out', t2, 'inp', dace.Memlet('scal'))

    stree = as_schedule_tree(sdfg)
    assert len(stree.children) == 2
    assert all(isinstance(c, tn.TaskletNode) for c in stree.children)
    assert stree.children[1].as_string().startswith('tasklet(scal')


def test_dyn_map_range():
    H = dace.symbol()
    nnz = dace.symbol('nnz')
    W = dace.symbol()

    @dace.program
    def spmv(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W]):
        b = np.zeros([H], dtype=np.float32)

        for i in dace.map[0:H]:
            for j in dace.map[A_row[i]:A_row[i + 1]]:
                b[i] += A_val[j] * x[A_col[j]]

        return b

    sdfg = spmv.to_sdfg()
    stree = as_schedule_tree(sdfg)
    assert len(stree.children) == 2
    assert all(isinstance(c, tn.MapScope) for c in stree.children)
    mapscope = stree.children[1]
    start, end, dynrangemap = mapscope.children
    assert isinstance(start, tn.DynScopeCopyNode)
    assert isinstance(end, tn.DynScopeCopyNode)
    assert isinstance(dynrangemap, tn.MapScope)


if __name__ == '__main__':
    test_for_in_map_in_for()
    test_libnode()
    test_nesting()
    test_nesting_view()
    test_nesting_nview()
    test_irreducible_sub_sdfg()
    test_irreducible_in_loops()
    test_reference()
    test_code_to_code()
    test_dyn_map_range()
