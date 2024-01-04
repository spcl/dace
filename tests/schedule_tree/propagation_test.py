# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests schedule tree input/output memlet computation
"""
import dace
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import tree_to_sdfg as t2s, treenodes as tn
from dace.properties import CodeBlock
import numpy as np


def test_stree_propagation_forloop():
    N = dace.symbol('N')

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(1, N):
            a[i] = 2
        a[1] = 1

    stree = tester.to_sdfg().as_schedule_tree()
    stree = t2s.insert_state_boundaries_to_tree(stree)

    node_types = [n for n in stree.preorder_traversal()]
    assert isinstance(node_types[2], tn.ForScope)
    memlet = dace.Memlet('a[1:N]')
    memlet._is_data_src = False
    assert list(node_types[2].output_memlets()) == [memlet]


def test_stree_propagation_symassign():
    # Manually create a schedule tree
    N = dace.symbol('N')
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
        },
        symbols={
            'N': N,
        },
        children=[
            tn.MapScope(node=dace.nodes.MapEntry(dace.nodes.Map('map', ['i'], dace.subsets.Range([(1, N - 1, 1)]))),
                        children=[
                            tn.AssignNode('j', CodeBlock('N + i'), dace.InterstateEdge(assignments=dict(j='N + i'))),
                            tn.TaskletNode(nodes.Tasklet('inner', {}, {'out'}, 'out = inp + 2'),
                                           {'inp': dace.Memlet('A[j]')}, {'out': dace.Memlet('A[j]')}),
                        ]),
        ],
    )
    stree.children[0].parent = stree
    for c in stree.children[0].children:
        c.parent = stree.children[0]

    assert list(stree.children[0].input_memlets()) == [dace.Memlet('A[0:20]', volume=N - 1)]


def test_stree_propagation_dynset():
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
    stree = sdfg.as_schedule_tree()
    assert len(stree.children) == 2
    assert all(isinstance(c, tn.MapScope) for c in stree.children)
    mapscope = stree.children[1]
    _, _, dynrangemap = mapscope.children
    assert isinstance(dynrangemap, tn.MapScope)
    print('internal:', list(dynrangemap.input_memlets()))
    print('external:', list(mapscope.input_memlets()))
    assert list(dynrangemap.input_memlets()) == []
    assert list(mapscope.input_memlets()) == []


if __name__ == '__main__':
    test_stree_propagation_forloop()
    test_stree_propagation_symassign()
    test_stree_propagation_dynset()
