# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for NumPy-style broadcasting in the next-generation frontend's
elementwise mechanism: operand-vs-operand broadcast validation, right-aligned
rank promotion, degenerate-dimension pinning, and rank-mismatch fallback.
"""

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')
M = dace.symbol('M')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_row_vector_broadcast():

    @dace.program
    def row_broadcast(A: dace.float64[N, M], b: dace.float64[M], C: dace.float64[N, M]):
        C[:] = A + b

    tree = nextgen.parse_program(row_broadcast)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert maps[0].node.map.params == ['__i0', '__i1']
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    # The row vector is indexed only by the innermost (right-aligned) parameter
    b_memlet = next(memlet for memlet in tasklets[0].in_memlets.values() if memlet.data == 'b')
    assert '__i1' in str(b_memlet.subset)
    assert '__i0' not in str(b_memlet.subset)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_degenerate_dims_broadcast():

    @dace.program
    def outer_broadcast(A: dace.float64[N, 1], B: dace.float64[1, M], C: dace.float64[N, M]):
        C[:] = A * B

    tree = nextgen.parse_program(outer_broadcast)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert maps[0].node.map.params == ['__i0', '__i1']
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    a_memlet = next(memlet for memlet in tasklets[0].in_memlets.values() if memlet.data == 'A')
    b_memlet = next(memlet for memlet in tasklets[0].in_memlets.values() if memlet.data == 'B')
    # Size-1 dimensions are pinned; the other dimension follows its map parameter
    assert '__i0' in str(a_memlet.subset) and '__i1' not in str(a_memlet.subset)
    assert '__i1' in str(b_memlet.subset) and '__i0' not in str(b_memlet.subset)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_scalar_rhs_into_array_target():

    @dace.program
    def fill_constant(A: dace.float64[N]):
        A[:] = 0.0

    tree = nextgen.parse_program(fill_constant)
    # A constant RHS still maps over the full target extent
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert maps[0].node.map.params == ['__i0']
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_fresh_target_gets_broadcast_shape():

    @dace.program
    def fresh_target(A: dace.float64[N, M], b: dace.float64[M]):
        c = A + b
        c[0, 0] = 1.0

    tree = nextgen.parse_program(fresh_target)
    assert 'c' in tree.containers
    assert tuple(tree.containers['c'].shape) == (N, M)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_operand_rank_exceeds_target_falls_back():

    @dace.program
    def rank_mismatch(A: dace.float64[N, M], B: dace.float64[N]):
        B[:] = A + 1.0

    tree = nextgen.parse_program(rank_mismatch)
    # A 2-D operand cannot be written elementwise into a 1-D subset: the
    # statement becomes a callback rather than mislowered dataflow.
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert not _nodes_of_type(tree, tn.MapScope)
    assert not _nodes_of_type(tree, tn.StatementNode)


if __name__ == '__main__':
    test_row_vector_broadcast()
    test_degenerate_dims_broadcast()
    test_scalar_rhs_into_array_target()
    test_fresh_target_gets_broadcast_shape()
    test_operand_rank_exceeds_target_falls_back()
