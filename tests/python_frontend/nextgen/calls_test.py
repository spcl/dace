# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for registry-driven call lowering in the next-generation frontend:
descriptor inference through the replacement registry, NumPy ufunc/creation/
reduction mechanisms, and callback fallback for unknown calls.
"""
import numpy as np

import dace
from dace import data, dtypes
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_registry_is_populated():
    from dace.frontend.common import op_repository as oprepo
    assert oprepo.Replacements._dtype_rep, 'descriptor-inference registry is empty'


def test_np_zeros_descriptor_and_fill():

    @dace.program
    def make_zeros(A: dace.float64[10]):
        z = np.zeros(10)
        A[:] = z

    tree = nextgen.parse_program(make_zeros)
    assert 'z' in tree.containers
    assert tuple(tree.containers['z'].shape) == (10, )
    assert tree.containers['z'].dtype == dtypes.float64
    # The fill emits a map with a constant tasklet
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_np_full_value():

    @dace.program
    def make_full(A: dace.float64[10]):
        f = np.full(10, 3.5)
        A[:] = f

    tree = nextgen.parse_program(make_full)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert any('3.5' in t.node.code.as_string for t in tasklets)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_np_empty_allocates_only():

    @dace.program
    def make_empty(A: dace.float64[N]):
        e = np.empty(N)
        e[0] = 1.0
        A[0] = e[0]

    tree = nextgen.parse_program(make_empty)
    assert 'e' in tree.containers
    assert tuple(tree.containers['e'].shape) == (N, )
    # No fill nodes for empty: only the two element assignments
    assert len(_nodes_of_type(tree, tn.TaskletNode)) == 2
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_ufunc_add_elementwise():

    @dace.program
    def ufunc_add(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        C[:] = np.add(A, B)

    tree = nextgen.parse_program(ufunc_add)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert set(tasklets[0].in_memlets.keys()) == {'__in1', '__in2'}
    assert tasklets[0].out_memlets['__out'].data == 'C'
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_ufunc_unary():

    @dace.program
    def ufunc_sin(A: dace.float64[N], B: dace.float64[N]):
        B[:] = np.sin(A)

    tree = nextgen.parse_program(ufunc_sin)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert 'sin' in tasklets[0].node.code.as_string
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_ufunc_broadcasts():

    @dace.program
    def ufunc_broadcast(A: dace.float64[N, N], b: dace.float64[N], C: dace.float64[N, N]):
        C[:] = np.multiply(A, b)

    tree = nextgen.parse_program(ufunc_broadcast)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert maps[0].node.map.params == ['__i0', '__i1']
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    b_memlet = next(memlet for memlet in tasklets[0].in_memlets.values() if memlet.data == 'b')
    assert '__i1' in str(b_memlet.subset) and '__i0' not in str(b_memlet.subset)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_np_sum_reduction_wcr():

    @dace.program
    def sum_all(A: dace.float64[N]):
        s = np.sum(A)
        return s

    tree = nextgen.parse_program(sum_all)
    assert isinstance(tree.containers['s'], data.Scalar)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    # Initialization tasklet writes the identity without inputs
    init = [t for t in tasklets if not t.in_memlets and t.out_memlets['__out'].data == 's']
    assert len(init) == 1
    assert '0' in init[0].node.code.as_string
    # Reduction tasklet carries the WCR on its output memlet
    wcr = [t for t in tasklets if t.out_memlets.get('__out') is not None and t.out_memlets['__out'].wcr is not None]
    assert len(wcr) == 1
    assert 'a + b' in wcr[0].out_memlets['__out'].wcr
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_method_max_first_element_init():

    @dace.program
    def max_all(A: dace.float64[N]):
        m = A.max()
        return m

    tree = nextgen.parse_program(max_all)
    assert isinstance(tree.containers['m'], data.Scalar)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    # max has no identity: initialization reads the first element instead
    init = [t for t in tasklets if t.in_memlets and t.out_memlets['__out'].wcr is None]
    assert len(init) == 1
    assert str(init[0].in_memlets['__in0'].subset) == '0'
    wcr = [t for t in tasklets if t.out_memlets['__out'].wcr is not None]
    assert len(wcr) == 1
    assert 'max' in wcr[0].out_memlets['__out'].wcr
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_sum_axis0():

    @dace.program
    def sum_axis(A: dace.float64[N, 12]):
        s = np.sum(A, axis=0)
        return s

    tree = nextgen.parse_program(sum_axis)
    assert tuple(tree.containers['s'].shape) == (12, )
    maps = _nodes_of_type(tree, tn.MapScope)
    # Initialization map over the kept dimension + full-rank WCR map
    assert len(maps) == 2
    wcr_tasklets = [
        t for t in _nodes_of_type(tree, tn.TaskletNode)
        if t.out_memlets.get('__out') is not None and t.out_memlets['__out'].wcr is not None
    ]
    assert len(wcr_tasklets) == 1
    # The output drops the reduced dimension: indexed by the kept parameter only
    out_subset = str(wcr_tasklets[0].out_memlets['__out'].subset)
    assert '__i1' in out_subset and '__i0' not in out_subset
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_method_max_axis1_first_slice_init():

    @dace.program
    def max_axis(A: dace.float64[N, 12]):
        m = A.max(1)
        return m

    tree = nextgen.parse_program(max_axis)
    assert tuple(tree.containers['m'].shape) == (N, )
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    # max has no identity: the initialization reads the first slice along axis 1
    init = [t for t in tasklets if t.in_memlets and t.out_memlets['__out'].wcr is None]
    assert len(init) == 1
    init_subset = str(init[0].in_memlets['__in0'].subset)
    assert '__i0' in init_subset and '__i1' not in init_subset
    wcr = [t for t in tasklets if t.out_memlets['__out'].wcr is not None]
    assert len(wcr) == 1
    assert 'max' in wcr[0].out_memlets['__out'].wcr
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_sum_negative_axis():

    @dace.program
    def sum_last_axis(A: dace.float64[N, 12]):
        s = np.sum(A, axis=-1)
        return s

    tree = nextgen.parse_program(sum_last_axis)
    # axis=-1 on a 2-D array reduces the second dimension
    assert tuple(tree.containers['s'].shape) == (N, )
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_nonconstant_axis_falls_back():

    @dace.program
    def dynamic_axis(A: dace.float64[N, 12], k: dace.int32):
        s = np.sum(A, axis=k)
        A[0, 0] = 1.0

    tree = nextgen.parse_program(dynamic_axis)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_arange():

    @dace.program
    def make_arange(A: dace.float64[10]):
        r = np.arange(2, 22, 2)
        A[:] = r

    tree = nextgen.parse_program(make_arange)
    assert 'r' in tree.containers
    assert tuple(tree.containers['r'].shape) == (10, )
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    arange_tasklets = [t for t in tasklets if '__i0' in t.node.code.as_string]
    assert len(arange_tasklets) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_unknown_call_falls_back():

    @dace.program
    def unknown_call(A: dace.float64[3, 3]):
        u = np.linalg.svd(A)
        A[0, 0] = 1.0

    tree = nextgen.parse_program(unknown_call)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert 'svd' in callbacks[0].reason or 'no lowering' in callbacks[0].reason
    assert not _nodes_of_type(tree, tn.StatementNode)


if __name__ == '__main__':
    test_registry_is_populated()
    test_np_zeros_descriptor_and_fill()
    test_np_full_value()
    test_np_empty_allocates_only()
    test_ufunc_add_elementwise()
    test_ufunc_unary()
    test_ufunc_broadcasts()
    test_np_sum_reduction_wcr()
    test_method_max_first_element_init()
    test_sum_axis0()
    test_method_max_axis1_first_slice_init()
    test_sum_negative_axis()
    test_nonconstant_axis_falls_back()
    test_arange()
    test_unknown_call_falls_back()
