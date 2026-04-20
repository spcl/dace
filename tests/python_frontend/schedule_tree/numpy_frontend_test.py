# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_python_frontend_schedule_tree_numpy_elementwise_assignment_and_update():

    @dace.program
    def computed(A: dace.float64[8], B: dace.float64[8], out: dace.float64[8]):
        out[:] = A[:] + B[:]
        out[:] += A[:]

    stree = computed.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[0].node, tn.FrontendMap)
    assert isinstance(stree.children[0].children[0], tn.TaskletNode)
    assert isinstance(stree.children[0].children[0].node, tn.FrontendTasklet)
    assert len(stree.children[0].children[0].in_memlets) == 2
    assert isinstance(stree.children[1], tn.MapScope)
    assert isinstance(stree.children[1].node, tn.FrontendMap)
    assert isinstance(stree.children[1].children[0], tn.TaskletNode)
    assert isinstance(stree.children[1].children[0].node, tn.FrontendTasklet)
    assert len(stree.children[1].children[0].in_memlets) == 2


def test_python_frontend_schedule_tree_numpy_broadcast_map():

    @dace.program
    def computed(A: dace.float64[2, 3], B: dace.float64[3], out: dace.float64[2, 3]):
        out[:] = A[:] + B

    stree = computed.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[0].node, tn.FrontendMap)
    assert len(stree.children[0].node.params) == 2
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert str(tasklet.in_memlets['in1'].subset) == '__i1'


def test_python_frontend_schedule_tree_numpy_broadcast_column_assignment():

    @dace.program
    def computed(A: dace.float64[5, 3], B: dace.float64[5, 1], out: dace.float64[5, 3]):
        out[:] = A - B

    stree = computed.to_schedule_tree()

    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 - in1)'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0, __i1'
    assert str(tasklet.in_memlets['in1'].subset) == '__i0, 0'
    assert str(tasklet.out_memlets['out'].subset) == '__i0, __i1'


def test_python_frontend_schedule_tree_numpy_broadcast_prepended_dimension_assignment():

    @dace.program
    def computed(A: dace.float64[5, 3], B: dace.float64[2, 5, 1], out: dace.float64[2, 5, 3]):
        out[:] = A - B

    stree = computed.to_schedule_tree()

    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1', '__i2']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 - in1)'
    assert str(tasklet.in_memlets['in0'].subset) == '__i1, __i2'
    assert str(tasklet.in_memlets['in1'].subset) == '__i0, __i1, 0'
    assert str(tasklet.out_memlets['out'].subset) == '__i0, __i1, __i2'


def test_python_frontend_schedule_tree_numpy_broadcast_both_axes_assignment():

    @dace.program
    def computed(A: dace.float64[5, 1], B: dace.float64[1, 3], out: dace.float64[5, 3]):
        out[:] = A + B

    stree = computed.to_schedule_tree()

    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 + in1)'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0, 0'
    assert str(tasklet.in_memlets['in1'].subset) == '0, __i1'
    assert str(tasklet.out_memlets['out'].subset) == '__i0, __i1'


def test_python_frontend_schedule_tree_numpy_map_inside_loop_with_scalar_index():

    @dace.program
    def computed(A: dace.float64[8], out: dace.float64[8]):
        ref1: dace.data.ArrayReference(A.dtype, A.shape) = A
        ref2: dace.data.ArrayReference(out.dtype, out.shape) = out
        for i in range(8):
            ref2[:] = ref1[:] + i * ref1[2]

    stree = computed.to_schedule_tree()

    assert isinstance(stree.children[0], tn.RefSetNode)
    assert isinstance(stree.children[1], tn.RefSetNode)
    assert isinstance(stree.children[2], tn.LoopScope)
    assert isinstance(stree.children[2].children[0], tn.MapScope)
    tasklet = stree.children[2].children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 + (i * in1))'


def test_python_frontend_schedule_tree_advanced_indexing_is_copy_not_view():

    @dace.program
    def advanced_prog(A: dace.float64[8], ind: dace.int32[4]):
        basic = A[1:5]
        advanced = A[ind]
        return advanced

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.ViewNode)
    assert isinstance(stree.children[1], tn.MapScope)
    assert isinstance(stree.children[1].children[0], tn.TaskletNode)
    assert stree.children[1].children[0].node.code.as_string == 'out = in0[idx0_0]'
    assert isinstance(stree.children[2], tn.ReturnNode)


def test_python_frontend_schedule_tree_advanced_indexing_expression_map():

    @dace.program
    def advanced_prog(A: dace.float64[8], ind: dace.int32[4], B: dace.float64[4], out: dace.float64[4]):
        out[:] = A[ind] + B[:]

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert len(tasklet.in_memlets) == 3
    assert tasklet.node.code.as_string == 'out = (in0[idx0_0] + in1)'


def test_python_frontend_schedule_tree_multidim_advanced_indexing_expression_map():

    @dace.program
    def advanced_prog(A: dace.float64[6, 6], I: dace.int32[4], J: dace.int32[4], out: dace.float64[4]):
        out[:] = A[I, J]

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert len(tasklet.in_memlets) == 3
    assert tasklet.node.code.as_string == 'out = in0[idx0_0, idx0_1]'


def test_python_frontend_schedule_tree_advanced_indexing_target_assign():

    @dace.program
    def advanced_prog(A: dace.float64[8], ind: dace.int32[4]):
        A[ind] = 2

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert len(tasklet.in_memlets) == 1
    assert tasklet.node.code.as_string == 'out[outidx_0] = 2'


def test_python_frontend_schedule_tree_advanced_indexing_target_augassign():

    @dace.program
    def advanced_prog(A: dace.float64[8], ind: dace.int32[4], B: dace.float64[4]):
        A[ind] += B

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert len(tasklet.in_memlets) == 3
    assert tasklet.node.code.as_string == 'out[outidx_0] = (cur + in0)'
    assert str(tasklet.in_memlets['outidx_0'].subset) == '__i0'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0'
    assert str(tasklet.in_memlets['cur'].subset) == '0:8'


def test_python_frontend_schedule_tree_advanced_indexing_target_mixed_range():

    @dace.program
    def advanced_prog(A: dace.float64[20, 20, 20], ind: dace.int32[4]):
        A[1:2, ind, 3:10] = 2

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    assert len(stree.children[0].node.params) == 2
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out[outidx_0] = 2'


def test_python_frontend_schedule_tree_boolean_mask_target_augassign():

    @dace.program
    def advanced_prog(A: dace.float64[20, 30], barr: dace.bool_[20, 30]):
        A[barr] += 5

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    assert len(stree.children[0].node.params) == 2
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert 'if mask:' in tasklet.node.code.as_string
    assert 'out = (cur + 5)' in tasklet.node.code.as_string


def test_python_frontend_schedule_tree_boolean_mask_target_inline_assign():

    @dace.program
    def advanced_prog(A: dace.float64[20, 30]):
        A[A > 15] = 2

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    assert len(stree.children[0].node.params) == 2
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert 'if (in100 > 15):' in tasklet.node.code.as_string
    assert 'out = 2' in tasklet.node.code.as_string


def test_python_frontend_schedule_tree_boolean_mask_read_named_library_call():

    @dace.program
    def advanced_prog(A: dace.float64[20, 30], barr: dace.bool_[20, 30]):
        return A[barr]

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'boolean_mask_gather'
    assert set(stree.children[0].in_memlets.keys()) == {'data', 'mask'}
    result_name = stree.children[0].out_memlets['out'].data
    result_desc = stree.containers[result_name]
    assert isinstance(result_desc, dace.data.Array)
    assert len(result_desc.shape) == 1
    assert result_desc.total_size == 600
    assert str(result_desc.shape[0]).startswith('__stree_mask_nnz')
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_boolean_mask_read_inline_library_call():

    @dace.program
    def advanced_prog(A: dace.float64[20, 30], B: dace.float64[20, 30]):
        return A[(A > 15) & (B < 20)]

    stree = advanced_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'boolean_mask_gather'
    assert 'mask_expr' in stree.children[0].node.properties
    assert 'in100' in stree.children[0].node.properties['mask_expr']
    assert 'in101' in stree.children[0].node.properties['mask_expr']
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_numpy_indirection_update_lowering():

    M, N = (dace.symbol(name) for name in ['M', 'N'])

    @dace.program
    def indirection(A: dace.float64[M], x: dace.int32[N]):
        A[:] = 1.0
        for j in range(1, N):
            A[x[j]] += A[x[j - 1]]

    stree = indirection.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[0].children[0], tn.TaskletNode)
    assert stree.children[0].children[0].node.code.as_string == 'out = 1.0'
    assert isinstance(stree.children[1], tn.LoopScope)
    assert [type(child) for child in stree.children[1].children] == [tn.TaskletNode, tn.TaskletNode, tn.TaskletNode]
    first_idx = stree.children[1].children[0]
    second_idx = stree.children[1].children[1]
    update = stree.children[1].children[2]
    assert first_idx.node.code.as_string == '__stree_idx = x[j]'
    assert str(first_idx.in_memlets['in0'].subset) == 'j'
    assert str(first_idx.out_memlets['out'].subset) == '0'
    assert second_idx.node.code.as_string == '__stree_idx1 = x[(j - 1)]'
    assert str(second_idx.in_memlets['in0'].subset) == 'j - 1'
    assert str(second_idx.out_memlets['out'].subset) == '0'
    assert update.node.code.as_string == 'out = (in0 + in1)'
    assert str(update.in_memlets['in0'].subset) == '__stree_idx'
    assert str(update.in_memlets['in1'].subset) == '__stree_idx1'
    assert str(update.out_memlets['out'].subset) == '__stree_idx'


def test_python_frontend_schedule_tree_numpy_nested_indirection_copy_lowering():

    @dace.program
    def nested(A: dace.float64[50], f: dace.int32[40], g: dace.int32[30], out: dace.float64[1]):
        out[0] = A[f[g[0]]]

    stree = nested.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.TaskletNode, tn.TaskletNode, tn.CopyNode]
    first_idx = stree.children[0]
    second_idx = stree.children[1]
    copy_node = stree.children[2]
    assert first_idx.node.code.as_string == '__stree_idx = g[0]'
    assert str(first_idx.in_memlets['in0'].subset) == '0'
    assert str(first_idx.out_memlets['out'].subset) == '0'
    assert second_idx.node.code.as_string == '__stree_idx1 = f[__stree_idx]'
    assert str(second_idx.in_memlets['in0'].subset) == '__stree_idx'
    assert str(second_idx.out_memlets['out'].subset) == '0'
    assert str(copy_node.memlet) == 'A[__stree_idx1] -> [0]'


def test_python_frontend_schedule_tree_numpy_newaxis_map():

    @dace.program
    def computed(A: dace.float64[2], B: dace.float64[3], out: dace.float64[2, 3]):
        out[:] = A[:, None] + B[None, :]

    stree = computed.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 + in1)'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0'
    assert str(tasklet.in_memlets['in1'].subset) == '__i1'


def test_python_frontend_schedule_tree_numpy_explicit_newaxis_map():

    @dace.program
    def computed(A: dace.float64[2], B: dace.float64[3], out: dace.float64[2, 3]):
        out[:] = A[:, np.newaxis] + B[np.newaxis, :]

    stree = computed.to_schedule_tree()

    assert isinstance(stree.children[0], tn.MapScope)
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 + in1)'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0'
    assert str(tasklet.in_memlets['in1'].subset) == '__i1'


def test_python_frontend_schedule_tree_numpy_explicit_newaxis_return_shape():

    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        return A[:, np.newaxis, np.newaxis, :]

    stree = indexing_test.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1', '__i2', '__i3']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = in0'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0, __i3'
    result_name = tasklet.out_memlets['out'].data
    result_desc = stree.containers[result_name]
    assert isinstance(result_desc, dace.data.Array)
    assert tuple(result_desc.shape) == (20, 1, 1, 30)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_multiple_newaxis_return_shape():

    @dace.program
    def indexing_test(A: dace.float64[10, 20, 30]):
        return A[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis, :, np.newaxis]

    stree = indexing_test.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1', '__i2', '__i3', '__i4', '__i5', '__i6', '__i7']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = in0'
    assert str(tasklet.in_memlets['in0'].subset) == '__i1, __i4, __i6'
    result_name = tasklet.out_memlets['out'].data
    result_desc = stree.containers[result_name]
    assert isinstance(result_desc, dace.data.Array)
    assert tuple(result_desc.shape) == (1, 10, 1, 1, 20, 1, 30, 1)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_ellipsis_return_shape():

    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        return A[1:5, ..., 0]

    stree = indexing_test.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1', '__i2', '__i3']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = in0'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0 + 1, __i1, __i2, __i3, 0'
    result_name = tasklet.out_memlets['out'].data
    result_desc = stree.containers[result_name]
    assert isinstance(result_desc, dace.data.Array)
    assert tuple(result_desc.shape) == (4, 5, 5, 5)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_advanced_indexing_with_newaxes_return_shape():

    @dace.program
    def indexing_test(A: dace.float64[6, 6, 6, 6, 6, 6, 6], indices: dace.int32[3, 3], indices2: dace.int32[3, 3, 3]):
        return A[None, :5, indices, indices2, ..., 1:6:3, 4, np.newaxis]

    stree = indexing_test.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1', '__i2', '__i3', '__i4', '__i5', '__i6', '__i7', '__i8']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = in0[idx0_0, idx0_1]'
    assert str(tasklet.in_memlets['in0'].subset) == '__i4, 0:6, 0:6, __i5, __i6, 3*__i7 + 1, 4'
    assert str(tasklet.in_memlets['idx0_0'].subset) == '__i1, __i2'
    assert str(tasklet.in_memlets['idx0_1'].subset) == '__i0, __i1, __i2'
    result_name = tasklet.out_memlets['out'].data
    result_desc = stree.containers[result_name]
    assert isinstance(result_desc, dace.data.Array)
    assert tuple(result_desc.shape) == (3, 3, 3, 1, 5, 6, 6, 2, 1)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_ufunc_map():

    @dace.program
    def called(A: dace.float64[8], out: dace.float64[8]):
        out[:] = np.sqrt(A[:])

    stree = called.to_schedule_tree()

    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[0].node, tn.FrontendMap)
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = numpy.sqrt(in0)'


def test_python_frontend_schedule_tree_numpy_batched_matmul_library_call():

    @dace.program
    def mmmtest(a: dace.float64[3, 34, 32], b: dace.float64[3, 32, 31]):
        return a @ b

    stree = mmmtest.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'MatMul'
    result_name = stree.children[0].out_memlets['out'].data
    result_desc = stree.containers[result_name]
    assert isinstance(result_desc, dace.data.Array)
    assert tuple(result_desc.shape) == (3, 34, 31)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_batched_matmul_stationary_left_library_call():

    @dace.program
    def mmmtest(a: dace.float64[34, 32], b: dace.float64[3, 32, 31]):
        return a @ b

    stree = mmmtest.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'MatMul'
    result_name = stree.children[0].out_memlets['out'].data
    result_desc = stree.containers[result_name]
    assert isinstance(result_desc, dace.data.Array)
    assert tuple(result_desc.shape) == (3, 34, 31)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_bitxor_pseudoscalar_dtype_inference():

    @dace.program
    def scalar_bitxor_prog(A: dace.int64[5, 5], B: dace.int64[1]):
        return A ^ B

    stree = scalar_bitxor_prog.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 ^ in1)'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0, __i1'
    assert str(tasklet.in_memlets['in1'].subset) == '0'
    result_desc = stree.containers[tasklet.out_memlets['out'].data]
    assert tuple(result_desc.shape) == (5, 5)
    assert result_desc.dtype == dace.int64
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_compare_pseudoscalar_dtype_inference():

    @dace.program
    def scalar_lt_prog(A: dace.int64[5, 5], B: dace.int64[1]):
        return A < B

    stree = scalar_lt_prog.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0', '__i1']
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 < in1)'
    result_desc = stree.containers[tasklet.out_memlets['out'].data]
    assert tuple(result_desc.shape) == (5, 5)
    assert result_desc.dtype == dace.bool_
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0].as_string == '__stree_tmp'


def test_python_frontend_schedule_tree_numpy_transpose_stays_library_call():

    @dace.program
    def called(A: dace.float64[3, 5]):
        return np.transpose(A)

    stree = called.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'numpy.transpose'
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_numpy_method_reshape_stays_library_call():

    @dace.program
    def called(A: dace.float64[3, 4]):
        return A.reshape((12, ))

    stree = called.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'reshape'
    assert stree.children[0].node.properties['receiver_class'] == 'Array'
    assert stree.children[0].node.properties['access_kind'] == 'method'
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_numpy_attribute_stays_library_call():

    @dace.program
    def called(A: dace.float64[3, 5]):
        return A.T

    stree = called.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'T'
    assert stree.children[0].node.properties['receiver_class'] == 'Array'
    assert stree.children[0].node.properties['access_kind'] == 'attribute'
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_nested_numpy_attributes_are_materialized():

    @dace.program
    def called(A: dace.float64[3, 5]):
        return A.T.T

    stree = called.to_schedule_tree()

    library_calls = [node for node in stree.children if isinstance(node, tn.LibraryCall)]
    assert len(library_calls) == 2
    assert all(node.node.name == 'T' for node in library_calls)
    assert all(node.node.properties['access_kind'] == 'attribute' for node in library_calls)
    assert isinstance(stree.children[-1], tn.ReturnNode)


def test_python_frontend_schedule_tree_nested_numpy_attribute_method_chain_is_materialized():

    @dace.program
    def called(A: dace.float64[3, 5]):
        return A.T.T.ravel()

    stree = called.to_schedule_tree()

    library_calls = [node for node in stree.children if isinstance(node, tn.LibraryCall)]
    assert [node.node.name for node in library_calls] == ['T', 'T', 'ravel']
    assert library_calls[0].node.properties['access_kind'] == 'attribute'
    assert library_calls[1].node.properties['access_kind'] == 'attribute'
    assert library_calls[2].node.properties['access_kind'] == 'method'
    assert isinstance(stree.children[-1], tn.ReturnNode)


def test_python_frontend_schedule_tree_numpy_compiletime_full_slice_lowers_to_map():

    @dace.program
    def sliceprog(A: dace.float64[20], slc: dace.compiletime):
        A[slc] += 5

    stree = sliceprog.to_schedule_tree(slc=slice(None, None, None))

    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0']
    assert stree.children[0].node.ranges == [('0', '20', '1')]
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = (in0 + 5)'
    assert str(tasklet.in_memlets['in0'].subset) == '__i0'
    assert str(tasklet.out_memlets['out'].subset) == '__i0'


def test_python_frontend_schedule_tree_numpy_literal_slice_lowers_to_map():

    @dace.program
    def slicer(A: dace.float64[20]):
        A[slice(2, 10, 2)] = 2

    stree = slicer.to_schedule_tree()

    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.MapScope)
    assert stree.children[0].node.params == ['__i0']
    assert stree.children[0].node.ranges == [('2', '10', '2')]
    tasklet = stree.children[0].children[0]
    assert isinstance(tasklet, tn.TaskletNode)
    assert tasklet.node.code.as_string == 'out = 2'
    assert str(tasklet.out_memlets['out'].subset) == '__i0'
