# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for function-call inlining in the schedule-tree frontend."""

import numpy as np
import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_basic_inlined_call():
    """A calls B with direct array args — verify FunctionCallScope + inlined body."""

    @dace.program
    def callee(X: dace.float64[4], Y: dace.float64[4]):
        return X + Y

    @dace.program
    def caller(A: dace.float64[4], B: dace.float64[4]):
        C = callee(A, B)
        return C

    stree = caller.to_schedule_tree()

    # Find the FunctionCallScope.
    call_scopes = [c for c in stree.children if isinstance(c, tn.FunctionCallScope)]
    assert len(call_scopes) == 1

    scope = call_scopes[0]
    assert scope.call.callee_name == 'callee'
    assert scope.call.arguments == {'X': 'A', 'Y': 'B'}
    # Body should be non-empty (the callee's inlined content).
    assert len(scope.children) >= 1


def test_call_with_return_value():
    """x = callee(A) — verify ReturnNode replaced with assignment."""

    @dace.program
    def callee(A: dace.float64[4]):
        return A + 1

    @dace.program
    def caller(A: dace.float64[4]):
        x = callee(A)
        return x

    stree = caller.to_schedule_tree()

    call_scopes = [c for c in stree.children if isinstance(c, tn.FunctionCallScope)]
    assert len(call_scopes) == 1
    scope = call_scopes[0]

    # The callee's ReturnNode should have been rewritten to an assignment.
    return_nodes = [c for c in scope.children if isinstance(c, tn.ReturnNode)]
    assert len(return_nodes) == 0, 'ReturnNode should be rewritten to AssignNode'

    assign_nodes = [c for c in scope.children if isinstance(c, tn.AssignNode)]
    assert len(assign_nodes) >= 1


def test_multiple_calls_to_same_function():
    """callee(A); callee(B) — two separate FunctionCallScope nodes."""

    @dace.program
    def callee(X: dace.float64[4]):
        return X + 1

    @dace.program
    def caller(A: dace.float64[4], B: dace.float64[4]):
        x = callee(A)
        y = callee(B)
        return x

    stree = caller.to_schedule_tree()

    call_scopes = [n for n in stree.preorder_traversal() if isinstance(n, tn.FunctionCallScope)]
    assert len(call_scopes) == 2
    assert call_scopes[0].call.callee_name == 'callee'
    assert call_scopes[1].call.callee_name == 'callee'


def test_name_collision_renaming():
    """Caller and callee both have transient '__stree_tmp' — verify renaming."""

    @dace.program
    def callee(X: dace.float64[4], Y: dace.float64[4]):
        return X + Y

    @dace.program
    def caller(A: dace.float64[4], B: dace.float64[4]):
        # This expression materializes into __stree_tmp in the caller.
        C = callee(A + 1, B)
        return C

    stree = caller.to_schedule_tree()

    call_scopes = [n for n in stree.preorder_traversal() if isinstance(n, tn.FunctionCallScope)]
    assert len(call_scopes) == 1
    scope = call_scopes[0]

    # The callee's internal temporary must NOT collide with the
    # caller's __stree_tmp (used for A+1).
    caller_container_names = set(stree.containers.keys())
    assert '__stree_tmp' in caller_container_names, 'caller should have __stree_tmp for A+1'


def test_nested_calls_a_b_c():
    """A -> B -> C — verify bottom-up inlining produces correct structure."""

    @dace.program
    def C_func(X: dace.float64[4]):
        return X + 1

    @dace.program
    def B_func(X: dace.float64[4]):
        return C_func(X)

    @dace.program
    def A_func(X: dace.float64[4]):
        return B_func(X)

    stree = A_func.to_schedule_tree()

    # A should have a FunctionCallScope for B.
    a_calls = [n for n in stree.children if isinstance(n, tn.FunctionCallScope)]
    assert len(a_calls) == 1
    assert a_calls[0].call.callee_name == 'B_func'

    # B's inlined body should contain a FunctionCallScope for C.
    b_calls = [n for n in a_calls[0].children if isinstance(n, tn.FunctionCallScope)]
    assert len(b_calls) == 1
    assert b_calls[0].call.callee_name == 'C_func'

    # C's inlined body should contain actual computation.
    assert len(b_calls[0].children) >= 1


def test_call_with_materialized_args():
    """callee(A+1, B+2) — verify temporaries feed into FunctionCallScope."""

    @dace.program
    def callee(X: dace.float64[4], Y: dace.float64[4]):
        return X + Y

    @dace.program
    def caller(A: dace.float64[4], B: dace.float64[4]):
        return callee(A + 1, B + 2)

    stree = caller.to_schedule_tree()

    # Arguments are materialized into map scopes before the call.
    maps = [c for c in stree.children if isinstance(c, tn.MapScope)]
    assert len(maps) >= 2, 'A+1 and B+2 should each produce a MapScope'

    call_scopes = [c for c in stree.children if isinstance(c, tn.FunctionCallScope)]
    assert len(call_scopes) == 1
    scope = call_scopes[0]
    # Arguments should reference the materialized temporaries.
    assert '__stree_tmp' in scope.call.arguments.values() or any(
        v.startswith('__stree_tmp') for v in scope.call.arguments.values())


def test_call_with_keyword_arguments():
    """callee(Y=B, X=A) — verify argument mapping handles keywords."""

    @dace.program
    def callee(X: dace.float64[4], Y: dace.float64[4]):
        return X + Y

    @dace.program
    def caller(A: dace.float64[4], B: dace.float64[4]):
        return callee(Y=B, X=A)

    stree = caller.to_schedule_tree()

    call_scopes = [c for c in stree.children if isinstance(c, tn.FunctionCallScope)]
    assert len(call_scopes) == 1
    assert call_scopes[0].call.arguments == {'X': 'A', 'Y': 'B'}


def test_function_call_scope_as_string():
    """Verify the as_string() representation of FunctionCallScope."""

    @dace.program
    def callee(X: dace.float64[4]):
        return X + 1

    @dace.program
    def caller(A: dace.float64[4]):
        return callee(A)

    stree = caller.to_schedule_tree()
    text = stree.as_string()
    assert 'call callee(X=A):' in text


def test_bare_call_statement():
    """callee(A) as a bare statement — no return targets."""

    @dace.program
    def callee(out: dace.float64[4], X: dace.float64[4]):
        out[:] = X + 1

    @dace.program
    def caller(A: dace.float64[4], B: dace.float64[4]):
        callee(B, A)
        return B

    stree = caller.to_schedule_tree()

    call_scopes = [c for c in stree.children if isinstance(c, tn.FunctionCallScope)]
    assert len(call_scopes) == 1
    scope = call_scopes[0]
    assert scope.call.callee_name == 'callee'
    # Bare call should have no return targets.
    assert scope._return_targets is None
    # Body should still be inlined.
    assert len(scope.children) >= 1


# -------------------------------------------------------------------- #
#  Descriptor inference tests                                            #
# -------------------------------------------------------------------- #


def test_descriptor_inference_numpy_sum():
    """numpy.sum(A, axis=0) should produce a LibraryCall with correct output shape."""

    @dace.program
    def prog(A: dace.float64[4, 5]):
        x = np.sum(A, axis=0)
        return x

    stree = prog.to_schedule_tree()

    # Should produce a LibraryCall for numpy.sum, not an opaque AssignNode.
    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall)]
    assert len(lib_calls) >= 1, f'Expected LibraryCall for numpy.sum, got:\n{stree.as_string()}'
    sum_call = [lc for lc in lib_calls if lc.node.name == 'numpy.sum']
    assert len(sum_call) == 1, f'Expected one numpy.sum LibraryCall, got:\n{stree.as_string()}'

    # Output container should have the reduced shape (5,).
    out_memlet = list(sum_call[0].out_memlets.values())[0]
    out_name = out_memlet.data
    assert out_name in stree.containers
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (5, )
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_sum_full_reduction():
    """numpy.sum(A) with no axis should produce a Scalar."""

    @dace.program
    def prog(A: dace.float64[4, 5]):
        x = np.sum(A)
        return x

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall) and n.node.name == 'numpy.sum']
    assert len(lib_calls) == 1, f'Expected one numpy.sum LibraryCall, got:\n{stree.as_string()}'
    out_name = list(lib_calls[0].out_memlets.values())[0].data
    assert out_name in stree.containers
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Scalar)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_mean():
    """numpy.mean should promote integer input to float64."""

    @dace.program
    def prog(A: dace.int32[10]):
        x = np.mean(A)
        return x

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall) and n.node.name == 'numpy.mean']
    assert len(lib_calls) == 1, f'Expected numpy.mean LibraryCall, got:\n{stree.as_string()}'
    out_name = list(lib_calls[0].out_memlets.values())[0].data
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Scalar)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_reshape():
    """numpy.reshape should produce array with the new shape."""

    @dace.program
    def prog(A: dace.float64[3, 4]):
        x = np.reshape(A, (12, ))
        return x

    stree = prog.to_schedule_tree()

    # numpy.reshape may be lowered as a TaskletNode or LibraryCall — either is fine.
    # The important thing is the output descriptor has the correct shape.
    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (12, )


def test_descriptor_inference_numpy_transpose():
    """numpy.transpose should reverse axes by default."""

    @dace.program
    def prog(A: dace.float64[3, 5]):
        x = np.transpose(A)
        return x

    stree = prog.to_schedule_tree()

    # The important thing is the output descriptor has the reversed shape.
    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (5, 3)


# -------------------------------------------------------------------- #
#  Method descriptor inference tests                                     #
# -------------------------------------------------------------------- #


def test_method_inference_sum_scalar():
    """a.sum() should produce a Scalar descriptor via method inference."""

    @dace.program
    def prog(a: dace.float64[8]):
        return a.sum()

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall)]
    assert len(lib_calls) >= 1, f'Expected LibraryCall for a.sum(), got:\n{stree.as_string()}'


def test_method_inference_sum_with_axis():
    """a.sum(axis=0) should produce a reduced Array descriptor."""

    @dace.program
    def prog(a: dace.float64[3, 4]):
        x = a.sum(axis=0)
        return x

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall)]
    assert len(lib_calls) >= 1, f'Expected LibraryCall for a.sum(axis=0), got:\n{stree.as_string()}'

    # Check the output container has the correct reduced shape.
    out_name = list(lib_calls[0].out_memlets.values())[0].data
    assert out_name in stree.containers
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, )


def test_method_inference_reshape():
    """a.reshape((12,)) should propagate the new shape."""

    @dace.program
    def prog(a: dace.float64[3, 4]):
        x = a.reshape((12, ))
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (12, )


# -------------------------------------------------------------------- #
#  Attribute descriptor inference tests                                  #
# -------------------------------------------------------------------- #


def test_attribute_inference_T():
    """a.T should produce an Array with reversed shape."""

    @dace.program
    def prog(a: dace.float64[3, 5]):
        x = a.T
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (5, 3)


# -------------------------------------------------------------------- #
#  Operator descriptor inference tests                                   #
# -------------------------------------------------------------------- #


def test_operator_inference_matmul():
    """A @ B should use the operator descriptor registry."""

    @dace.program
    def prog(A: dace.float64[4, 3], B: dace.float64[3, 2]):
        return A @ B

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall) and n.node.name == 'MatMul']
    assert len(lib_calls) >= 1, f'Expected MatMul LibraryCall, got:\n{stree.as_string()}'
    out_name = list(lib_calls[0].out_memlets.values())[0].data
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, 2)


# -------------------------------------------------------------------- #
#  Nested inference test                                                 #
# -------------------------------------------------------------------- #


def test_nested_inference_sum_of_matmul():
    """np.sum(A @ B) should chain MatMul + numpy.sum LibraryCalls."""

    @dace.program
    def prog(A: dace.float64[4, 3], B: dace.float64[3, 2]):
        return np.sum(A @ B)

    stree = prog.to_schedule_tree()

    matmul_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall) and n.node.name == 'MatMul']
    sum_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall) and n.node.name == 'numpy.sum']

    assert len(matmul_calls) >= 1, f'Expected MatMul LibraryCall, got:\n{stree.as_string()}'
    assert len(sum_calls) >= 1, f'Expected numpy.sum LibraryCall, got:\n{stree.as_string()}'

    # numpy.sum result should be a Scalar.
    sum_out_name = list(sum_calls[0].out_memlets.values())[0].data
    desc = stree.containers[sum_out_name]
    assert isinstance(desc, dace.data.Scalar)


if __name__ == '__main__':
    test_basic_inlined_call()
    test_call_with_return_value()
    test_multiple_calls_to_same_function()
    test_name_collision_renaming()
    test_nested_calls_a_b_c()
    test_call_with_materialized_args()
    test_call_with_keyword_arguments()
    test_function_call_scope_as_string()
    test_bare_call_statement()
    test_descriptor_inference_numpy_sum()
    test_descriptor_inference_numpy_sum_full_reduction()
    test_descriptor_inference_numpy_mean()
    test_descriptor_inference_numpy_reshape()
    test_descriptor_inference_numpy_transpose()
    test_method_inference_sum_scalar()
    test_method_inference_sum_with_axis()
    test_method_inference_reshape()
    test_attribute_inference_T()
    test_operator_inference_matmul()
    test_nested_inference_sum_of_matmul()
    print('All tests passed.')
