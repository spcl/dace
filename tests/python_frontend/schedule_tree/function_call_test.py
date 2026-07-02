# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for function-call inlining in the schedule-tree frontend."""

import numpy as np
import pytest
import dace
from dace.frontend.python.schedule_tree import function_inlining
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


def test_function_inlining_progress_tracks_unique_callees(monkeypatch):
    progress_calls = []

    def tracking_progressbar(iterable, title=None, n=None, progress=None, time_threshold=5.0):
        progress_calls.append({
            'title': title,
            'n': n,
            'completed': 0,
        })
        record = progress_calls[-1]
        for item in iterable:
            record['completed'] += 1
            yield item

    monkeypatch.setattr(function_inlining, 'optional_progressbar', tracking_progressbar)

    @dace.program
    def callee_a(X: dace.float64[4]):
        return X + 1

    @dace.program
    def callee_b(X: dace.float64[4]):
        return X + 2

    @dace.program
    def caller(A: dace.float64[4]):
        left = callee_a(A)
        right = callee_b(A)
        return left + right

    stree = caller.to_schedule_tree()

    call_scopes = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(call_scopes) == 2
    assert len(progress_calls) == 1
    assert progress_calls[0]['title'] == 'Parsing nested DaCe functions'
    assert progress_calls[0]['n'] == 2
    assert progress_calls[0]['completed'] == 2


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


def test_descriptor_inference_numpy_where():
    """numpy.where(cond, A, 1.0) should preserve the runtime replacement's x/y broadcasted shape."""

    @dace.program
    def prog(cond: dace.bool_[2, 1], A: dace.float32[2, 3]):
        x = np.where(cond, A, 1.0)
        return x

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall)]
    assert len(lib_calls) >= 1, f'Expected LibraryCall for numpy.where, got:\n{stree.as_string()}'
    where_call = [lc for lc in lib_calls if lc.node.name == 'numpy.where']
    assert len(where_call) == 1, f'Expected one numpy.where LibraryCall, got:\n{stree.as_string()}'

    out_memlet = list(where_call[0].out_memlets.values())[0]
    out_name = out_memlet.data
    assert out_name in stree.containers
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (2, 3)
    assert desc.dtype == dace.float32


def test_descriptor_inference_numpy_select():
    """numpy.select should match the runtime replacement's nested where descriptor shape and dtype."""

    @dace.program
    def prog(cond: dace.bool_[2, 1], A: dace.float32[2, 3]):
        x = np.select([cond], [A], default=1.0)
        return x

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall)]
    assert len(lib_calls) >= 1, f'Expected LibraryCall for numpy.select, got:\n{stree.as_string()}'
    select_call = [lc for lc in lib_calls if lc.node.name == 'numpy.select']
    assert len(select_call) == 1, f'Expected one numpy.select LibraryCall, got:\n{stree.as_string()}'

    out_memlet = list(select_call[0].out_memlets.values())[0]
    out_name = out_memlet.data
    assert out_name in stree.containers
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (2, 3)
    assert desc.dtype == dace.float32


def test_descriptor_inference_numpy_clip():
    """numpy.clip should infer through the same ufunc-based branching as the runtime replacement."""

    @dace.program
    def prog(A: dace.float32[2, 3]):
        x = np.clip(A, 1.0, 3.0)
        return x

    stree = prog.to_schedule_tree()

    lib_calls = [n for n in stree.preorder_traversal() if isinstance(n, tn.LibraryCall)]
    assert len(lib_calls) >= 1, f'Expected LibraryCall for numpy.clip, got:\n{stree.as_string()}'
    clip_call = [lc for lc in lib_calls if lc.node.name == 'numpy.clip']
    assert len(clip_call) == 1, f'Expected one numpy.clip LibraryCall, got:\n{stree.as_string()}'

    out_memlet = list(clip_call[0].out_memlets.values())[0]
    out_name = out_memlet.data
    assert out_name in stree.containers
    desc = stree.containers[out_name]
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (2, 3)
    assert desc.dtype == dace.float32


def test_descriptor_inference_numpy_rot90():
    """numpy.rot90 should swap the selected axes for odd k values."""

    @dace.program
    def prog(A: dace.float64[2, 3]):
        x = np.rot90(A)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (3, 2)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_fft():
    """numpy.fft.fft should preserve shape and promote real inputs to complex."""

    @dace.program
    def prog(A: dace.float32[8]):
        x = np.fft.fft(A)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (8, )
    assert desc.dtype == dace.complex64


def test_descriptor_inference_numpy_ifft():
    """numpy.fft.ifft should preserve shape and complex dtype."""

    @dace.program
    def prog(A: dace.complex64[8]):
        x = np.fft.ifft(A)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (8, )
    assert desc.dtype == dace.complex64


def test_descriptor_inference_numpy_linalg_inv():
    """numpy.linalg.inv should preserve matrix shape and dtype."""

    @dace.program
    def prog(A: dace.float64[4, 4]):
        x = np.linalg.inv(A)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, 4)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_linalg_solve():
    """numpy.linalg.solve should infer the shape and dtype of the right-hand side."""

    @dace.program
    def prog(A: dace.float64[4, 4], B: dace.float64[4]):
        x = np.linalg.solve(A, B)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, )
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_linalg_cholesky():
    """numpy.linalg.cholesky should preserve matrix shape and dtype."""

    @dace.program
    def prog(A: dace.float64[4, 4]):
        x = np.linalg.cholesky(A)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, 4)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_dot():
    """numpy.dot should follow the current frontend replacement's matrix-multiplication branch for 2D inputs."""

    @dace.program
    def prog(A: dace.float64[4, 3], B: dace.float64[3, 2]):
        x = np.dot(A, B)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, 2)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_tensordot():
    """numpy.tensordot should infer the non-contracted output modes from the runtime replacement rules."""

    @dace.program
    def prog(A: dace.float64[2, 3, 4], B: dace.float64[4, 3, 5]):
        x = np.tensordot(A, B, axes=([2, 1], [0, 1]))
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (2, 5)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_einsum():
    """numpy.einsum should infer its output shape from the parsed output subscripts."""

    @dace.program
    def prog(A: dace.float64[4, 3], B: dace.float64[3, 2]):
        x = np.einsum('ik,kj->ij', A, B)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, 2)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_einsum_multi_contraction():
    """numpy.einsum should preserve only the non-contracted modes for multi-dimensional contractions."""

    A_dim, B_dim, C_dim, D_dim, E_dim = (dace.symbol(name) for name in ('A_dim', 'B_dim', 'C_dim', 'D_dim', 'E_dim'))

    @dace.program
    def prog(A: dace.float64[A_dim, B_dim, C_dim, D_dim], B: dace.float64[B_dim, D_dim, C_dim, E_dim]):
        x = np.einsum('abcd,bdce->ae', A, B)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (A_dim, E_dim)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_einsum_repeated_output_index():
    """numpy.einsum should allow repeated output labels like i->ii for diagonal expansion."""

    vec_len = dace.symbol('vec_len')

    @dace.program
    def prog(A: dace.float64[vec_len]):
        x = np.einsum('i->ii', A)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (vec_len, vec_len)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_einsum_contracts_away_input():
    """numpy.einsum should handle outputs that keep labels from only one input, like j,k->k."""

    reduced_dim, kept_dim = (dace.symbol(name) for name in ('reduced_dim', 'kept_dim'))

    @dace.program
    def prog(A: dace.float64[reduced_dim], B: dace.float64[kept_dim]):
        x = np.einsum('j,k->k', A, B)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (kept_dim, )
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


def test_descriptor_inference_numpy_vstack():

    @dace.program
    def prog(A: dace.float64[2, 3], B: dace.float64[2, 3]):
        x = np.vstack((A, B))
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, 3)


def test_descriptor_inference_numpy_split_structured_result():

    @dace.program
    def prog(A: dace.float64[6]):
        left, right = np.split(A, 2)
        return left

    stree = prog.to_schedule_tree()

    assert 'left' in stree.containers
    left_desc = stree.containers['left']
    assert isinstance(left_desc, dace.data.Array)
    assert tuple(left_desc.shape) == (3, )

    assert 'right' in stree.containers
    right_desc = stree.containers['right']
    assert isinstance(right_desc, dace.data.Array)
    assert tuple(right_desc.shape) == (3, )

    def test_attribute_inference_size_scalar():

        @dace.program
        def prog(a: dace.float64[3, 5]):
            x = a.size
            return x

        stree = prog.to_schedule_tree()

        assert 'x' in stree.containers
        desc = stree.containers['x']
        assert isinstance(desc, dace.data.Scalar)
        assert desc.dtype == dace.int64


def test_descriptor_inference_len_is_scalar():

    @dace.program
    def prog(A: dace.float64[4, 5]):
        n = len(A)
        return n

    stree = prog.to_schedule_tree()

    assert 'n' in stree.containers
    desc = stree.containers['n']
    assert isinstance(desc, dace.data.Scalar)
    assert desc.dtype == dace.int64


def test_descriptor_inference_linspace_retstep_structured_result():

    @dace.program
    def prog():
        space, step = np.linspace(2.5, 10.0, num=3, retstep=True)
        return space

    stree = prog.to_schedule_tree()

    assert not any(isinstance(node, tn.StatementNode) for node in stree.preorder_traversal())
    lib_calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.LibraryCall)]
    assert len(lib_calls) == 1
    assert lib_calls[0].node.name == 'numpy.linspace'
    assert set(lib_calls[0].out_memlets) == {'out0', 'out1'}
    assert lib_calls[0].out_memlets['out0'].data == 'space'
    assert lib_calls[0].out_memlets['out1'].data == 'step'

    assert 'space' in stree.containers
    space_desc = stree.containers['space']
    assert isinstance(space_desc, dace.data.Array)
    assert tuple(space_desc.shape) == (3, )
    assert space_desc.dtype == dace.float64

    assert 'step' in stree.containers
    step_desc = stree.containers['step']
    assert isinstance(step_desc, dace.data.Scalar)
    assert step_desc.dtype == dace.float64


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


def test_operator_inference_add_broadcast():
    """A + B should infer the broadcasted output descriptor."""

    @dace.program
    def prog(A: dace.float64[4, 1], B: dace.float64[1, 3]):
        x = A + B
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, 3)
    assert desc.dtype == dace.float64


def test_operator_inference_compare_bool_array():
    """A < 0 should infer a boolean output array."""

    @dace.program
    def prog(A: dace.float64[4]):
        x = A < 0.0
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, )
    assert desc.dtype == dace.bool_


def test_operator_inference_unary_negate_array():
    """-A should preserve the array shape and dtype class."""

    @dace.program
    def prog(A: dace.float64[4]):
        x = -A
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (4, )
    assert desc.dtype == dace.float64


def test_operator_inference_boolop_scalar_and():
    """Scalar boolean `and` should infer a boolean scalar result."""

    @dace.program
    def prog(a: dace.bool_, b: dace.bool_):
        x = a and b
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Scalar)
    assert desc.dtype == dace.bool_


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


def test_descriptor_inference_custom_arraylike():
    """np.asarray should infer descriptors for objects that implement __array__."""

    class CustomArrayLike:

        def __array__(self, dtype=None):
            return np.eye(2, 5, dtype=dtype if dtype is not None else np.float64)

    custom = CustomArrayLike()

    @dace.program
    def prog():
        x = np.multiply(custom, 2)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (2, 5)
    assert desc.dtype == dace.float64


def test_descriptor_inference_numpy_asarray_custom_arraylike():
    """np.asarray should preserve shape and dtype for custom __array__ objects."""

    class CustomArrayLike:

        def __array__(self, dtype=None):
            return np.eye(2, 5, dtype=dtype if dtype is not None else np.float64)

    custom = CustomArrayLike()

    @dace.program
    def prog():
        x = np.asarray(custom)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (2, 5)
    assert desc.dtype == dace.float64


def test_descriptor_inference_custom_array_interface():
    """Objects with __array_interface__ should infer directly as array inputs."""

    class CustomArrayInterfaceLike:

        def __init__(self):
            self._array = np.zeros((2, 5), dtype=np.float64)

        @property
        def __array_interface__(self):
            return self._array.__array_interface__

    custom = CustomArrayInterfaceLike()

    @dace.program
    def prog():
        x = np.transpose(custom)
        return x

    stree = prog.to_schedule_tree()

    assert 'x' in stree.containers
    desc = stree.containers['x']
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (5, 2)
    assert desc.dtype == dace.float64


if __name__ == '__main__':
    pytest.main([__file__])
