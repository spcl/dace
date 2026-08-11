# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for function-call inlining in the schedule-tree frontend."""

import numpy as np
import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn


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


# -------------------------------------------------------------------- #
#  Descriptor inference tests                                            #
# -------------------------------------------------------------------- #


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


# -------------------------------------------------------------------- #
#  Method descriptor inference tests                                     #
# -------------------------------------------------------------------- #


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

# -------------------------------------------------------------------- #
#  Operator descriptor inference tests                                   #
# -------------------------------------------------------------------- #


def test_operator_inference_add_broadcast():
    """A + B should infer the broadcasted output descriptor."""

    @dace.program
    def prog(A: dace.float64[4, 1], B: dace.float64[1, 3]):
        x = A + B
        return x

    stree = prog.to_schedule_tree()

    # `x` is returned, so `elide_return_copies` renames its container to
    # `__return` and drops the copy that would otherwise materialize it. The
    # inferred descriptor -- which is what this test is about -- is unchanged.
    assert '__return' in stree.containers
    desc = stree.containers['__return']
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
    test_call_with_keyword_arguments()
    test_descriptor_inference_numpy_rot90()
    test_descriptor_inference_numpy_fft()
    test_descriptor_inference_numpy_ifft()
    test_descriptor_inference_numpy_linalg_inv()
    test_descriptor_inference_numpy_linalg_solve()
    test_descriptor_inference_numpy_linalg_cholesky()
    test_descriptor_inference_numpy_dot()
    test_descriptor_inference_numpy_tensordot()
    test_descriptor_inference_numpy_einsum()
    test_descriptor_inference_numpy_einsum_multi_contraction()
    test_descriptor_inference_numpy_einsum_repeated_output_index()
    test_descriptor_inference_numpy_einsum_contracts_away_input()
    test_descriptor_inference_numpy_reshape()
    test_descriptor_inference_numpy_transpose()
    test_descriptor_inference_numpy_vstack()
    test_descriptor_inference_numpy_split_structured_result()
    test_method_inference_reshape()
    test_operator_inference_add_broadcast()
    test_operator_inference_compare_bool_array()
    test_operator_inference_unary_negate_array()
    test_operator_inference_boolop_scalar_and()
    test_descriptor_inference_numpy_asarray_custom_arraylike()
    test_descriptor_inference_custom_array_interface()
