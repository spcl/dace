# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


@dace.program
def viewtest(A: dace.float64[20, 20]):
    return A + 1


def test_view_argument():
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=False):
        with pytest.raises(TypeError):
            A = np.random.rand(20, 20)
            viewtest(A.T)


def test_view_argument_override():
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        A = np.random.rand(40, 20)
        result = viewtest(A[20:, :])
        assert np.allclose(result, A[20:, :] + 1)


def test_unpickled_array_is_not_a_view():
    """An array that crossed a pickle boundary is still the whole of its buffer.

    numpy 2.4 rebuilds a pickled array on top of a shared one, so ``.base`` is set even though the
    array covers that buffer whole in the declared layout. Rejecting it made every argument handed
    to an isolated child look like a sub-array."""
    import pickle

    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=False):
        A = pickle.loads(pickle.dumps(np.random.rand(20, 20)))
        assert np.allclose(viewtest(A), A + 1)


def test_contiguous_subarray_is_still_a_view():
    """The relaxation above must not let a genuine sub-array through: it looks into a buffer bigger
    than itself, so its pointer and extent no longer describe the same array the descriptor does."""
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=False):
        A = np.random.rand(40, 20)
        with pytest.raises(TypeError):
            viewtest(A[20:, :])


@dace.program
def rowsum(A: dace.float64[20, 30], out: dace.float64[20]):
    for i in range(20):
        out[i] = np.sum(A[i, :])


@dace.program
def rowsum_fortran(A: dace.data.Array(dace.float64, (20, 30), strides=(1, 20)), out: dace.float64[20]):
    for i in range(20):
        out[i] = np.sum(A[i, :])


def test_fortran_ordered_array_to_c_descriptor_raises():
    """An owning Fortran-ordered array is not a view, yet the SDFG walks it with the C strides its
    descriptor declares. ``astype`` of a transposed array owns its buffer in Fortran order, which is how
    the QE vexx_k kernel handed its G-vector table to DaCe and got every Coulomb factor permuted."""
    A = np.arange(30 * 20, dtype=np.int64).reshape(30, 20).T.astype(np.float64)
    assert A.base is None and A.flags.f_contiguous and not A.flags.c_contiguous
    out = np.zeros(20)
    with pytest.raises(TypeError, match='Fortran-ordered'):
        rowsum(A, out)


def test_c_ordered_array_to_fortran_descriptor_raises():
    """The mirror case: a C-ordered array handed to a column-major descriptor is read transposed too."""
    A = np.random.rand(20, 30)
    out = np.zeros(20)
    with pytest.raises(TypeError, match='C-ordered'):
        rowsum_fortran(A, out)


def test_a_c_array_of_the_transposed_shape_is_the_fortran_array():
    """A C array of shape (30, 20) is the column-major (20, 30) array in the same memory, the usual
    Fortran-interop spelling; the order check refused it (npbench cloudsc's (5, klon) fields in the
    vectorization kernels) because it compared only the rank."""
    A = np.random.rand(30, 20)
    out = np.zeros(20)
    rowsum_fortran(A, out)
    assert np.allclose(out, A.T.sum(axis=1))


def test_fortran_ordered_array_to_fortran_descriptor():
    """A Fortran-ordered array bound to the column-major descriptor it matches is read correctly."""
    A = np.asfortranarray(np.random.rand(20, 30))
    out = np.zeros(20)
    rowsum_fortran(A, out)
    assert np.allclose(out, A.sum(axis=1))


if __name__ == '__main__':
    test_view_argument()
    test_view_argument_override()
    test_unpickled_array_is_not_a_view()
    test_contiguous_subarray_is_still_a_view()
    test_fortran_ordered_array_to_c_descriptor_raises()
    test_c_ordered_array_to_fortran_descriptor_raises()
    test_fortran_ordered_array_to_fortran_descriptor()
