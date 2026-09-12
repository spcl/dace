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


if __name__ == '__main__':
    test_view_argument()
    test_view_argument_override()
    test_unpickled_array_is_not_a_view()
    test_contiguous_subarray_is_still_a_view()
