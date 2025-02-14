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


if __name__ == '__main__':
    test_view_argument()
    test_view_argument_override()
