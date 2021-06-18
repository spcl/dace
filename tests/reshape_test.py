# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.sdfg import propagation as prop
from dace import nodes
import dace.library
from dace.transformation import transformation as xf


def test_unsqueeze():
    """ Tests for an issue in unsqueeze not allowing reshape. """
    @dace.program
    def callee(A: dace.float64[60, 2]):
        A[:, 1] = 5.0

    @dace.program
    def caller(A: dace.float64[2, 3, 4, 5]):
        callee(A)

    A = np.random.rand(2, 3, 4, 5)
    expected = A[:]
    expected.reshape(60, 2)[:, 1] = 5.0

    sdfg = caller.to_sdfg()
    prop.propagate_memlets_sdfg(sdfg)
    sdfg(A=A)

    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_unsqueeze()
