# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.data import Array
import numpy as np


def myfunction(mytype: Array) -> dace.SDFG:

    @dace.program
    def op(A: mytype):
        return A + A

    return op.to_sdfg()


def test_symbol_in_function():
    N = dace.symbol('N')
    dtype = dace.float32
    sdfg = myfunction(Array(dtype, [N, N]))
    A = np.ones((20, 20), dtype=np.float32)
    B = sdfg(A=A, N=20)
    assert np.allclose(B, 2 * A)


if __name__ == '__main__':
    test_symbol_in_function()
