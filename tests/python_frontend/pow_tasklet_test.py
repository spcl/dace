# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import dace
import numpy as np

types = [dace.float32, dace.float64, dace.int8, dace.int16, dace.int32, dace.int64]

@pytest.mark.parametrize("a_type", types)
@pytest.mark.parametrize("b_type", types)
def test_tasklet_pow(a_type, b_type):
    """ Tests tasklets containing power operations """

    @dace.program
    def pow(A: a_type[1], B: b_type[1], R: dace.float64[1]):
        @dace.tasklet('Python')
        def pow():
            a << A[0]
            b << B[0]
            r >> R[0]
            """r = a ** b"""

    sdfg = pow.to_sdfg()
    sdfg.validate()

    # a ** b needs to fit into the smallest type (int8)
    a = np.random.rand(1) * 4
    b = np.random.rand(1) * 4
    r = np.random.rand(1).astype(np.float64)

    a = a.astype(a_type.as_numpy_dtype())
    b = b.astype(b_type.as_numpy_dtype())

    sdfg(A=a, B=b, R=r)
    assert np.allclose(r, a ** b)


if __name__ == "__main__":
    for a_type in types:
        for b_type in types:
          test_tasklet_pow(a_type, b_type)
