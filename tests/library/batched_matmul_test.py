# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np

import dace
import dace.libraries.blas as blas

from dace.library import change_default


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("pure", dace.float32),
    pytest.param("pure", dace.float64),
    pytest.param("MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param("cuBLAS", dace.float32, marks=pytest.mark.gpu),
    pytest.param("cuBLAS", dace.float64, marks=pytest.mark.gpu)
])
def test_batchmm(implementation: str, dtype):
    b, m, n, k = tuple(dace.symbol(k) for k in 'bmnk')

    @dace.program
    def bmm(A: dtype[b, m, k], B: dtype[b, k, n], C: dtype[b, m, n]):
        C[:] = A @ B

    with change_default(blas, implementation):
        sdfg = bmm.to_sdfg()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        b, m, n, k = 3, 32, 31, 30

        x = np.random.rand(b, m, k).astype(dtype.as_numpy_dtype())
        y = np.random.rand(b, k, n).astype(dtype.as_numpy_dtype())
        z = np.zeros([b, m, n]).astype(dtype.as_numpy_dtype())

        csdfg = sdfg.compile()
        csdfg(A=x, B=y, C=z, b=b, m=m, n=n, k=k)

        ref = x @ y

        assert np.allclose(ref, z)


if __name__ == "__main__":
    test_batchmm("pure", dace.float32)
    test_batchmm("pure", dace.float64)
    test_batchmm("MKL", dace.float32)
    test_batchmm("MKL", dace.float64)
    test_batchmm("cuBLAS", dace.float32)
    test_batchmm("cuBLAS", dace.float64)
