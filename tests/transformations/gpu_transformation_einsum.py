# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')

def test_batch_matmul():

    @dace.program
    def einsumtest(A: dace.float64[4, M, N], B: dace.float64[4, N, M], C: dace.float64[4, M, M]):
        C[:] = np.einsum('bik,bkj->bij', A, B)

    sdfg = einsumtest.to_sdfg()

    A = np.random.rand(4, 10, 20)
    B = np.random.rand(4, 20, 10)
    C = np.random.rand(4, 10, 10)
    sdfg(A=A, B=B, C=C, M=10, N=20)
    assert np.allclose(C, A @ B)
    print("CPU testing passed")

    sdfg.apply_gpu_transformations()
    sdfg(A=A, B=B, C=C, M=10, N=20)
    assert np.allclose(C, A @ B)
    print("GPU testing passed")

if __name__ == '__main__':
    test_batch_matmul()
