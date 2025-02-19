# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Tests over-approximation of transient shapes.
    In the computation, the result produced by the inner loop
    is stored in a transient container, whose shape should be correctly overapproximated.
"""

import numpy as np
import dace
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG

M, N = (dace.symbol(s, dtype=dace.int32) for s in ('M', 'N'))


@dace.program
def overapprox(alpha: dace.float32, C: dace.float32[N, N], A: dace.float32[N, M]):

    for i in range(N):
        tmp = np.zeros((N, ), dtype=np.float32)
        for k in range(M):
            tmp[:i + 1] += alpha * A[:i + 1, k]
        C[i, :i + 1] = tmp[:i + 1]


def reference(alpha, A, C, N, M):

    for i in range(N):
        tmp = np.zeros((N, ), dtype=np.float32)
        for k in range(M):
            tmp[:i + 1] += alpha * A[:i + 1, k]
        C[i, :i + 1] = tmp[:i + 1]


@fpga_test()
def test_overapprox_transient_shapes():
    size_n = 4
    size_m = 8
    alpha = 1.1
    C = np.random.rand(size_n, size_n).astype(np.float32)
    A = np.random.rand(size_n, size_m).astype(np.float32)
    C_np = np.copy(C)
    sdfg = overapprox.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg(N=size_n, M=size_m, A=A, C=C, alpha=alpha)
    reference(alpha, A, C_np, size_n, size_m)
    assert np.allclose(C_np, C)
    return sdfg


if __name__ == "__main__":
    test_overapprox_transient_shapes(None)
