# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" FPGA Tests for GlobalToLocal transformation"""

import dace
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG, FPGAGlobalToLocal

N = dace.symbol('N')

def test_global_to_local(size: int):
    '''
    Dace program with numpy reshape, transformed for FPGA
    :return:
    '''
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in range(N):
            tmp = A[i]
            B[i] = tmp + 1

    A = np.random.rand(size).astype(np.float32)
    B = np.random.rand(size).astype(np.float32)

    sdfg = program.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.apply_transformations([FPGAGlobalToLocal])
    sdfg.save('/tmp/out.sdfg')
    sdfg(A=A, B=B)
    assert np.allclose(A+1, B)



if __name__ == "__main__":
    test_global_to_local(8)