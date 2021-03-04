# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" FPGA Tests for reshaping and reinterpretation of existing arrays.
    These are based on numpy/reshape_test"""
import dace
import numpy as np
import pytest
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG, GPUTransformSDFG

N = dace.symbol('N')


def test_reshape_np():
    @dace.program
    def reshp_np(A: dace.float32[3, 4], B: dace.float32[2, 6]):
        B[:] = np.reshape(A, [2, 6])

    A = np.random.rand(3, 4).astype(np.float32)
    B = np.random.rand(2, 6).astype(np.float32)

    sdfg = reshp_np.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg(A=A, B=B)
    assert np.allclose(np.reshape(A, [2, 6]), B)


if __name__ == "__main__":
    test_reshape_np()
