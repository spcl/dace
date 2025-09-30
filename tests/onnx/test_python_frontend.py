"""
Test the python frontend of onnx nodes
"""

import pytest
import numpy as np

import dace
import dace.libraries.onnx as donnx


@pytest.mark.onnx
def test_matmul():

    @dace
    def matmul(inp1: dace.float32[5, 5], inp2: dace.float32[5, 3]):
        out = dace.define_local([5, 3], dace.float32)
        donnx.ONNXMatMul(A=inp1, B=inp2, Y=out)
        return out

    A = np.random.normal(size=(5, 5)).astype(np.float32)
    B = np.random.normal(size=(5, 3)).astype(np.float32)
    result = matmul(inp1=A.copy(), inp2=B.copy())
    np.testing.assert_allclose(A @ B, result, atol=1e-5, rtol=1e-5, err_msg="MatMul output mismatch")
