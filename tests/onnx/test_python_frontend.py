# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Test the python frontend of onnx nodes
"""

import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
import numpy as np

import dace
import dace.libraries.onnx as donnx

from tests.ml_gpu_utils import DEVICES, is_gpu, run_sdfg


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_matmul(device):

    @dace
    def matmul(inp1: dace.float32[5, 5], inp2: dace.float32[5, 3]):
        out = dace.define_local([5, 3], dace.float32)
        donnx.ONNXMatMul(A=inp1, B=inp2, Y=out)
        return out

    A = np.random.normal(size=(5, 5)).astype(np.float32)
    B = np.random.normal(size=(5, 3)).astype(np.float32)
    if is_gpu(device):
        # No ONNXModel here: build the SDFG explicitly so the GPU variant can drive the
        # experimental CUDA codegen via run_sdfg (host numpy in/out is copied by the SDFG).
        sdfg = matmul.to_sdfg()
        result = run_sdfg(sdfg, device, inp1=A.copy(), inp2=B.copy())
    else:
        result = matmul(inp1=A.copy(), inp2=B.copy())
    np.testing.assert_allclose(A @ B, result, atol=1e-5, rtol=1e-5, err_msg="MatMul output mismatch")


if __name__ == "__main__":
    test_matmul("cpu")
