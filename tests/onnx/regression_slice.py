import copy
import numpy as np

import onnx
import onnxruntime as rt

import dace
from dace.frontend.onnx import ONNXModel

def test_slice():
    model = onnx.load("onnx_files/slice.onnx")
    dace_model = ONNXModel("slice", model, cuda=True)

    out = dace_model(data=np.ones((2,), dtype=np.float32))
    assert out.shape == (1,)
    assert out[0] == 1.0

if __name__ == "__main__":
    test_slice()
