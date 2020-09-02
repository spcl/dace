import copy
import numpy as np

import onnx

import dace
from dace.frontend.onnx import ONNXModel

if __name__ == "__main__":
    model = onnx.load("onnx_files/reshape.onnx")
    dace_model = ONNXModel("reshape", model, cuda=True)
    out = dace_model()
