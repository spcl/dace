import numpy as np

import onnx
import onnxruntime as rt

import dace
from dace.frontend.onnx import ONNXModel
from dace.libraries.onnx import check_op
from dace.libraries.onnx.nodes.onnx_op import ONNXOp


def test_importer_resnet():

    model = onnx.load("/home/orausch/resnet50_torch_infer.onnx")
    dace_model = ONNXModel("model", model)

    test_input = np.random.rand(10, 3, 224, 224).astype(np.float32)
    test_output = dace_model(test_input)
    #test_output2 = dace_model(test_input)

if __name__ == "__main__":
    test_importer_resnet()
