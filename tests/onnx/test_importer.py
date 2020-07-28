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
    sdfg = dace_model.sdfg
    sdfg.validate()
    sdfg.save("/home/orausch/resnet.sdfg")
    sdfg.states()[0].instrument = dace.InstrumentationType.Timer

    nodes = [node for node in sdfg.states()[0].nodes() if isinstance(node, ONNXOp)]
    check_op(sdfg, sdfg.states()[0], nodes[0])

    test_input = np.random.rand(10, 3, 224, 224).astype(np.float32)
    test_output = dace_model(test_input)

    sess = rt.InferenceSession("/home/orausch/resnet50_torch_infer.onnx")

    res = sess.run(["output"], {"input1": test_input.copy()})

    assert np.allclose(res, test_output, atol=0.00001)


def test_importer_bert():
    model = onnx.load("/home/orausch/bert_infer.onnx")
    dace_model = ONNXModel("bert", model)
    sdfg = dace_model.sdfg
    sdfg.validate()
    sdfg.view()


if __name__ == "__main__":
    test_importer_resnet()
    test_importer_bert()
