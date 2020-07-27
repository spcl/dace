import numpy as np

import onnx
import onnxruntime as rt

import dace
from dace.frontend.onnx import ONNXModel


def test_importer_resnet():

    model = onnx.load("/home/orausch/infer.onnx")
    dace_model = ONNXModel("model", model)
    sdfg = dace_model.sdfg
    sdfg.validate()
    sdfg.save("/home/orausch/resnet.sdfg")
    sdfg.states()[0].instrument = dace.InstrumentationType.Timer

    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    test_output = dace_model(test_input)

    sess = rt.InferenceSession("/home/orausch/infer.onnx")

    res = sess.run(["191"], {"input.1": test_input.copy()})

    assert np.allclose(res, test_output, atol=0.00001)


def test_importer_bert():
    model = onnx.load("/home/orausch/bert_infer.onnx")
    dace_model = ONNXModel("bert", model)
    sdfg = dace_model.sdfg
    sdfg.validate()
    sdfg.view()


if __name__ == "__main__":
    test_importer_bert()
    test_importer_resnet()
