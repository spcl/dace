"""
Regression tests for BERT subgraphs
"""
import os
import numpy as np

import pytest
import onnx
import torch

from dace.libraries.onnx import ONNXModel
from dace.testing import copy_to_gpu

data_directory = os.path.join(os.path.dirname(__file__), "onnx_files")


def test_slice(gpu, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "slice.onnx"))
    dace_model = ONNXModel(sdfg_name, model, cuda=gpu, onnx_simplify=False)

    data = copy_to_gpu(gpu, torch.ones(2))

    out = dace_model(data=data)
    assert out.shape == (1, )
    assert out[0] == 1.0


# this test contains an ORT slice node
def test_reshape(gpu, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    dace_model = ONNXModel(sdfg_name, model, cuda=gpu)
    dace_model()


def test_save_transients(gpu, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    transients = {}
    dace_model = ONNXModel(sdfg_name, model, save_transients=transients, cuda=gpu)
    dace_model()
    assert torch.allclose(transients["bertSLASHembeddingsSLASHReshape_4COLON0"].cpu(),
                          dace_model.weights["bert/embeddings/Reshape_4:0"])


if __name__ == "__main__":
    gpu = False
    sdfg_name = "test_bert_subgraphs"
    test_slice(gpu, sdfg_name)
    test_reshape(gpu, sdfg_name)
    test_save_transients(gpu, sdfg_name)
