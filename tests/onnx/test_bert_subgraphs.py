"""
Regression tests for BERT subgraphs
"""
import os

import onnx
import pytest
import torch

from dace.libraries.onnx import ONNXModel

data_directory = os.path.join(os.path.dirname(__file__), "onnx_files")


@pytest.mark.onnx
def test_slice(sdfg_name: str):
    model = onnx.load(os.path.join(data_directory, "slice.onnx"))
    dace_model = ONNXModel(sdfg_name, model, onnx_simplify=False)

    data = torch.ones(2)

    out = dace_model(data=data)
    assert out.shape == (1, ), f"Expected output shape (1,), got {out.shape}"
    assert out[0] == 1.0, f"Expected output value 1.0, got {out[0]}"


@pytest.mark.onnx
def test_reshape(sdfg_name: str):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    dace_model = ONNXModel(sdfg_name, model)
    dace_model()


@pytest.mark.onnx
def test_save_transients(sdfg_name: str):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    transients = {}
    dace_model = ONNXModel(sdfg_name, model, save_transients=transients)
    dace_model()
    assert torch.allclose(transients["bertSLASHembeddingsSLASHReshape_4COLON0"].cpu(),
                          dace_model.weights["bert/embeddings/Reshape_4:0"])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
