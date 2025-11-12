# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression tests for BERT subgraphs
"""
import os
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import onnx
import torch
from dace.ml import ONNXModel

data_directory = os.path.join(os.path.dirname(__file__), "onnx_files")


@pytest.mark.onnx
def test_slice(sdfg_name: str):
    model = onnx.load(os.path.join(data_directory, "slice.onnx"))
    dace_model = ONNXModel(sdfg_name, model)

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
    output = dace_model()
    # Verify that the transient was saved and matches the computed output
    assert "bertSLASHembeddingsSLASHReshape_4COLON0" in transients, \
        f"Expected transient not found. Available: {list(transients.keys())}"
    # Convert to torch tensors for comparison
    transient_tensor = transients["bertSLASHembeddingsSLASHReshape_4COLON0"]
    if not isinstance(transient_tensor, torch.Tensor):
        transient_tensor = torch.from_numpy(transient_tensor)
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    assert torch.allclose(transient_tensor.cpu(), output.cpu()), \
        "Saved transient does not match computed output"


if __name__ == "__main__":
    test_slice(sdfg_name="test_slice")
    test_reshape(sdfg_name="test_reshape")
    test_save_transients(sdfg_name="test_save_transients")
