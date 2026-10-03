# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression tests for BERT subgraphs
"""
import numpy as np
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

from onnx import helper, numpy_helper, TensorProto
import torch
from dace.ml import ONNXModel

from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


def make_slice_model():
    """Create a simple ONNX model with a Slice operation."""
    data_input = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])

    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name='starts')
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name='ends')
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name='axes')

    slice_node = helper.make_node('Slice', inputs=['data', 'starts', 'ends', 'axes'], outputs=['output'])

    graph = helper.make_graph([slice_node],
                              'slice_graph',
                              inputs=[data_input],
                              outputs=[output],
                              initializer=[starts, ends, axes])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
    model.ir_version = 7
    return model


def make_reshape_model():
    """Create an ONNX model simulating BERT embedding reshape operations."""
    output = helper.make_tensor_value_info('bert/embeddings/Reshape_4:0', TensorProto.FLOAT, [1, 256, 768])

    position_embeddings = numpy_helper.from_array(np.random.randn(512, 768).astype(np.float32),
                                                  name='bert/embeddings/position_embeddings:0')
    slice_starts = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name='const_slice__40')
    slice_ends = numpy_helper.from_array(np.array([256, 2147483647], dtype=np.int64), name='const_slice__41')
    reshape_shape = numpy_helper.from_array(np.array([1, 256, 768], dtype=np.int32),
                                            name='bert/embeddings/Reshape_4/shape:0')

    slice_node = helper.make_node(
        'Slice',
        inputs=['bert/embeddings/position_embeddings:0', 'const_slice__40', 'const_slice__41'],
        outputs=['bert/embeddings/Slice:0'])

    cast_node = helper.make_node('Cast',
                                 inputs=['bert/embeddings/Reshape_4/shape:0'],
                                 outputs=['bert/embeddings/Reshape_4__42:0'],
                                 to=TensorProto.INT64)

    reshape_node = helper.make_node('Reshape',
                                    inputs=['bert/embeddings/Slice:0', 'bert/embeddings/Reshape_4__42:0'],
                                    outputs=['bert/embeddings/Reshape_4:0'])

    graph = helper.make_graph([slice_node, cast_node, reshape_node],
                              'reshape_graph',
                              inputs=[],
                              outputs=[output],
                              initializer=[position_embeddings, slice_starts, slice_ends, reshape_shape])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
    model.ir_version = 7
    return model


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_slice(device):
    model = make_slice_model()
    dace_model = ONNXModel(f"test_slice_{device}", model, onnx_simplify=False, cuda=is_gpu(device))

    data = torch.ones(2).to(torch_device(device))

    with experimental_cuda():
        out = dace_model(data=data)
    if is_gpu(device):
        out = out.cpu()
    assert out.shape == (1, ), f"Expected output shape (1,), got {out.shape}"
    assert out[0] == 1.0, f"Expected output value 1.0, got {out[0]}"


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_reshape(device):
    model = make_reshape_model()
    dace_model = ONNXModel(f"test_reshape_{device}", model, cuda=is_gpu(device))
    with experimental_cuda():
        dace_model()


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_save_transients(device):
    model = make_reshape_model()
    transients = {}
    dace_model = ONNXModel(f"test_save_transients_{device}", model, save_transients=transients, cuda=is_gpu(device))
    with experimental_cuda():
        dace_model()
    weight = dace_model.weights["bert/embeddings/Reshape_4:0"]
    if is_gpu(device):
        weight = weight.cpu()
    assert torch.allclose(transients["bertSLASHembeddingsSLASHReshape_4COLON0"].cpu(), weight)


if __name__ == "__main__":
    test_slice("cpu")
    test_reshape("cpu")
    test_save_transients("cpu")
