# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch
import onnx
import numpy as np

from dace.libraries import onnx as donnx


@pytest.mark.onnx
def test_onnx_trilu_upper_default(sdfg_name: str):
    """Test Trilu with default upper triangular (k=0)"""
    # Create input tensor
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [4, 5])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [4, 5])

    # Create Trilu node with upper=1 (default)
    node_def = onnx.helper.make_node(
        'Trilu',
        ['X'],
        ['Y'],
        upper=1,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-upper',
        [X],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    # Test with DaCe
    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(20, dtype=torch.float32).reshape(4, 5)

    result = dace_model(inp)
    expected = torch.triu(inp, diagonal=0)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


@pytest.mark.onnx
def test_onnx_trilu_lower_default(sdfg_name: str):
    """Test Trilu with lower triangular (k=0)"""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [4, 5])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [4, 5])

    # Create Trilu node with upper=0 (lower triangular)
    node_def = onnx.helper.make_node(
        'Trilu',
        ['X'],
        ['Y'],
        upper=0,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-lower',
        [X],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(20, dtype=torch.float32).reshape(4, 5)

    result = dace_model(inp)
    expected = torch.tril(inp, diagonal=0)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


@pytest.mark.onnx
def test_onnx_trilu_upper_with_k_positive(sdfg_name: str):
    """Test Trilu with upper triangular and positive k offset"""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [4, 5])
    k = onnx.helper.make_tensor_value_info('k', onnx.TensorProto.INT64, [])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [4, 5])

    node_def = onnx.helper.make_node(
        'Trilu',
        ['X', 'k'],
        ['Y'],
        upper=1,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-upper-k-pos',
        [X, k],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    k_val = torch.tensor(2, dtype=torch.int64)

    result = dace_model(inp, k_val)
    expected = torch.triu(inp, diagonal=2)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


@pytest.mark.onnx
def test_onnx_trilu_upper_with_k_negative(sdfg_name: str):
    """Test Trilu with upper triangular and negative k offset"""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [4, 5])
    k = onnx.helper.make_tensor_value_info('k', onnx.TensorProto.INT64, [])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [4, 5])

    node_def = onnx.helper.make_node(
        'Trilu',
        ['X', 'k'],
        ['Y'],
        upper=1,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-upper-k-neg',
        [X, k],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    k_val = torch.tensor(-1, dtype=torch.int64)

    result = dace_model(inp, k_val)
    expected = torch.triu(inp, diagonal=-1)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


@pytest.mark.onnx
def test_onnx_trilu_lower_with_k_positive(sdfg_name: str):
    """Test Trilu with lower triangular and positive k offset"""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [4, 5])
    k = onnx.helper.make_tensor_value_info('k', onnx.TensorProto.INT64, [])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [4, 5])

    node_def = onnx.helper.make_node(
        'Trilu',
        ['X', 'k'],
        ['Y'],
        upper=0,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-lower-k-pos',
        [X, k],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    k_val = torch.tensor(1, dtype=torch.int64)

    result = dace_model(inp, k_val)
    expected = torch.tril(inp, diagonal=1)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


@pytest.mark.onnx
def test_onnx_trilu_lower_with_k_negative(sdfg_name: str):
    """Test Trilu with lower triangular and negative k offset"""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [4, 5])
    k = onnx.helper.make_tensor_value_info('k', onnx.TensorProto.INT64, [])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [4, 5])

    node_def = onnx.helper.make_node(
        'Trilu',
        ['X', 'k'],
        ['Y'],
        upper=0,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-lower-k-neg',
        [X, k],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    k_val = torch.tensor(-2, dtype=torch.int64)

    result = dace_model(inp, k_val)
    expected = torch.tril(inp, diagonal=-2)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


@pytest.mark.onnx
def test_onnx_trilu_batched(sdfg_name: str):
    """Test Trilu with batched 3D tensor"""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [2, 4, 5])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [2, 4, 5])

    node_def = onnx.helper.make_node(
        'Trilu',
        ['X'],
        ['Y'],
        upper=1,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-batched',
        [X],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)

    result = dace_model(inp)
    expected = torch.triu(inp, diagonal=0)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


@pytest.mark.onnx
def test_onnx_trilu_square_matrix(sdfg_name: str):
    """Test Trilu with square matrix"""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [5, 5])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [5, 5])

    node_def = onnx.helper.make_node(
        'Trilu',
        ['X'],
        ['Y'],
        upper=1,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-trilu-square',
        [X],
        [Y],
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 14)])
    onnx.checker.check_model(model_def)

    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(25, dtype=torch.float32).reshape(5, 5)

    result = dace_model(inp)
    expected = torch.triu(inp, diagonal=0)

    assert torch.allclose(result, expected), f"Mismatch: got {result}, expected {expected}"


if __name__ == "__main__":
    test_onnx_trilu_upper_default(sdfg_name="test_onnx_trilu_upper_default")
    test_onnx_trilu_lower_default(sdfg_name="test_onnx_trilu_lower_default")
    test_onnx_trilu_upper_with_k_positive(sdfg_name="test_onnx_trilu_upper_with_k_positive")
    test_onnx_trilu_upper_with_k_negative(sdfg_name="test_onnx_trilu_upper_with_k_negative")
    test_onnx_trilu_lower_with_k_positive(sdfg_name="test_onnx_trilu_lower_with_k_positive")
    test_onnx_trilu_lower_with_k_negative(sdfg_name="test_onnx_trilu_lower_with_k_negative")
    test_onnx_trilu_batched(sdfg_name="test_onnx_trilu_batched")
    test_onnx_trilu_square_matrix(sdfg_name="test_onnx_trilu_square_matrix")
