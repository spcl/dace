import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch
import onnx

from dace.libraries import onnx as donnx


@pytest.mark.onnx
def test_onnx_return_scalars(sdfg_name: str):
    # Dace programs can't return scalars.
    # this test checks that we correctly copy out the scalars using a size [1] array

    # we will have a single operator that computes the sum of a 1D tensor
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [5])

    # Create axes constant with value 0
    axes_constant = onnx.helper.make_tensor(
        name='axes',
        data_type=onnx.TensorProto.INT64,
        dims=[1],  # Single element array
        vals=[0]  # Reduce along axis 0
    )

    # return value is a scalar
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [])

    node_def = onnx.helper.make_node(
        'ReduceSum',
        ['X', "axes"],
        ['Y'],
        keepdims=0,
    )

    graph_def = onnx.helper.make_graph(
        [node_def],
        'test-scalar-return',
        [X],  # inputs
        [Y],  # outputs
        [axes_constant]  # initializers (constants)
    )

    model_def = onnx.helper.make_model(graph_def, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 13)])

    onnx.checker.check_model(model_def)

    # now we can test the backend
    dace_model = donnx.ONNXModel(sdfg_name, model_def)
    inp = torch.arange(5).type(torch.float32)

    result = dace_model(inp)
    assert result.shape == (), f"Expected scalar shape (), got {result.shape}"
    assert result[()] == 1 + 2 + 3 + 4, f"Expected sum 10, got {result[()]}"


if __name__ == "__main__":
    test_onnx_return_scalars(sdfg_name="test_onnx_return_scalars")
