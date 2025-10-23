# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import torch
import torch.nn.functional as F
import dace
import dace.libraries.onnx as donnx


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


# ==============================================================================
# MaxPool Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("kernel_size, stride", [
    ((2, 2), (2, 2)),
    ((3, 3), (2, 2)),
    ((2, 2), (1, 1)),
    ((3, 3), (3, 3)),
])
def test_maxpool_2d(kernel_size, stride, sdfg_name):
    """Test MaxPool operation with different kernel sizes and strides."""

    sdfg = dace.SDFG(sdfg_name)

    batch_size = 2
    channels = 3
    height, width = 8, 8

    inp = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # Compute expected output with PyTorch
    torch_inp = torch.from_numpy(inp)
    expected = F.max_pool2d(torch_inp, kernel_size=kernel_size, stride=stride).numpy()

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXMaxPool("maxpool")
    op_node.kernel_shape = kernel_size
    op_node.strides = stride

    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.onnx
@pytest.mark.parametrize("kernel_size, stride, padding", [
    ((2, 2), (2, 2), (0, 0, 0, 0)),
    ((3, 3), (2, 2), (1, 1, 1, 1)),
])
def test_maxpool_2d_with_padding(kernel_size, stride, padding, sdfg_name):
    """Test MaxPool operation with padding."""

    sdfg = dace.SDFG(sdfg_name)

    batch_size = 2
    channels = 3
    height, width = 8, 8

    inp = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # Compute expected output with PyTorch
    torch_inp = torch.from_numpy(inp)
    torch_padding = (padding[0], padding[2])  # PyTorch uses (left, right, top, bottom) -> (H, W) padding
    expected = F.max_pool2d(torch_inp, kernel_size=kernel_size, stride=stride, padding=torch_padding).numpy()

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXMaxPool("maxpool")
    op_node.kernel_shape = kernel_size
    op_node.strides = stride
    op_node.pads = padding

    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


# ==============================================================================
# BatchNormalization Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("epsilon", [1e-5, 1e-3], ids=["eps_1e-5", "eps_1e-3"])
def test_batch_normalization(epsilon, sdfg_name):
    """Test BatchNormalization operation."""

    sdfg = dace.SDFG(sdfg_name)

    batch_size = 2
    channels = 3
    height, width = 4, 4

    inp = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    scale = np.random.randn(channels).astype(np.float32)
    bias = np.random.randn(channels).astype(np.float32)
    mean = np.random.randn(channels).astype(np.float32)
    var = np.abs(np.random.randn(channels).astype(np.float32)) + 0.1

    # Compute expected output with PyTorch
    torch_inp = torch.from_numpy(inp)
    torch_scale = torch.from_numpy(scale)
    torch_bias = torch.from_numpy(bias)
    torch_mean = torch.from_numpy(mean)
    torch_var = torch.from_numpy(var)

    expected = F.batch_norm(torch_inp, torch_mean, torch_var, torch_scale, torch_bias, training=False,
                            eps=epsilon).numpy()

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("scale", scale.shape, dace.float32)
    sdfg.add_array("bias", bias.shape, dace.float32)
    sdfg.add_array("mean", mean.shape, dace.float32)
    sdfg.add_array("var", var.shape, dace.float32)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXBatchNormalization("batch_norm", epsilon=epsilon)
    # Add connectors
    op_node.add_in_connector("X")
    op_node.add_in_connector("scale")
    op_node.add_in_connector("B")
    op_node.add_in_connector("input_mean")
    op_node.add_in_connector("input_var")
    op_node.add_out_connector("Y")
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("scale"), None, op_node, "scale", sdfg.make_array_memlet("scale"))
    state.add_edge(state.add_read("bias"), None, op_node, "B", sdfg.make_array_memlet("bias"))
    state.add_edge(state.add_read("mean"), None, op_node, "input_mean", sdfg.make_array_memlet("mean"))
    state.add_edge(state.add_read("var"), None, op_node, "input_var", sdfg.make_array_memlet("var"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, scale=scale, bias=bias, mean=mean, var=var)

    assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.onnx
def test_batch_normalization_2d(sdfg_name):
    """Test BatchNormalization on 2D input (batch, channels)."""

    sdfg = dace.SDFG(sdfg_name)

    batch_size = 4
    channels = 5

    inp = np.random.randn(batch_size, channels).astype(np.float32)
    scale = np.random.randn(channels).astype(np.float32)
    bias = np.random.randn(channels).astype(np.float32)
    mean = np.random.randn(channels).astype(np.float32)
    var = np.abs(np.random.randn(channels).astype(np.float32)) + 0.1

    # Compute expected output with PyTorch
    # PyTorch batch_norm requires at least 3D for 1D data, so we reshape
    torch_inp = torch.from_numpy(inp).unsqueeze(-1)  # Add a spatial dimension
    torch_scale = torch.from_numpy(scale)
    torch_bias = torch.from_numpy(bias)
    torch_mean = torch.from_numpy(mean)
    torch_var = torch.from_numpy(var)

    expected = F.batch_norm(torch_inp, torch_mean, torch_var, torch_scale, torch_bias, training=False,
                            eps=1e-5).squeeze(-1).numpy()

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("scale", scale.shape, dace.float32)
    sdfg.add_array("bias", bias.shape, dace.float32)
    sdfg.add_array("mean", mean.shape, dace.float32)
    sdfg.add_array("var", var.shape, dace.float32)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXBatchNormalization("batch_norm", epsilon=1e-5)
    # Add connectors
    op_node.add_in_connector("X")
    op_node.add_in_connector("scale")
    op_node.add_in_connector("B")
    op_node.add_in_connector("input_mean")
    op_node.add_in_connector("input_var")
    op_node.add_out_connector("Y")
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("scale"), None, op_node, "scale", sdfg.make_array_memlet("scale"))
    state.add_edge(state.add_read("bias"), None, op_node, "B", sdfg.make_array_memlet("bias"))
    state.add_edge(state.add_read("mean"), None, op_node, "input_mean", sdfg.make_array_memlet("mean"))
    state.add_edge(state.add_read("var"), None, op_node, "input_var", sdfg.make_array_memlet("var"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, scale=scale, bias=bias, mean=mean, var=var)

    assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


# ==============================================================================
# GlobalAveragePool Tests
# ==============================================================================


@pytest.mark.onnx
def test_global_average_pool(sdfg_name):
    """Test GlobalAveragePool operation."""

    sdfg = dace.SDFG(sdfg_name)

    batch_size = 2
    channels = 3
    height, width = 8, 8

    inp = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # Compute expected output with PyTorch
    torch_inp = torch.from_numpy(inp)
    expected = F.adaptive_avg_pool2d(torch_inp, (1, 1)).numpy()

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXGlobalAveragePool("global_avg_pool")
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.onnx
@pytest.mark.parametrize("height, width", [(4, 4), (6, 8), (10, 10)])
def test_global_average_pool_various_sizes(height, width, sdfg_name):
    """Test GlobalAveragePool with various input sizes."""

    sdfg = dace.SDFG(sdfg_name)

    batch_size = 2
    channels = 3

    inp = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # Compute expected output with PyTorch
    torch_inp = torch.from_numpy(inp)
    expected = F.adaptive_avg_pool2d(torch_inp, (1, 1)).numpy()

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXGlobalAveragePool("global_avg_pool")
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    maxpool_params = [
        ((2, 2), (2, 2)),
        ((3, 3), (2, 2)),
        ((2, 2), (1, 1)),
        ((3, 3), (3, 3)),
    ]
    for kernel_size, stride in maxpool_params:
        test_maxpool_2d(kernel_size=kernel_size, stride=stride, sdfg_name=f"test_maxpool_2d_{kernel_size}_{stride}")

    maxpool_padding_params = [
        ((2, 2), (2, 2), (0, 0, 0, 0)),
        ((3, 3), (2, 2), (1, 1, 1, 1)),
    ]
    for kernel_size, stride, padding in maxpool_padding_params:
        test_maxpool_2d_with_padding(kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     sdfg_name=f"test_maxpool_2d_padding_{kernel_size}_{stride}_{padding}")

    for epsilon in [1e-5, 1e-3]:
        test_batch_normalization(epsilon=epsilon, sdfg_name=f"test_batch_normalization_{epsilon}")

    test_batch_normalization_2d(sdfg_name="test_batch_normalization_2d")

    test_global_average_pool(sdfg_name="test_global_average_pool")

    gap_sizes = [(4, 4), (6, 8), (10, 10)]
    for height, width in gap_sizes:
        test_global_average_pool_various_sizes(height=height,
                                               width=width,
                                               sdfg_name=f"test_global_average_pool_{height}_{width}")
