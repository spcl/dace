# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import torch
import dace
import dace.libraries.onnx as donnx


def assert_allclose(a, b, rtol=1e-4, atol=1e-5):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


# ==============================================================================
# Softmax Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_softmax(axis, sdfg_name):
    """Test Softmax operation along different axes."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(2, 3, 4).astype(np.float32)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("__return", inp.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXSoftmax("softmax", axis=axis)
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "input", sdfg.make_array_memlet("inp"))
    state.add_edge(op_node, "output", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    # Compute expected with PyTorch
    expected = torch.softmax(torch.from_numpy(inp), dim=axis).numpy()

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.onnx
def test_softmax_2d(sdfg_name):
    """Test Softmax operation on 2D array."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(5, 10).astype(np.float32)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("__return", inp.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXSoftmax("softmax", axis=1)
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "input", sdfg.make_array_memlet("inp"))
    state.add_edge(op_node, "output", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    # Compute expected with PyTorch
    expected = torch.softmax(torch.from_numpy(inp), dim=1).numpy()

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


# ==============================================================================
# LogSoftmax Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_log_softmax(axis, sdfg_name):
    """Test LogSoftmax operation along different axes."""

    @dace.program
    def log_softmax_prog(inp: dace.float32[2, 3, 4]):
        result = dace.define_local([2, 3, 4], dace.float32)
        donnx.ONNXLogSoftmax(input=inp, output=result, axis=axis)
        return result

    log_softmax_prog.__name__ = sdfg_name

    inp = np.random.randn(2, 3, 4).astype(np.float32)

    sdfg = log_softmax_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    # Compute expected with PyTorch
    expected = torch.log_softmax(torch.from_numpy(inp), dim=axis).numpy()

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.onnx
def test_log_softmax_2d(sdfg_name):
    """Test LogSoftmax operation on 2D array."""

    @dace.program
    def log_softmax_prog(inp: dace.float32[5, 10]):
        result = dace.define_local([5, 10], dace.float32)
        donnx.ONNXLogSoftmax(input=inp, output=result, axis=1)
        return result

    log_softmax_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 10).astype(np.float32)

    sdfg = log_softmax_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    # Compute expected with PyTorch
    expected = torch.log_softmax(torch.from_numpy(inp), dim=1).numpy()

    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


# ==============================================================================
# LayerNormalization Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("normalized_shape", [[4], [3, 4], [2, 3, 4]])
def test_layer_normalization(normalized_shape, sdfg_name):
    """Test LayerNormalization operation with different normalized shapes."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.ones(normalized_shape, dtype=np.float32)
    bias = np.zeros(normalized_shape, dtype=np.float32)

    # Compute axis based on normalized_shape
    axis = inp.ndim - len(normalized_shape)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("scale", scale.shape, dace.float32)
    sdfg.add_array("bias", bias.shape, dace.float32)
    sdfg.add_array("__return", inp.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXLayerNormalization("layer_norm", axis=axis, epsilon=1e-5, optional={'B'})
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("scale"), None, op_node, "Scale", sdfg.make_array_memlet("scale"))
    state.add_edge(state.add_read("bias"), None, op_node, "B", sdfg.make_array_memlet("bias"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, scale=scale, bias=bias)

    # Compute expected with PyTorch
    torch_inp = torch.from_numpy(inp)
    torch_scale = torch.from_numpy(scale)
    torch_bias = torch.from_numpy(bias)
    expected = torch.nn.functional.layer_norm(torch_inp, normalized_shape, torch_scale, torch_bias, eps=1e-5).numpy()

    assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.onnx
def test_layer_normalization_2d(sdfg_name):
    """Test LayerNormalization on 2D input."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(5, 10).astype(np.float32)
    scale = np.ones([10], dtype=np.float32)
    bias = np.zeros([10], dtype=np.float32)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("scale", scale.shape, dace.float32)
    sdfg.add_array("bias", bias.shape, dace.float32)
    sdfg.add_array("__return", inp.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXLayerNormalization("layer_norm", axis=1, epsilon=1e-5, optional={'B'})
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "X", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("scale"), None, op_node, "Scale", sdfg.make_array_memlet("scale"))
    state.add_edge(state.add_read("bias"), None, op_node, "B", sdfg.make_array_memlet("bias"))
    state.add_edge(op_node, "Y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, scale=scale, bias=bias)

    # Compute expected with PyTorch
    torch_inp = torch.from_numpy(inp)
    torch_scale = torch.from_numpy(scale)
    torch_bias = torch.from_numpy(bias)
    expected = torch.nn.functional.layer_norm(torch_inp, [10], torch_scale, torch_bias, eps=1e-5).numpy()

    assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


# ==============================================================================
# Dropout Tests
# ==============================================================================


@pytest.mark.onnx
def test_dropout_inference(sdfg_name):
    """Test Dropout operation in inference mode (ratio=0 or no training)."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(5, 10).astype(np.float32)
    ratio = np.array([0.0], dtype=np.float32)  # No dropout in inference

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("ratio", ratio.shape, dace.float32)
    sdfg.add_array("__return", inp.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXDropout("dropout", optional={'ratio'})
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "data", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("ratio"), None, op_node, "ratio", sdfg.make_array_memlet("ratio"))
    state.add_edge(op_node, "output", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, ratio=ratio)

    # In inference mode with ratio=0, output should be same as input
    assert_allclose(result, inp)


@pytest.mark.onnx
def test_dropout_no_ratio(sdfg_name):
    """Test Dropout operation without ratio input (defaults to 0.5)."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(5, 10).astype(np.float32)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("__return", inp.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXDropout("dropout")
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "data", sdfg.make_array_memlet("inp"))
    state.add_edge(op_node, "output", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    # Without training mode and with default/no ratio, output should be same as input
    # (implementation dependent, but typically in inference mode)
    assert result.shape == inp.shape


if __name__ == "__main__":
    for axis in [-1, 0, 1, 2]:
        test_softmax(axis=axis, sdfg_name=f"test_softmax_{axis}")

    test_softmax_2d(sdfg_name="test_softmax_2d")

    for axis in [-1, 0, 1]:
        test_log_softmax(axis=axis, sdfg_name=f"test_log_softmax_{axis}")

    test_log_softmax_2d(sdfg_name="test_log_softmax_2d")

    for normalized_shape in [[4], [3, 4], [2, 3, 4]]:
        test_layer_normalization(normalized_shape=normalized_shape,
                                 sdfg_name=f"test_layer_normalization_{'_'.join(map(str, normalized_shape))}")

    test_layer_normalization_2d(sdfg_name="test_layer_normalization_2d")

    test_dropout_inference(sdfg_name="test_dropout_inference")

    test_dropout_no_ratio(sdfg_name="test_dropout_no_ratio")
