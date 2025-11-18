# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import dace
import dace.libraries.onnx as donnx
from onnx import helper, TensorProto
from dace.libraries.onnx import ONNXModel


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.onnx
@pytest.mark.parametrize(
    "axis, exclusive, reverse",
    [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        pytest.param(
            0,
            1,
            0,
        ),  # exclusive
        pytest.param(0, 0, 1),  # reverse
        pytest.param(1, 1, 0),  # exclusive on axis 1
        pytest.param(1, 0, 1),  # reverse on axis 1
    ])
def test_cumsum(axis, exclusive, reverse, sdfg_name):
    """Test CumSum operation with different parameters."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(3, 4, 5).astype(np.float32)
    axis_arr = np.array([axis], dtype=np.int64)

    # Compute expected result
    if reverse:
        inp_for_cumsum = np.flip(inp, axis=axis)
    else:
        inp_for_cumsum = inp

    if exclusive:
        # Exclusive cumsum: shift result and set first element to 0
        expected = np.cumsum(inp_for_cumsum, axis=axis)
        expected = np.roll(expected, 1, axis=axis)
        # Set the first slice along the axis to 0
        slices = [slice(None)] * expected.ndim
        slices[axis] = 0
        expected[tuple(slices)] = 0
    else:
        expected = np.cumsum(inp_for_cumsum, axis=axis)

    if reverse:
        expected = np.flip(expected, axis=axis)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("axis", axis_arr.shape, dace.int64)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXCumSum("cumsum", exclusive=exclusive, reverse=reverse)
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "x", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("axis"), None, op_node, "axis", sdfg.make_array_memlet("axis"))
    state.add_edge(op_node, "y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, axis=axis_arr)

    assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.onnx
def test_cumsum_1d(sdfg_name):
    """Test CumSum operation on 1D array."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    axis_arr = np.array([0], dtype=np.int64)
    expected = np.cumsum(inp)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("axis", axis_arr.shape, dace.int64)
    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXCumSum("cumsum", exclusive=0, reverse=0)
    state.add_node(op_node)

    state.add_edge(state.add_read("inp"), None, op_node, "x", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("axis"), None, op_node, "axis", sdfg.make_array_memlet("axis"))
    state.add_edge(op_node, "y", state.add_write("__return"), None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, axis=axis_arr)

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize("keepdims, axes", [
    (True, [0]),
    (False, [-1]),
    (True, [0, -1]),
    (False, [1]),
    (True, [1, 2]),
])
def test_reduce_min(keepdims, axes, sdfg_name):
    """Test ReduceMin operation with different axes and keepdims."""

    X = np.random.randn(3, 4, 5).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

    sdfg.add_array("X", [3, 4, 5], dace.float32)

    expected = np.min(X, axis=tuple(axes), keepdims=keepdims)

    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReduceMin("reduce_min")
    op_node.axes = axes
    op_node.keepdims = 1 if keepdims else 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))
    state.add_edge(op_node, "reduced", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(X=X)

    assert_allclose(expected, result, rtol=1e-5, atol=1e-5)


@pytest.mark.onnx
def test_reduce_min_all_axes(sdfg_name):
    """Test ReduceMin operation reducing all axes to scalar."""

    X = np.random.randn(3, 4, 5).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

    expected = np.min(X)

    sdfg.add_array("X", [3, 4, 5], dace.float32)
    sdfg.add_scalar("Y", dace.float32, transient=True)
    sdfg.add_array("__return", [1], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Y = state.add_access("Y")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReduceMin("reduce_min")
    op_node.keepdims = 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))
    state.add_edge(op_node, "reduced", access_Y, None, sdfg.make_array_memlet("Y"))
    state.add_edge(access_Y, None, access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(X=X)

    assert_allclose(expected, result)


@pytest.mark.onnx
@pytest.mark.parametrize("keepdims, axes", [
    (True, [1]),
    (False, [2]),
    (True, [0, 2]),
])
def test_reduce_sum_additional(keepdims, axes, sdfg_name):
    """Test ReduceSum with additional parameter combinations."""

    X = np.random.randn(4, 5, 6).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

    sdfg.add_array("X", [4, 5, 6], dace.float32)

    expected = np.sum(X, axis=tuple(axes), keepdims=keepdims)

    sdfg.add_array("__return", expected.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReduceSum("reduce_sum")
    op_node.axes = axes
    op_node.keepdims = 1 if keepdims else 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))
    state.add_edge(op_node, "reduced", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(X=X)

    assert_allclose(expected, result, rtol=1e-5, atol=1e-5)


@pytest.mark.onnx
#+yapf: disable
@pytest.mark.parametrize("reduce_type, keepdims, axes",
                         [('Sum',  True,  [0]),
                          ('Sum',  False, [-1]),
                          ('Sum',  True,  [0, -1]),
                          ('Max',  False, [0, -1]),
                          ('Max',  True,  [0]),
                          ('Max',  True,  [-1]),
                          ('Mean', True,  [-1]),
                          ('Mean', True,  [0, -1]),
                          ('Mean', False, [0])])
#+yapf: enable
def test_reduce(keepdims, reduce_type, axes, sdfg_name):

    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

    sdfg.add_array("X", [2, 4, 10], dace.float32)

    numpy_func = getattr(np, reduce_type.lower())
    numpy_result = numpy_func(X.copy(), axis=tuple(axes), keepdims=keepdims)

    resulting_shape = numpy_result.shape

    sdfg.add_array("__return", resulting_shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = getattr(donnx, "ONNXReduce" + reduce_type)("reduce")
    op_node.axes = axes
    op_node.keepdims = 1 if keepdims else 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "reduced", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(X=X)

    assert_allclose(numpy_result, result, rtol=1e-5, atol=1e-5)


@pytest.mark.onnx
def test_reduce_scalar(sdfg_name):
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

    numpy_result = np.mean(X)

    sdfg.add_array("X", [2, 4, 10], dace.float32)
    sdfg.add_scalar("Y", dace.float32, transient=True)
    sdfg.add_array("__return", [1], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Y = state.add_access("Y")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReduceMean("mean")
    op_node.keepdims = 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "reduced", access_Y, None, sdfg.make_array_memlet("Y"))

    state.add_edge(access_Y, None, access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = sdfg(X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
def test_reduce_l2():
    """Test ReduceL2 operator with DaCe ONNX frontend."""

    # Create a simple ReduceL2 model
    # In opset 13, axes is an attribute, not an input
    input_tensor = helper.make_tensor_value_info('data', TensorProto.FLOAT, [4, 5, 6])
    output_tensor = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [5, 6])

    node = helper.make_node('ReduceL2', inputs=['data'], outputs=['reduced'], axes=[0], keepdims=0)

    graph = helper.make_graph([node], 'reduce_l2_test', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # Use IR version 8 for compatibility

    dace_model = ONNXModel("reduce_l2", model)

    X = np.random.randn(4, 5, 6).astype(np.float32)
    result = dace_model(data=X)
    expected = np.sqrt(np.sum(np.square(X), axis=0))

    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5, err_msg="ReduceL2 output mismatch")


if __name__ == "__main__":
    cumsum_params = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    ]
    for axis, exclusive, reverse in cumsum_params:
        test_cumsum(axis=axis,
                    exclusive=exclusive,
                    reverse=reverse,
                    sdfg_name=f"test_cumsum_{axis}_{exclusive}_{reverse}")

    test_cumsum_1d(sdfg_name="test_cumsum_1d")

    # ReduceMin tests
    reduce_min_params = [
        (True, [0]),
        (False, [-1]),
        (True, [0, -1]),
        (False, [1]),
        (True, [1, 2]),
    ]
    for keepdims, axes in reduce_min_params:
        test_reduce_min(keepdims=keepdims, axes=axes, sdfg_name=f"test_reduce_min_{keepdims}_{axes}")

    test_reduce_min_all_axes(sdfg_name="test_reduce_min_all_axes")

    reduce_sum_params = [
        (True, [1]),
        (False, [2]),
        (True, [0, 2]),
    ]
    for keepdims, axes in reduce_sum_params:
        test_reduce_sum_additional(keepdims=keepdims,
                                   axes=axes,
                                   sdfg_name=f"test_reduce_sum_additional_{keepdims}_{axes}")
