"""
Testing input and output combinations for onnx Ops

| Output \ Input | Scalar CPU | Scalar GPU | Array CPU | Array GPU |
|----------------+------------+------------+-----------+-----------|
| Scalar CPU     | Add        |            | Shape     |           |
| Scalar GPU     |            | Add        |           | Squeeze   |
| Array CPU      | Unsqueeze  |            | Add       | Shape     |
| Array GPU      |            | Unsqueeze  |           | Add       |

ALSO: test CPU fallback for all combinations of GPU ops
"""
import os

import numpy as np
import pytest

import dace
import dace.libraries.onnx as donnx


def parameterize_gpu(function):
    use_gpu = "ONNX_TEST_CUDA" in os.environ
    if use_gpu:
        return pytest.mark.parametrize("gpu", [True, False])(function)
    else:
        return pytest.mark.parametrize("gpu", [False])(function)


@parameterize_gpu
@pytest.mark.parametrize("apply_strict", [True, False])
def test_squeeze(gpu, apply_strict):
    sdfg = dace.SDFG("test_expansion")

    sdfg.add_array("X_arr", [1], dace.float32)
    sdfg.add_scalar("scalar", dace.float32, transient=True)
    sdfg.add_array("__return", [1], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X_arr")
    access_scalar = state.add_access("scalar")

    access_result = state.add_access("__return")

    op_node = donnx.ONNXSqueeze("Squeeze")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data",
                   sdfg.get_array_memlet("X_arr"))

    state.add_edge(op_node, "squeezed", access_scalar, None,
                   sdfg.get_array_memlet("scalar"))

    unsqueeze_op = donnx.ONNXUnsqueeze("Unsqueeze", axes=[0])
    state.add_node(unsqueeze_op)
    state.add_edge(access_scalar, None, unsqueeze_op, "data",
                   sdfg.get_array_memlet("scalar"))
    state.add_edge(unsqueeze_op, "expanded", access_result, None,
                   sdfg.get_array_memlet("__return"))

    X = np.random.rand(1).astype(np.float32)

    if gpu:
        sdfg.apply_gpu_transformations()

    if apply_strict:
        sdfg.expand_library_nodes()
        sdfg.apply_strict_transformations()

    result = sdfg(X_arr=X)

    assert result.shape == (1, )
    assert result[0] == X

@parameterize_gpu
@pytest.mark.parametrize("apply_strict", [True, False])
def test_shape(gpu, apply_strict):
    sdfg = dace.SDFG("test_expansion")

    sdfg.add_array("X_arr", [2, 4], dace.float32)
    sdfg.add_array("__return", [2], dace.int64)

    state = sdfg.add_state()
    access_X = state.add_access("X_arr")

    access_result = state.add_access("__return")

    op_node = donnx.ONNXShape("Shape")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data",
                   sdfg.get_array_memlet("X_arr"))

    state.add_edge(op_node, "shape", access_result, None,
                   sdfg.get_array_memlet("__return"))

    X = np.random.rand(2, 4).astype(np.float32)

    if gpu:
        sdfg.apply_gpu_transformations()

    if apply_strict:
        sdfg.expand_library_nodes()
        sdfg.apply_strict_transformations()

    result = sdfg(X_arr=X)

    assert np.all(result == (2, 4))


@parameterize_gpu
@pytest.mark.parametrize("apply_strict", [True, False])
def test_unsqueeze(gpu, apply_strict):
    sdfg = dace.SDFG("test_expansion")

    sdfg.add_scalar("X_arr", dace.float32)
    sdfg.add_array("__return", [1], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X_arr")

    access_result = state.add_access("__return")

    op_node = donnx.ONNXUnsqueeze("Unsqueeze", axes=[0])

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data",
                   sdfg.get_array_memlet("X_arr"))

    state.add_edge(op_node, "expanded", access_result, None,
                   sdfg.get_array_memlet("__return"))

    X = np.float32(np.random.rand())

    if gpu:
        sdfg.apply_gpu_transformations()

    if apply_strict:
        sdfg.expand_library_nodes()
        sdfg.apply_strict_transformations()

    sdfg.view()
    result = sdfg(X_arr=X)

    assert result.shape == (1, )
    assert X == result[0]


@parameterize_gpu
@pytest.mark.parametrize("scalars", [True, False])
@pytest.mark.parametrize("apply_strict", [True, False])
def test_add(scalars, gpu, apply_strict):
    sdfg = dace.SDFG("test_expansion")

    if scalars:
        sdfg.add_scalar("X_arr", dace.float32)
        sdfg.add_scalar("W_arr", dace.float32)
        sdfg.add_scalar("Z_arr", dace.float32, transient=True)
        sdfg.add_array("__return", [1], dace.float32)
    else:
        sdfg.add_array("X_arr", [2, 2], dace.float32)
        sdfg.add_array("W_arr", [2, 2], dace.float32)
        sdfg.add_array("__return", [2, 2], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X_arr")
    access_W = state.add_access("W_arr")

    if scalars:
        access_Z = state.add_access("Z_arr")

    access_result = state.add_access("__return")

    op_node = donnx.ONNXAdd("Add")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.get_array_memlet("X_arr"))
    state.add_edge(access_W, None, op_node, "B", sdfg.get_array_memlet("W_arr"))

    if scalars:
        state.add_edge(op_node, "C", access_Z, None,
                       sdfg.get_array_memlet("Z_arr"))
    else:
        state.add_edge(op_node, "C", access_result, None,
                       sdfg.get_array_memlet("__return"))

    if scalars:
        unsqueeze_op = donnx.ONNXUnsqueeze("Unsqueeze", axes=[0])
        state.add_node(unsqueeze_op)
        state.add_edge(access_Z, None, unsqueeze_op, "data",
                       sdfg.get_array_memlet("Z_arr"))
        state.add_edge(unsqueeze_op, "expanded", access_result, None,
                       sdfg.get_array_memlet("__return"))

    shapes = [] if scalars else [2, 2]
    X = np.random.rand(*shapes)
    W = np.random.rand(*shapes)
    if not scalars:
        X = X.astype(np.float32)
        W = W.astype(np.float32)

    if gpu:
        sdfg.apply_gpu_transformations()

    if apply_strict:
        sdfg.expand_library_nodes()
        sdfg.apply_strict_transformations()

    print(X)
    print(W)
    result = sdfg(X_arr=X, W_arr=W)

    numpy_result = X + W

    assert np.allclose(result, numpy_result)


if __name__ == '__main__':
    pytest.main([__file__])
