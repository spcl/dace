"""
Testing input and output combinations for onnx Ops

| Output \ Input | Scalar CPU | Scalar GPU | Array CPU | Array GPU |
|----------------+------------+------------+-----------+-----------|
| Scalar CPU     | Add        |            | Shape     |           |
| Scalar GPU     |            | Add        |           | Squeeze   |
| Array CPU      | Unsqueeze  |            | Add       | Shape     |
| Array GPU      |            | Squeeze    |           | Add       |
"""
import os

import numpy as np

import dace
import dace.libraries.onnx as donnx

def test_add(scalars=False, gpu=False):
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
        state.add_edge(op_node, "C", access_Z,  None, sdfg.get_array_memlet("Z_arr"))
    else:
        state.add_edge(op_node, "C", access_result, None, sdfg.get_array_memlet("__return"))

    if scalars:
        unsqueeze_op = donnx.ONNXUnsqueeze("Unsqueeze", axes=[0])
        state.add_node(unsqueeze_op)
        state.add_edge(access_Z, None, unsqueeze_op, "data", sdfg.get_array_memlet("Z_arr"))
        state.add_edge(unsqueeze_op, "expanded", access_result, None, sdfg.get_array_memlet("__return"))

    shapes = [] if scalars else [2, 2]
    X = np.random.rand(*shapes)
    W = np.random.rand(*shapes)
    if not scalars:
        X = X.astype(np.float32)
        W = W.astype(np.float32)


    if gpu:
        sdfg.apply_gpu_transformations()

    print(X)
    print(W)
    result = sdfg(X_arr=X, W_arr=W)

    numpy_result = X + W

    assert np.allclose(result, numpy_result)


if __name__ == '__main__':
    use_gpu = "ONNX_TEST_CUDA" in os.environ

    for scalars in [True, False]:
        for gpu in [True, False] if use_gpu else [False]:
            test_add(scalars=scalars, gpu=gpu)


