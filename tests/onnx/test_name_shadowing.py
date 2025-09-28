import pytest

import dace

import dace.libraries.onnx as donnx


@pytest.mark.onnx
def test_shadowing(sdfg_name):
    new_shape = [8, 10]
    sdfg = dace.SDFG(sdfg_name)

    sdfg.add_array("X", [2, 4, 10], dace.float32)
    sdfg.add_array("shape", [len(new_shape)], dace.int64)
    sdfg.add_array("__return", new_shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_shape = state.add_access("shape")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReshape("reshape")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))
    state.add_edge(access_shape, None, op_node, "shape", sdfg.make_array_memlet("shape"))

    state.add_edge(op_node, "reshaped", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.compile()
