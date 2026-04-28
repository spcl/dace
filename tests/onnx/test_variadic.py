# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
import numpy as np

import dace
import dace.libraries.onnx as donnx


@pytest.mark.onnx
def test_sum():
    sdfg = dace.SDFG("test_sum")

    sdfg.add_array("A_arr", [2, 2], dace.float32)
    sdfg.add_array("B_arr", [2, 2], dace.float32)
    sdfg.add_array("C_arr", [2, 2], dace.float32)
    sdfg.add_array("__return", [2, 2], dace.float32)

    state = sdfg.add_state()
    access_A = state.add_access("A_arr")
    access_B = state.add_access("B_arr")
    access_C = state.add_access("C_arr")

    access_result = state.add_access("__return")

    op_node = donnx.ONNXSum("Sum")

    state.add_node(op_node)
    for i in range(3):
        op_node.add_in_connector("data_0__{}".format(i))
    state.add_edge(access_A, None, op_node, "data_0__0", sdfg.make_array_memlet("A_arr"))
    state.add_edge(access_B, None, op_node, "data_0__1", sdfg.make_array_memlet("B_arr"))
    state.add_edge(access_C, None, op_node, "data_0__2", sdfg.make_array_memlet("C_arr"))

    state.add_edge(op_node, "sum", access_result, None, sdfg.make_array_memlet("__return"))

    A = np.random.rand(2, 2).astype(np.float32)
    B = np.random.rand(2, 2).astype(np.float32)
    C = np.random.rand(2, 2).astype(np.float32)

    sdfg.validate()

    result = sdfg(A_arr=A, B_arr=B, C_arr=C)

    numpy_result = A + B + C

    assert np.allclose(result,
                       numpy_result), f"Variadic sum mismatch: max diff = {np.max(np.abs(result - numpy_result))}"


if __name__ == "__main__":
    test_sum()
