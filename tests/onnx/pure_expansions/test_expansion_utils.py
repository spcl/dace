import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
import numpy as np

import dace
import dace.libraries.onnx as donnx


@pytest.mark.onnx
def test_sqrt_expansion(sdfg_name):
    # sqrt expansion makes use of the program_for_node function
    sdfg = dace.SDFG(sdfg_name)

    sdfg.add_array("inp", [2, 4], dace.float32)
    sdfg.add_array("__return", [2, 4], dace.float32)

    state = sdfg.add_state()
    access_in = state.add_access("inp")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXSqrt("sqrt")

    state.add_node(op_node)
    state.add_edge(access_in, None, op_node, "X", sdfg.make_array_memlet("inp"))

    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    X = np.random.rand(2, 4).astype(np.float32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion wouldn't produce a map
    assert any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(inp=X)

    assert np.allclose(np.sqrt(X), result)


if __name__ == "__main__":
    test_sqrt_expansion(sdfg_name="test_sqrt_expansion")
