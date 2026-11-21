# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes import Dot, Einsum, Gemm, MatMul
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.libraries.standard.nodes import CopyLibraryNode, FillLibraryNode, Reduce
from dace.sdfg import nodes
from dace.transformation.passes.single_element_library_nodes import SingleElementLibraryNodes


def wired(name: str, node: nodes.LibraryNode, inputs: dict[str, str], outputs: dict[str, str]) -> dace.SDFG:
    """``node`` reading and writing the given ``connector: memlet`` pairs over 2x2 arrays A, B (inputs) and C."""
    sdfg = dace.SDFG(name)
    for array in "ABC":
        sdfg.add_array(array, (2, 2), dace.float64)
    state = sdfg.add_state()
    state.add_node(node)
    for conn, memlet in inputs.items():
        node.add_in_connector(conn)
        state.add_edge(state.add_read(memlet[0]), None, node, conn, dace.Memlet(memlet))
    for conn, memlet in outputs.items():
        node.add_out_connector(conn)
        state.add_edge(node, conn, state.add_write(memlet[0]), None, dace.Memlet(memlet))
    return sdfg


ONE_A, ONE_B, ONE_C = "A[1, 0]", "B[0, 1]", "C[1, 1]"
CPY_IN, CPY_OUT = CopyLibraryNode.INPUT_CONNECTOR_NAME, CopyLibraryNode.OUTPUT_CONNECTOR_NAME
CASES = {
    "copy": (lambda: CopyLibraryNode("cp"), {CPY_IN: ONE_A}, {CPY_OUT: ONE_C}, lambda a, b, c: a[1, 0]),
    "fill": (
        lambda: FillLibraryNode("fl", value=2.5),
        {},
        {FillLibraryNode.OUTPUT_CONNECTOR_NAME: ONE_C},
        lambda a, b, c: 2.5,
    ),
    "gemm": (
        lambda: Gemm("gm", alpha=2.0, beta=3.0),
        {"_a": "A[1:2, 0:1]", "_b": "B[0:1, 1:2]", "_c": "C[1:2, 1:2]"},
        {"_c": "C[1:2, 1:2]"},
        lambda a, b, c: 2.0 * a[1, 0] * b[0, 1] + 3.0 * c[1, 1],
    ),
    "matmul": (
        lambda: MatMul("mm"),
        {"_a": "A[1:2, 0:1]", "_b": "B[0:1, 1:2]"},
        {"_c": "C[1:2, 1:2]"},
        lambda a, b, c: a[1, 0] * b[0, 1],
    ),
    "dot": (
        lambda: Dot("dt"),
        {"_x": "A[1, 0:1]", "_y": "B[0:1, 1]"},
        {"_result": ONE_C},
        lambda a, b, c: a[1, 0] * b[0, 1],
    ),
    "reduce": (
        lambda: Reduce("rd", wcr="lambda x, y: x + y", identity=1.0),
        {"_in": "A[1:2, 0:1]"},
        {"_out": ONE_C},
        lambda a, b, c: 1.0 + a[1, 0],
    ),
    "transpose": (
        lambda: TensorTranspose("tt", axes=[1, 0], alpha=2.0),
        {"_inp_tensor": "A[1:2, 0:1]"},
        {"_out_tensor": "C[1:2, 1:2]"},
        lambda a, b, c: 2.0 * a[1, 0],
    ),
}


def einsum_node() -> Einsum:
    node = Einsum("es")
    node.einsum_str = "ij,jk->ik"
    node.alpha = 0.5
    return node


CASES["einsum"] = (
    einsum_node,
    {"x": "A[1:2, 0:1]", "y": "B[0:1, 1:2]"},
    {"z": "C[1:2, 1:2]"},
    lambda a, b, c: 0.5 * a[1, 0] * b[0, 1],
)


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_single_element_library_node_is_its_tasklet(case: str):
    make, inputs, outputs, expected = CASES[case]
    sdfg = wired(f"single_element_{case}", make(), inputs, outputs)
    assert SingleElementLibraryNodes().apply_pass(sdfg, {}) == 1
    assert not any(isinstance(n, nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
    sdfg.validate()
    rng = np.random.default_rng(20261112)
    a, b, c = rng.random((2, 2)), rng.random((2, 2)), rng.random((2, 2))
    want = c.copy()
    want[1, 1] = expected(a, b, c)
    sdfg(A=a, B=b, C=c)
    assert np.allclose(c, want)


def test_a_two_element_copy_stays_a_library_node():
    sdfg = wired("single_element_wide_copy", CopyLibraryNode("cp"), {CPY_IN: "A[1, 0:2]"}, {CPY_OUT: "C[1, 0:2]"})
    assert SingleElementLibraryNodes().apply_pass(sdfg, {}) is None


def test_a_reduction_without_identity_accumulates_and_stays():
    reduce = Reduce("rd", wcr="lambda x, y: x + y", identity=None)
    sdfg = wired("single_element_accumulating_reduce", reduce, {"_in": "A[1:2, 0:1]"}, {"_out": ONE_C})
    assert SingleElementLibraryNodes().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    for name in sorted(CASES):
        test_a_single_element_library_node_is_its_tasklet(name)
    test_a_two_element_copy_stays_a_library_node()
    test_a_reduction_without_identity_accumulates_and_stays()
