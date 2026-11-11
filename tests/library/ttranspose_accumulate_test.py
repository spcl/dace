# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``TensorTranspose`` that folds onto its output (``beta != 0``) reads and writes the output element."""

import numpy as np

import dace
from dace.libraries.linalg import TensorTranspose


def accumulating_transpose() -> dace.SDFG:
    sdfg = dace.SDFG("ttranspose_accumulate")
    sdfg.add_array("a", [3, 4, 5], dace.float64)
    sdfg.add_array("out", [5, 3, 4], dace.float64)
    state = sdfg.add_state()
    node = TensorTranspose("transpose", axes=[2, 0, 1], alpha=2, beta=1)
    node.implementation = "pure"
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_inp_tensor", dace.Memlet("a[0:3, 0:4, 0:5]"))
    state.add_edge(node, "_out_tensor", state.add_write("out"), None, dace.Memlet("out[0:5, 0:3, 0:4]"))
    return sdfg


def test_the_pure_expansion_of_an_accumulating_transpose_is_valid_and_computes_the_fold():
    """The read and the write of the output element are two edges with a Memlet each: one object on both is an
    invalid SDFG (cp2k_density_matrix_trs4's device form was refused with "Duplicate memlet detected")."""
    sdfg = accumulating_transpose()
    sdfg.expand_library_nodes()
    sdfg.validate()
    rng = np.random.default_rng(1)
    a, out = rng.random((3, 4, 5)), rng.random((5, 3, 4))
    expected = 2 * np.transpose(a, (2, 0, 1)) + out
    sdfg(a=a, out=out)
    assert np.allclose(out, expected)


if __name__ == "__main__":
    test_the_pure_expansion_of_an_accumulating_transpose_is_valid_and_computes_the_fold()
