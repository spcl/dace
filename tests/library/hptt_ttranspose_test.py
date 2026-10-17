# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The HPTT expansion of TensorTranspose calls the entry point of its element type with that type's signature."""

import pytest

import dace
from dace.libraries.linalg import TensorTranspose
from dace.libraries.linalg.nodes.ttranspose import ExpandHPTT


@pytest.mark.parametrize(
    "dtype, call",
    [
        (dace.float64, "dTensorTranspose(perm, 2, 1, _inp_tensor"),
        (dace.complex128, "zTensorTranspose(perm, 2, 1, false, _inp_tensor"),
    ],
)
def test_an_hptt_call_matches_the_signature_of_its_type(dtype, call):
    """HPTT's complex entry points take ``conjA`` right after ``alpha``; without it the input pointer lands on the
    ``conjA`` bool and the call does not compile."""
    sdfg = dace.SDFG(f"hptt_call_{dtype.to_string()}")
    sdfg.add_array("x", (4, 8), dtype)
    sdfg.add_array("y", (8, 4), dtype)
    state = sdfg.add_state(is_start_block=True)
    node = TensorTranspose("_TensorTranspose", axes=[1, 0])
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_inp_tensor", dace.Memlet("x[0:4, 0:8]"))
    state.add_edge(node, "_out_tensor", state.add_write("y"), None, dace.Memlet("y[0:8, 0:4]"))
    assert call in ExpandHPTT.expansion(node, state, sdfg).code.as_string


def test_hptt_inside_an_sdfg_that_names_its_output_out_tensor_validates():
    """TensorDot's TTGT expansion puts a TensorTranspose into an SDFG holding a container ``_out_tensor``; the HPTT
    expansion renames its pointer connectors so they cannot shadow that container (cp2k_grid_integrate)."""
    sdfg = dace.SDFG("hptt_out_tensor_name")
    sdfg.add_array("_inp_tensor", (4, 8), dace.float64)
    sdfg.add_array("_out_tensor", (8, 4), dace.float64)
    state = sdfg.add_state(is_start_block=True)
    node = TensorTranspose("_TensorTranspose", axes=[1, 0])
    node.implementation = "HPTT"
    state.add_node(node)
    state.add_edge(state.add_read("_inp_tensor"), None, node, "_inp_tensor", dace.Memlet("_inp_tensor[0:4, 0:8]"))
    state.add_edge(node, "_out_tensor", state.add_write("_out_tensor"), None, dace.Memlet("_out_tensor[0:8, 0:4]"))
    sdfg.expand_library_nodes()
    sdfg.validate()


if __name__ == "__main__":
    test_an_hptt_call_matches_the_signature_of_its_type(dace.float64, "dTensorTranspose(perm, 2, 1, _inp_tensor")
    test_an_hptt_call_matches_the_signature_of_its_type(
        dace.complex128, "zTensorTranspose(perm, 2, 1, false, _inp_tensor"
    )
    test_hptt_inside_an_sdfg_that_names_its_output_out_tensor_validates()
