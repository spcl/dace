# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The cuBLAS Gemm wrapper sizes each operand from the array its edge reads after densification."""

import dace
from dace.libraries.blas.nodes.gemm import Gemm


# Reproducer: an operand with unit dimensions, ``a[0:4, 0:3, 1, 1]`` of a 4-D array, is densified to a 2-D copy; the
# wrapper paired the copy's 2-D box with the 4-D array's strides ("Strides must be the same size as shape").
def test_a_unit_dimension_operand_expands_for_the_gpu():
    sdfg = dace.SDFG("gemm_gpu_window_rank")
    sdfg.add_array("a", [4, 3, 2, 2], dace.float64)
    sdfg.add_array("b", [3, 3], dace.float64)
    sdfg.add_array("out", [4, 3], dace.float64)
    state = sdfg.add_state()
    gemm = Gemm("gemm")
    gemm.implementation = "cuBLAS"
    state.add_edge(state.add_read("a"), None, gemm, "_a", dace.Memlet("a[0:4, 0:3, 1, 1]"))
    state.add_edge(state.add_read("b"), None, gemm, "_b", dace.Memlet("b"))
    state.add_edge(gemm, "_c", state.add_write("out"), None, dace.Memlet("out"))

    sdfg.expand_library_nodes(recursive=False)  # expanding needs no device

    nested = next(node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.NestedSDFG)).sdfg
    assert nested.arrays["_a"].total_size == 12


if __name__ == "__main__":
    test_a_unit_dimension_operand_expands_for_the_gpu()
