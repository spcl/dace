# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Gemm on OpenBLAS over operands BLAS cannot read as they are: no unit stride, or a single row with a free stride."""

import numpy as np
import pytest

import dace
from dace.libraries.blas import Gemm
from dace.libraries.blas.nodes.matmul import densify_matrix_operands

N = 4


def gemm_sdfg(name: str, a_shape: list[int], a_subset: str, b_shape: list[int], out_shape: list[int]) -> dace.SDFG:
    """``out = a[a_subset] @ b`` through one OpenBLAS Gemm."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", a_shape, dace.float64)
    sdfg.add_array("b", b_shape, dace.float64)
    sdfg.add_array("out", out_shape, dace.float64)
    state = sdfg.add_state()
    gemm = Gemm("gemm")
    gemm.implementation = "OpenBLAS"
    state.add_edge(state.add_read("a"), None, gemm, "_a", dace.Memlet(f"a[{a_subset}]"))
    state.add_edge(state.add_read("b"), None, gemm, "_b", dace.Memlet.from_array("b", sdfg.arrays["b"]))
    state.add_edge(gemm, "_c", state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    return sdfg


def test_an_operand_without_a_unit_stride_is_copied_dense():
    sdfg = gemm_sdfg("gemm_strided", [N, N, 3], "0:4, 0:4, 1", [N, N], [N, N])
    state = sdfg.start_state
    gemm = next(node for node in state.nodes() if isinstance(node, Gemm))
    densify_matrix_operands(gemm, state, sdfg)
    assert any(name.endswith("_dense") for name in sdfg.arrays)

    rng = np.random.default_rng(0)
    a, b, out = rng.random((N, N, 3)), rng.random((N, N)), np.zeros((N, N))
    sdfg(a=a, b=b, out=out)
    assert np.allclose(out, a[:, :, 1] @ b)


def test_a_single_row_reaches_blas_with_a_valid_leading_dimension():
    """The row's stride is free, but BLAS checks the leading dimension against the column count and computes nothing."""
    sdfg = gemm_sdfg("gemm_row", [1, N], "0:1, 0:4", [N, N], [1, N])
    rng = np.random.default_rng(1)
    a, b, out = rng.random((1, N)), rng.random((N, N)), np.zeros((1, N))
    sdfg(a=a, b=b, out=out)
    assert np.allclose(out, a @ b)


def test_an_output_without_a_unit_stride_expands_to_pure():
    """``out(a, :, :) = matmul(g, h)``: BLAS writes along a unit stride, and this slice of a Fortran array has none."""
    sdfg = dace.SDFG("gemm_strided_output")
    sdfg.add_array("a", [N, 3], dace.float64)
    sdfg.add_array("b", [3, 5], dace.float64)
    sdfg.add_array("out", [2, N, 5], dace.float64, strides=[1, 2, 2 * N])
    state = sdfg.add_state()
    gemm = Gemm("gemm")
    gemm.implementation = "OpenBLAS"
    state.add_edge(state.add_read("a"), None, gemm, "_a", dace.Memlet("a"))
    state.add_edge(state.add_read("b"), None, gemm, "_b", dace.Memlet("b"))
    state.add_edge(gemm, "_c", state.add_write("out"), None, dace.Memlet("out[1, 0:4, 0:5]"))

    rng = np.random.default_rng(0)
    a, b, out = rng.random((N, 3)), rng.random((3, 5)), np.zeros((2, N, 5), order="F")
    sdfg(a=a, b=b, out=out)

    assert np.allclose(out[1], a @ b)
    assert np.all(out[0] == 0)


def test_a_one_element_operand_reaches_blas_as_a_pointer():
    """A one-element memlet binds a value, and ``cblas_dgemm`` takes a pointer."""
    sdfg = dace.SDFG("gemm_one_element")
    sdfg.add_array("a", [N, 1], dace.float64)
    sdfg.add_array("b", [N, N], dace.float64)
    sdfg.add_array("out", [N, 1], dace.float64)
    state = sdfg.add_state()
    gemm = Gemm("gemm")
    gemm.implementation = "OpenBLAS"
    state.add_edge(state.add_read("a"), None, gemm, "_a", dace.Memlet("a[0:4, 0:1]"))
    state.add_edge(state.add_read("b"), None, gemm, "_b", dace.Memlet("b[0:1, 0:1]"))
    state.add_edge(gemm, "_c", state.add_write("out"), None, dace.Memlet("out[0:4, 0:1]"))

    rng = np.random.default_rng(0)
    a, b, out = rng.random((N, 1)), rng.random((N, N)), np.zeros((N, 1))
    sdfg(a=a, b=b, out=out)

    assert np.allclose(out, a @ b[:1, :1])


@pytest.mark.parametrize(
    "a_shape,a_subset,b_shape,out_shape",
    [([4, 3], "2:4, 0:3", [3, 3], [2, 3]), ([4, 3, 2, 2], "0:4, 0:3, 1, 1", [3, 3], [4, 3])],
)
def test_a_gpu_copy_moves_each_operand_s_window(a_shape, a_subset, b_shape, out_shape):
    """The host-to-GPU copy starts at the connector's pointer, so a whole container read past the operand's end.

    Expanding needs no device.
    """
    sdfg = dace.SDFG(f"gemm_gpu_window_{len(a_shape)}")
    sdfg.add_array("a", a_shape, dace.float64)
    sdfg.add_array("b", b_shape, dace.float64)
    sdfg.add_array("out", out_shape, dace.float64)
    state = sdfg.add_state()
    gemm = Gemm("gemm")
    gemm.implementation = "cuBLAS"
    state.add_edge(state.add_read("a"), None, gemm, "_a", dace.Memlet(f"a[{a_subset}]"))
    state.add_edge(state.add_read("b"), None, gemm, "_b", dace.Memlet("b"))
    state.add_edge(gemm, "_c", state.add_write("out"), None, dace.Memlet("out"))

    sdfg.expand_library_nodes(recursive=False)

    nested = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.NestedSDFG)).sdfg
    expected = int(np.prod(out_shape))
    assert nested.arrays["_a"].total_size == expected
    copies = [e for s in nested.states() for e in s.edges() if e.src.label == "_a" and e.dst.label == "_a_gpu"]
    assert [copy.data.num_elements() for copy in copies] == [expected]


if __name__ == "__main__":
    test_an_operand_without_a_unit_stride_is_copied_dense()
    test_a_single_row_reaches_blas_with_a_valid_leading_dimension()
    test_an_output_without_a_unit_stride_expands_to_pure()
    test_a_one_element_operand_reaches_blas_as_a_pointer()
    test_a_gpu_copy_moves_each_operand_s_window([4, 3], "2:4, 0:3", [3, 3], [2, 3])
    test_a_gpu_copy_moves_each_operand_s_window([4, 3, 2, 2], "0:4, 0:3, 1, 1", [3, 3], [4, 3])
