# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Gemm on OpenBLAS over operands BLAS cannot read as they are: no unit stride, or a single row with a free stride."""

import numpy as np

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


if __name__ == "__main__":
    test_an_operand_without_a_unit_stride_is_copied_dense()
    test_a_single_row_reaches_blas_with_a_valid_leading_dimension()
