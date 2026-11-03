# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""After finalize, a matrix transpose whose only reader is a BLAS operand is that operand's transpose flag."""

import numpy as np

import dace
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.transpose_into_blas_flag import transposes_as_blas_flags

G, P, S = (dace.symbol(s) for s in "GPS")


def transposes(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (Transpose, TensorTranspose))]


def finalized(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg)
    return finalize_for_target(sdfg, "cpu")


def test_a_transposed_gemm_operand_is_read_in_place():
    """kleinman_bylander_nonlocal's ``beta @ (dij @ (beta.T @ psi))``: no 116 MB copy of ``beta``, the GEMM transposes
    it by its flag."""

    @dace.program
    def nonlocal_apply(
        beta: dace.float64[G, P], psi: dace.float64[G, S], dij: dace.float64[P, P], out: dace.float64[G, S]
    ):
        overlap = beta.T @ psi
        out[:] = beta @ (dij @ overlap)

    sdfg = finalized(nonlocal_apply)
    assert not transposes(sdfg)
    rng = np.random.default_rng(6)
    beta, psi, dij = rng.random((40, 7)), rng.random((40, 5)), rng.random((7, 7))
    out = np.zeros((40, 5))
    sdfg(beta=beta, psi=psi, dij=dij, out=out, G=40, P=7, S=5)
    np.testing.assert_allclose(out, beta @ (dij @ (beta.T @ psi)))


def test_a_transpose_an_elementwise_map_reads_stays_a_copy():
    @dace.program
    def shifted(beta: dace.float64[G, P], out: dace.float64[P, G]):
        out[:] = np.transpose(beta) + 1.0

    sdfg = finalized(shifted)
    beta = np.random.default_rng(7).random((40, 7))
    out = np.zeros((7, 40))
    sdfg(beta=beta, out=out, G=40, P=7)
    np.testing.assert_allclose(out, beta.T + 1.0)


def test_a_gemm_on_a_window_of_a_transpose_keeps_the_transpose():
    """The operand is ``T[0:2, 0:2]``: flipping the flag would hand the GEMM all of ``A``."""
    sdfg = dace.SDFG("transpose_window_operand")
    sdfg.add_array("A", [3, 5], dace.float64)
    sdfg.add_array("B", [2, 4], dace.float64)
    sdfg.add_array("C", [2, 4], dace.float64)
    sdfg.add_array("T", [5, 3], dace.float64, transient=True)
    state = sdfg.add_state()
    temp = state.add_access("T")
    transpose = Transpose("transpose", dtype=dace.float64)
    state.add_edge(state.add_read("A"), None, transpose, "_inp", dace.Memlet("A[0:3, 0:5]"))
    state.add_edge(transpose, "_out", temp, None, dace.Memlet("T[0:5, 0:3]"))
    gemm = MatMul("gemm")
    state.add_edge(temp, None, gemm, "_a", dace.Memlet("T[0:2, 0:2]"))
    state.add_edge(state.add_read("B"), None, gemm, "_b", dace.Memlet("B[0:2, 0:4]"))
    state.add_edge(gemm, "_c", state.add_write("C"), None, dace.Memlet("C[0:2, 0:4]"))

    assert transposes_as_blas_flags(sdfg) == 0
    assert transposes(sdfg) == [transpose]


if __name__ == "__main__":
    test_a_gemm_on_a_window_of_a_transpose_keeps_the_transpose()
    test_a_transposed_gemm_operand_is_read_in_place()
    test_a_transpose_an_elementwise_map_reads_stays_a_copy()
