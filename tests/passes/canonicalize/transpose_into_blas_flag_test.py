# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""After finalize, a matrix transpose whose only reader is a BLAS operand is that operand's transpose flag."""

import numpy as np

import dace
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

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
    def nonlocal_apply(beta: dace.float64[G, P], psi: dace.float64[G, S], dij: dace.float64[P, P], out: dace.float64[G, S]):
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


if __name__ == "__main__":
    test_a_transposed_gemm_operand_is_read_in_place()
    test_a_transpose_an_elementwise_map_reads_stays_a_copy()
