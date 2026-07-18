# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pad legality via zero-fill (PadZeroFill).

Pad grows a dimension but library nodes (TensorDot/Gemm/Reduce) read the descriptor *shape*, so a padded
dim would silently pull the pad region into the op. PadZeroFill zeros the pad cells once at entry: a
sum-of-products contraction over a padded contracted dim then adds 0 (legal); a sum reduction over a padded
dim adds 0 (legal); a NON-sum reduction (max/min/product) over a padded dim is refused (0 is not its identity).
"""
import numpy as np
import pytest
import dace
from dace.memlet import Memlet
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.layout.pad_dimensions import PadZeroFill


def _tensordot_sdfg(M, Kpad, N):
    """C[M,N] = A[M,Kpad] . B[Kpad,N] (contraction over the padded K axis), pure lowering."""
    sdfg = dace.SDFG("pad_td")
    sdfg.add_array("A", [M, Kpad], dace.float64)
    sdfg.add_array("B", [Kpad, N], dace.float64)
    sdfg.add_array("C", [M, N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    td = TensorDot("dot", left_axes=[1], right_axes=[0])
    td.implementation = "pure"
    st.add_node(td)
    st.add_edge(st.add_read("A"), None, td, "_left_tensor", Memlet.from_array("A", sdfg.arrays["A"]))
    st.add_edge(st.add_read("B"), None, td, "_right_tensor", Memlet.from_array("B", sdfg.arrays["B"]))
    st.add_edge(td, "_out_tensor", st.add_write("C"), None, Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def test_pad_contracted_dim_zero_fill_is_legal():
    """Padding the contracted K axis of a TensorDot and zero-filling it leaves the live result unchanged."""
    M, K, p, N = 4, 5, 3, 6
    Kpad = K + p
    sdfg = _tensordot_sdfg(M, Kpad, N)
    # A padded on dim 1 (K), B padded on dim 0 (K); zero those pad slices.
    assert PadZeroFill(pad_map={"A": [0, p], "B": [p, 0]}).apply_pass(sdfg, {}) == 0
    sdfg.validate()

    A = np.random.rand(M, Kpad)  # random pad columns -> zeroed by the pass
    B = np.random.rand(Kpad, N)  # random pad rows    -> zeroed by the pass
    C = np.zeros((M, N))
    sdfg(A=A.copy(), B=B.copy(), C=C)
    assert np.allclose(C, A[:, :K] @ B[:K, :])  # == the live (unpadded) contraction


def test_pad_free_output_dim_zero_fill():
    """Padding a free (M) dim: with the pad rows of A zeroed, the pad output rows of C are 0; the live rows match."""
    M, K, p, N = 4, 5, 2, 6
    Mpad = M + p
    sdfg = _tensordot_sdfg(Mpad, K, N)
    assert PadZeroFill(pad_map={"A": [p, 0], "C": [p, 0]}).apply_pass(sdfg, {}) == 0
    sdfg.validate()

    A = np.random.rand(Mpad, K)
    B = np.random.rand(K, N)
    C = np.zeros((Mpad, N))
    sdfg(A=A.copy(), B=B.copy(), C=C)
    assert np.allclose(C[:M], A[:M] @ B)
    assert np.allclose(C[M:], 0.0)  # pad rows produced from zeroed A pad rows


def _reduce_sdfg(name, wcr, identity):
    """Y[M] = reduce over axis 1 of X[M, Kpad]."""
    M, Kpad = 4, 8
    sdfg = dace.SDFG(name)
    sdfg.add_array("X", [M, Kpad], dace.float64)
    sdfg.add_array("Y", [M], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    red = Reduce("reduce", wcr=wcr, axes=[1], identity=identity)
    red.add_in_connector("_in")
    red.add_out_connector("_out")
    st.add_node(red)
    st.add_edge(st.add_read("X"), None, red, "_in", Memlet.from_array("X", sdfg.arrays["X"]))
    st.add_edge(red, "_out", st.add_write("Y"), None, Memlet.from_array("Y", sdfg.arrays["Y"]))
    return sdfg, M, Kpad


def test_pad_sum_reduction_over_pad_is_legal():
    """A sum reduction over a padded axis is legal after zero-fill (pad cells add 0)."""
    sdfg, M, Kpad = _reduce_sdfg("pad_sum", "lambda a, b: a + b", 0.0)
    K = Kpad - 2
    assert PadZeroFill(pad_map={"X": [0, 2]}).apply_pass(sdfg, {}) == 0
    sdfg.validate()

    X = np.random.rand(M, Kpad)
    Y = np.zeros(M)
    sdfg(X=X.copy(), Y=Y)
    assert np.allclose(Y, X[:, :K].sum(axis=1))  # pad columns contributed 0


def test_pad_max_reduction_over_pad_raises():
    """A max reduction over a padded axis is refused (0 is not the identity of max)."""
    sdfg, _, _ = _reduce_sdfg("pad_max", "lambda a, b: max(a, b)", -1e38)
    with pytest.raises(NotImplementedError):
        PadZeroFill(pad_map={"X": [0, 2]}).apply_pass(sdfg, {})


def test_pad_max_reduction_over_free_dim_ok():
    """A max reduction is fine if the PADDED dim is not the reduced one (here dim 0 padded, axis 1 reduced)."""
    M, Kpad, p = 4, 8, 2
    sdfg = dace.SDFG("pad_max_free")
    sdfg.add_array("X", [M + p, Kpad], dace.float64)
    sdfg.add_array("Y", [M + p], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    red = Reduce("reduce", wcr="lambda a, b: max(a, b)", axes=[1], identity=-1e38)
    red.add_in_connector("_in")
    red.add_out_connector("_out")
    st.add_node(red)
    st.add_edge(st.add_read("X"), None, red, "_in", Memlet.from_array("X", sdfg.arrays["X"]))
    st.add_edge(red, "_out", st.add_write("Y"), None, Memlet.from_array("Y", sdfg.arrays["Y"]))
    # dim 0 of X is padded but the reduction is over axis 1 -> not refused.
    assert PadZeroFill(pad_map={"X": [p, 0], "Y": [p]}).apply_pass(sdfg, {}) == 0


if __name__ == "__main__":
    test_pad_contracted_dim_zero_fill_is_legal()
    test_pad_free_output_dim_zero_fill()
    test_pad_sum_reduction_over_pad_is_legal()
    test_pad_max_reduction_over_pad_raises()
    test_pad_max_reduction_over_free_dim_ok()
    print("pad_zero_fill tests PASS")
