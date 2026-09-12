# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pad legality via zero-fill (PadZeroFill).

Pad grows a dimension but library nodes (TensorDot/Gemm/Reduce) read the descriptor *shape*, so a padded
dim would silently pull the pad region into the op. PadZeroFill zeros the pad cells once at entry: a
sum-of-products contraction over a padded contracted dim then adds 0 (legal); a sum reduction over a padded
dim adds 0 (legal); a NON-sum reduction (max/min/product) over a padded dim is refused (0 is not its identity).

PadZeroFill requires PadDimensions' result (the pre-pad shapes) so it zeroes only dead cells; run without it,
or against a shape that was never grown, it is a hard error rather than a silent wipe of live data.
"""
import numpy as np
import pytest
import dace
from dace.memlet import Memlet
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.layout.pad_dimensions import PadDimensions, PadZeroFill


def _pad_then_zero(sdfg, pad_map):
    """Run the two companion passes as intended: grow, then zero-fill using the recorded pre-pad shapes."""
    originals = PadDimensions(pad_map=pad_map).apply_pass(sdfg, {})
    return PadZeroFill(pad_map=pad_map).apply_pass(sdfg, {"PadDimensions": originals})


def _tensordot_sdfg(M, K, N):
    """C[M,N] = A[M,K] . B[K,N] (contraction over K), pure lowering. Built at the *live* shape."""
    sdfg = dace.SDFG("pad_td")
    sdfg.add_array("A", [M, K], dace.float64)
    sdfg.add_array("B", [K, N], dace.float64)
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
    sdfg = _tensordot_sdfg(M, K, N)
    # A padded on dim 1 (K), B padded on dim 0 (K); grow, then zero those pad slices.
    assert _pad_then_zero(sdfg, {"A": [0, p], "B": [p, 0]}) == 0
    sdfg.validate()

    Kpad = K + p
    A = np.random.rand(M, Kpad)  # random pad columns -> zeroed by the pass
    B = np.random.rand(Kpad, N)  # random pad rows    -> zeroed by the pass
    C = np.zeros((M, N))
    sdfg(A=A.copy(), B=B.copy(), C=C)
    assert np.allclose(C, A[:, :K] @ B[:K, :])  # == the live (unpadded) contraction


def test_pad_free_output_dim_zero_fill():
    """Padding a free (M) dim: with the pad rows of A zeroed, the pad output rows of C are 0; the live rows match."""
    M, K, p, N = 4, 5, 2, 6
    sdfg = _tensordot_sdfg(M, K, N)
    assert _pad_then_zero(sdfg, {"A": [p, 0], "C": [p, 0]}) == 0
    sdfg.validate()

    Mpad = M + p
    A = np.random.rand(Mpad, K)
    B = np.random.rand(K, N)
    C = np.zeros((Mpad, N))
    sdfg(A=A.copy(), B=B.copy(), C=C)
    assert np.allclose(C[:M], A[:M] @ B)
    assert np.allclose(C[M:], 0.0)  # pad rows produced from zeroed A pad rows


def _reduce_sdfg(name, wcr, identity, M=4, K=6):
    """Y[M] = reduce over axis 1 of X[M, K], built at the live shape."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("X", [M, K], dace.float64)
    sdfg.add_array("Y", [M], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    red = Reduce("reduce", wcr=wcr, axes=[1], identity=identity)
    red.add_in_connector("_in")
    red.add_out_connector("_out")
    st.add_node(red)
    st.add_edge(st.add_read("X"), None, red, "_in", Memlet.from_array("X", sdfg.arrays["X"]))
    st.add_edge(red, "_out", st.add_write("Y"), None, Memlet.from_array("Y", sdfg.arrays["Y"]))
    return sdfg, M, K


def test_pad_sum_reduction_over_pad_is_legal():
    """A sum reduction over a padded axis is legal after zero-fill (pad cells add 0)."""
    sdfg, M, K = _reduce_sdfg("pad_sum", "lambda a, b: a + b", 0.0)
    assert _pad_then_zero(sdfg, {"X": [0, 2]}) == 0
    sdfg.validate()

    Kpad = K + 2
    X = np.random.rand(M, Kpad)
    Y = np.zeros(M)
    sdfg(X=X.copy(), Y=Y)
    assert np.allclose(Y, X[:, :K].sum(axis=1))  # pad columns contributed 0


def test_pad_max_reduction_over_pad_raises():
    """A max reduction over a padded axis is refused (0 is not the identity of max)."""
    sdfg, _, _ = _reduce_sdfg("pad_max", "lambda a, b: max(a, b)", -1e38)
    pm = {"X": [0, 2]}
    originals = PadDimensions(pad_map=pm).apply_pass(sdfg, {})
    with pytest.raises(NotImplementedError):
        PadZeroFill(pad_map=pm).apply_pass(sdfg, {"PadDimensions": originals})


def test_pad_max_reduction_over_free_dim_ok():
    """A max reduction is fine if the PADDED dim is not the reduced one (here dim 0 padded, axis 1 reduced)."""
    M, K, p = 4, 6, 2
    sdfg = dace.SDFG("pad_max_free")
    sdfg.add_array("X", [M, K], dace.float64)
    sdfg.add_array("Y", [M], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    red = Reduce("reduce", wcr="lambda a, b: max(a, b)", axes=[1], identity=-1e38)
    red.add_in_connector("_in")
    red.add_out_connector("_out")
    st.add_node(red)
    st.add_edge(st.add_read("X"), None, red, "_in", Memlet.from_array("X", sdfg.arrays["X"]))
    st.add_edge(red, "_out", st.add_write("Y"), None, Memlet.from_array("Y", sdfg.arrays["Y"]))
    # dim 0 of X is padded but the reduction is over axis 1 -> not refused.
    assert _pad_then_zero(sdfg, {"X": [p, 0], "Y": [p]}) == 0


def test_zero_fill_without_pad_dimensions_raises():
    """PadZeroFill without PadDimensions' record refuses to run (it would zero live tail cells)."""
    sdfg = _tensordot_sdfg(4, 5, 6)
    with pytest.raises(ValueError):
        PadZeroFill(pad_map={"A": [0, 3]}).apply_pass(sdfg, {})


def test_zero_fill_shape_not_grown_raises():
    """A record whose original+pad does not match the current shape (array never grown) is refused."""
    sdfg = _tensordot_sdfg(4, 5, 6)  # A stays [4, 5]
    with pytest.raises(ValueError):
        # claims K was 5 and padded by 3 -> expects extent 8, but A is still 5 -> mismatch -> raise
        PadZeroFill(pad_map={"A": [0, 3]}).apply_pass(sdfg, {"PadDimensions": {"A": [4, 5]}})


if __name__ == "__main__":
    test_pad_contracted_dim_zero_fill_is_legal()
    test_pad_free_output_dim_zero_fill()
    test_pad_sum_reduction_over_pad_is_legal()
    test_pad_max_reduction_over_pad_raises()
    test_pad_max_reduction_over_free_dim_ok()
    test_zero_fill_without_pad_dimensions_raises()
    test_zero_fill_shape_not_grown_raises()
    print("pad_zero_fill tests PASS")
