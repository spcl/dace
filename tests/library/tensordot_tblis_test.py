# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the TBLIS expansion of the TensorDot library node (native CPU contraction).

The label-derivation and codegen checks run everywhere (no TBLIS needed): the einsum labels
are correct iff ``np.einsum(labels)`` equals ``np.tensordot``. The execution test is marked
``tblis`` and only runs where the TBLIS library is installed (``-m tblis``).
"""
import numpy as np
import pytest
import dace
from dace.memlet import Memlet
from dace.libraries.linalg.nodes.tensordot import TensorDot, ExpandTBLIS

# (left_shape, right_shape, left_axes, right_axes, permutation)
CASES = [
    ((4, 5), (5, 6), [1], [0], None),  # matmul
    ((4, 5), (5, 6), [1], [0], [1, 0]),  # matmul, transposed output
    ((3, 4, 5), (3, 5, 6), [2], [1], None),  # single contracted axis, outer over both dim0
    ((4, 5, 6), (5, 6, 7), [1, 2], [0, 1], None),  # two contracted axes
    ((4, 5, 6), (6, 5), [2, 1], [0, 1], None),  # contracted axes in swapped order
]


def _make_sdfg(left_shape, right_shape, left_axes, right_axes, permutation, dtype=dace.float64, impl="TBLIS"):
    dot_shape = [s for i, s in enumerate(left_shape) if i not in left_axes]
    dot_shape += [s for i, s in enumerate(right_shape) if i not in right_axes]
    out_shape = [dot_shape[p] for p in permutation] if permutation else dot_shape

    sdfg = dace.SDFG("tblis_tensordot")
    state = sdfg.add_state()
    sdfg.add_array("A", left_shape, dtype)
    sdfg.add_array("B", right_shape, dtype)
    sdfg.add_array("C", out_shape or [1], dtype)
    a, b, c = state.add_read("A"), state.add_read("B"), state.add_write("C")
    node = TensorDot("dot", left_axes=left_axes, right_axes=right_axes, permutation=permutation)
    node.implementation = impl
    state.add_edge(a, None, node, "_left_tensor", Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(b, None, node, "_right_tensor", Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(node, "_out_tensor", c, None, Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def test_tblis_registered():
    assert "TBLIS" in TensorDot("t").implementations


@pytest.mark.parametrize("left_shape,right_shape,left_axes,right_axes,permutation", CASES)
def test_contraction_labels_match_numpy(left_shape, right_shape, left_axes, right_axes, permutation):
    """The TBLIS index labels are correct iff einsum(labels) == np.tensordot(+permute)."""
    A = np.random.rand(*left_shape)
    B = np.random.rand(*right_shape)
    ia, ib, ic = ExpandTBLIS.contraction_labels(len(left_shape), len(right_shape), left_axes, right_axes, permutation)
    ref = np.tensordot(A, B, axes=(left_axes, right_axes))
    if permutation:
        ref = ref.transpose(permutation)
    got = np.einsum(f"{ia},{ib}->{ic}", A, B)
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


def test_expansion_emits_tblis_call():
    """Matmul expands to a tblis_tensor_mult(ab,bc->ac) with a scaled (beta=0) C; SDFG stays valid."""
    sdfg = _make_sdfg((4, 4), (4, 4), [1], [0], None)
    sdfg.expand_library_nodes()
    code = "".join(t.code.as_string for st in sdfg.states() for t in st.nodes() if isinstance(t, dace.nodes.Tasklet))
    assert 'tblis_tensor_mult(NULL, NULL, &A, "ab", &B, "bc", &C, "ac")' in code
    assert "tblis_init_tensor_scaled_d(&C" in code
    sdfg.validate()


def test_unsupported_dtype_raises():
    sdfg = _make_sdfg((4, 4), (4, 4), [1], [0], None, dtype=dace.complex64)
    with pytest.raises(NotImplementedError):
        sdfg.expand_library_nodes()


@pytest.mark.tblis
@pytest.mark.parametrize("left_shape,right_shape,left_axes,right_axes,permutation", CASES)
def test_tblis_execution(left_shape, right_shape, left_axes, right_axes, permutation):
    """Compile + run through TBLIS and compare to numpy. Requires the TBLIS library."""
    A = np.random.rand(*left_shape)
    B = np.random.rand(*right_shape)
    ref = np.tensordot(A, B, axes=(left_axes, right_axes))
    if permutation:
        ref = ref.transpose(permutation)

    sdfg = _make_sdfg(left_shape, right_shape, left_axes, right_axes, permutation)
    C = np.zeros(ref.shape or [1])
    sdfg(A=A.copy(), B=B.copy(), C=C)
    assert np.allclose(C, ref)
