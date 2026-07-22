# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra tests for the LayoutChange library node and fold_layout_changes.

Complements ``layout_change_node_test.py`` by covering paths that file leaves untested:

  * ``fold_layout_changes`` collapsing two adjacent nodes' op sequences into ONE op (concat +
    simplify) and the folded single node computing the same result as the two-node pipeline,
    bit-exact against a numpy oracle;
  * the HPTT dispatch (``ExpandHPTT``): a pure permutation delegating to a ``TensorTranspose``
    (HPTT) and a non-permutation falling back to the pure relayout map;
  * a cancelling op sequence lowering to a plain (identity-order) copy rather than a reindex;
  * the ``Pad`` op growing a dimension while copying the original region bit-exact;
  * the ``LayoutChange.validate`` guard rails (dtype / storage mismatch, missing input).
"""
import numpy
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.libraries.linalg import TensorTranspose
from dace.libraries.layout.algebra import Permute, Block, Pad, simplify_ops
from dace.libraries.layout.layout_change import (
    LayoutChange,
    ExpandPure,
    ExpandHPTT,
    add_layout_change,
    fold_layout_changes,
)

_counter = [0]


def _fresh_sdfg():
    """A uniquely named empty SDFG + its start state (unique name avoids build-cache clashes)."""
    _counter[0] += 1
    sdfg = dace.SDFG(f"lc_extra_{_counter[0]}")
    state = sdfg.add_state("s", is_start_block=True)
    return sdfg, state


def test_fold_two_nodes_to_single_runs():
    """Two chained permutes fold (concat + simplify) to ONE Permute; the folded single node
    computes the same tensor as the two-node A->B->C pipeline, bit-exact vs numpy."""
    ops1 = [Permute((1, 0, 2))]
    ops2 = [Permute((2, 0, 1))]
    first = LayoutChange("first", ops=ops1)
    second = LayoutChange("second", ops=ops2)
    folded = fold_layout_changes(first, second)

    # The concatenated sequence simplifies to a single permutation op.
    assert folded.op_sequence() == simplify_ops(ops1 + ops2)
    assert folded.op_sequence() == [Permute((2, 1, 0))]
    assert len(folded.op_sequence()) == 1

    A = numpy.random.rand(3, 4, 5)

    # Two-node pipeline: A -> B -> C.
    two_sdfg, two_state = _fresh_sdfg()
    two_sdfg.add_array("A", [3, 4, 5], dace.float64)
    add_layout_change(two_sdfg, two_state, "A", "B", ops1)
    add_layout_change(two_sdfg, two_state, "B", "C", ops2)
    assert tuple(int(s) for s in two_sdfg.arrays["C"].shape) == (5, 4, 3)
    two_sdfg.expand_library_nodes()
    two_sdfg.validate()
    B_two = numpy.zeros((4, 3, 5))
    C_two = numpy.zeros((5, 4, 3))
    two_sdfg(A=A.copy(), B=B_two, C=C_two)

    # Single folded node: A -> C.
    one_sdfg, one_state = _fresh_sdfg()
    one_sdfg.add_array("A", [3, 4, 5], dace.float64)
    add_layout_change(one_sdfg, one_state, "A", "C", folded.op_sequence())
    assert tuple(int(s) for s in one_sdfg.arrays["C"].shape) == (5, 4, 3)
    one_sdfg.expand_library_nodes()
    one_sdfg.validate()
    C_one = numpy.zeros((5, 4, 3))
    one_sdfg(A=A.copy(), C=C_one)

    assert numpy.array_equal(C_one, A.transpose(2, 1, 0))
    assert numpy.array_equal(C_one, C_two)


def test_hptt_permute_delegates_to_tensortranspose():
    """A permute-only sequence expanded through the HPTT impl becomes a nested SDFG whose only
    compute node is a TensorTranspose(implementation='HPTT') carrying the permutation axes."""
    sdfg, state = _fresh_sdfg()
    sdfg.add_array("A", [6, 8], dace.float64)
    node = add_layout_change(sdfg, state, "A", "B", [Permute((1, 0))])

    nested = ExpandHPTT.expansion(node, state, sdfg)
    tts = [n for st in nested.states() for n in st.nodes() if isinstance(n, TensorTranspose)]
    tasklets = [n for st in nested.states() for n in st.nodes() if isinstance(n, nd.Tasklet)]
    assert len(tts) == 1
    assert len(tasklets) == 0
    assert tts[0].implementation == "HPTT"
    assert tts[0].axes == [1, 0]


def test_hptt_block_falls_back_to_pure():
    """A Block (not a permutation) has no HPTT tensor-transpose path, so ExpandHPTT falls back to
    the pure relayout map: no TensorTranspose, one copy tasklet."""
    sdfg, state = _fresh_sdfg()
    sdfg.add_array("A", [64], dace.float64)
    node = add_layout_change(sdfg, state, "A", "B", [Block(0, 16)])

    nested = ExpandHPTT.expansion(node, state, sdfg)
    tts = [n for st in nested.states() for n in st.nodes() if isinstance(n, TensorTranspose)]
    tasklets = [n for st in nested.states() for n in st.nodes() if isinstance(n, nd.Tasklet)]
    assert len(tts) == 0
    assert len(tasklets) == 1


def test_cancelling_sequence_lowers_to_plain_copy():
    """A self-cancelling permute pair simplifies to the identity: same shape, a pure expansion that
    is a single copy tasklet with no TensorTranspose, and a bit-exact identity copy at run time."""
    ops = [Permute((1, 0)), Permute((1, 0))]
    assert simplify_ops(ops) == []

    sdfg, state = _fresh_sdfg()
    sdfg.add_array("A", [6, 8], dace.float64)
    node = add_layout_change(sdfg, state, "A", "B", ops)
    # Cancelling sequence keeps the packed-C shape (no transpose, no reshape).
    assert tuple(int(s) for s in sdfg.arrays["B"].shape) == (6, 8)

    # The pure expansion is a plain copy: one tasklet, zero tensor transposes.
    nested = ExpandPure.expansion(node, state, sdfg)
    tts = [n for st in nested.states() for n in st.nodes() if isinstance(n, TensorTranspose)]
    tasklets = [n for st in nested.states() for n in st.nodes() if isinstance(n, nd.Tasklet)]
    assert len(tts) == 0
    assert len(tasklets) == 1

    sdfg.expand_library_nodes()
    sdfg.validate()
    A = numpy.random.rand(6, 8)
    B = numpy.zeros((6, 8))
    sdfg(A=A.copy(), B=B)
    # Identity copy, NOT a transpose.
    assert numpy.array_equal(B, A)


def test_pad_grows_dimension_copies_region():
    """A Pad grows the output dimension; the copy fills the original region bit-exact and leaves the
    padded tail untouched (the map only ranges over the logical input extent)."""
    sdfg, state = _fresh_sdfg()
    sdfg.add_array("A", [4], dace.float64)
    add_layout_change(sdfg, state, "A", "B", [Pad(0, 2)])
    assert tuple(int(s) for s in sdfg.arrays["B"].shape) == (6, )

    sdfg.expand_library_nodes()
    sdfg.validate()
    A = numpy.arange(4, dtype=numpy.float64)
    sentinel = -7.0
    B = numpy.full((6, ), sentinel)
    sdfg(A=A.copy(), B=B)
    assert numpy.array_equal(B[:4], A)
    assert numpy.array_equal(B[4:], numpy.full((2, ), sentinel))


def test_validate_rejects_dtype_mismatch():
    """validate() raises when the input and output arrays disagree on dtype."""
    sdfg, state = _fresh_sdfg()
    sdfg.add_array("A", [4, 4], dace.float64)
    sdfg.add_array("B", [4, 4], dace.float32)
    node = LayoutChange("lc", ops=[Permute((1, 0))])
    state.add_node(node)
    rin = state.add_read("A")
    rout = state.add_write("B")
    state.add_edge(rin, None, node, "_inp", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(node, "_out", rout, None, dace.Memlet.from_array("B", sdfg.arrays["B"]))
    with pytest.raises(ValueError):
        node.validate(sdfg, state)


def test_validate_rejects_storage_mismatch():
    """validate() raises when input and output arrays live in different storage."""
    sdfg, state = _fresh_sdfg()
    sdfg.add_array("A", [4, 4], dace.float64, storage=dace.StorageType.CPU_Heap)
    sdfg.add_array("B", [4, 4], dace.float64, storage=dace.StorageType.GPU_Global)
    node = LayoutChange("lc", ops=[Permute((1, 0))])
    state.add_node(node)
    rin = state.add_read("A")
    rout = state.add_write("B")
    state.add_edge(rin, None, node, "_inp", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(node, "_out", rout, None, dace.Memlet.from_array("B", sdfg.arrays["B"]))
    with pytest.raises(ValueError):
        node.validate(sdfg, state)


def test_validate_rejects_missing_input():
    """validate() raises when the '_inp' connector has no incoming array edge."""
    sdfg, state = _fresh_sdfg()
    sdfg.add_array("B", [4, 4], dace.float64)
    node = LayoutChange("lc", ops=[Permute((1, 0))])
    state.add_node(node)
    rout = state.add_write("B")
    state.add_edge(node, "_out", rout, None, dace.Memlet.from_array("B", sdfg.arrays["B"]))
    with pytest.raises(ValueError):
        node.validate(sdfg, state)


if __name__ == "__main__":
    test_fold_two_nodes_to_single_runs()
    test_hptt_permute_delegates_to_tensortranspose()
    test_hptt_block_falls_back_to_pure()
    test_cancelling_sequence_lowers_to_plain_copy()
    test_pad_grows_dimension_copies_region()
    test_validate_rejects_dtype_mismatch()
    test_validate_rejects_storage_mismatch()
    test_validate_rejects_missing_input()
    print("LayoutChange extra tests PASS")
