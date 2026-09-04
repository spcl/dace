# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the LayoutChange library node: pure relayout expansion, op-sequence serialization,
node chaining/folding, and the permute -> TensorTranspose (cuTENSOR/HPTT) dispatch selection."""
import json
import numpy
import dace

from dace.sdfg import nodes as nd
from dace.libraries.layout.algebra import Permute, Block, Unblock, is_identity
from dace.libraries.layout.layout_change import (
    LayoutChange,
    ExpandCuTensor,
    add_layout_change,
    fold_layout_changes,
)

_counter = [0]


def _make(shape, ops):
    """A one-state SDFG: input A -> LayoutChange -> output B."""
    _counter[0] += 1
    sdfg = dace.SDFG(f"lc_{_counter[0]}")
    sdfg.add_array("A", shape, dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    node = add_layout_change(sdfg, state, "A", "B", ops)
    return sdfg, state, node


def test_node_permute_is_transpose():
    sdfg, _, _ = _make([6, 8], [Permute((1, 0))])
    sdfg.expand_library_nodes()
    sdfg.validate()
    A = numpy.random.rand(6, 8)
    B = numpy.zeros((8, 6))
    sdfg(A=A.copy(), B=B)
    assert numpy.array_equal(B, A.T)


def test_node_block_is_reshape():
    sdfg, _, _ = _make([64], [Block(0, 16)])
    assert tuple(int(s) for s in sdfg.arrays["B"].shape) == (4, 16)
    sdfg.expand_library_nodes()
    sdfg.validate()
    A = numpy.random.rand(64)
    B = numpy.zeros((4, 16))
    sdfg(A=A.copy(), B=B)
    assert numpy.array_equal(B, A.reshape(4, 16))


def test_node_block_unblock_folds_to_copy():
    sdfg, _, _ = _make([64], [Block(0, 16), Unblock(0, 16)])
    # simplify_ops folds Block∘Unblock -> [] so the output stays 1-D.
    assert tuple(int(s) for s in sdfg.arrays["B"].shape) == (64, )
    sdfg.expand_library_nodes()
    sdfg.validate()
    A = numpy.random.rand(64)
    B = numpy.zeros((64, ))
    sdfg(A=A.copy(), B=B)
    assert numpy.array_equal(B, A)


def test_two_chained_nodes_run():
    """A -> B (transpose) -> C (transpose back): both nodes expand independently, C == A."""
    sdfg, state, _ = _make([6, 8], [Permute((1, 0))])
    add_layout_change(sdfg, state, "B", "C", [Permute((1, 0))])
    sdfg.expand_library_nodes()
    sdfg.validate()
    A = numpy.random.rand(6, 8)
    B = numpy.zeros((8, 6))
    C = numpy.zeros((6, 8))
    sdfg(A=A.copy(), B=B, C=C)
    assert numpy.array_equal(C, A)


def test_fold_inverse_pair_is_identity():
    a = LayoutChange("a", ops=[Permute((1, 0))])
    b = LayoutChange("b", ops=[Permute((1, 0))])
    folded = fold_layout_changes(a, b)
    assert folded.op_sequence() == []
    assert is_identity(a.op_sequence() + b.op_sequence())


def test_op_sequence_serialization_roundtrip():
    ops = [Permute((1, 0)), Block(0, 16)]
    node = LayoutChange("x", ops=ops)
    # Stored as JSON, decoded back to identical op dataclasses.
    assert isinstance(node.ops, str)
    assert json.loads(node.ops)  # valid JSON
    assert node.op_sequence() == ops


def test_sdfg_json_roundtrip_preserves_ops():
    sdfg, _, _ = _make([6, 8], [Permute((1, 0)), Block(0, 16)])
    sdfg2 = dace.SDFG.from_json(sdfg.to_json())
    lc = [n for st in sdfg2.states() for n in st.nodes() if isinstance(n, LayoutChange)]
    assert len(lc) == 1
    assert lc[0].op_sequence() == [Permute((1, 0)), Block(0, 16)]


def test_cutensor_dispatch_permute_uses_tensortranspose():
    """A pure permutation lowers to a nested SDFG containing a TensorTranspose (cuTENSOR)."""
    from dace.libraries.linalg import TensorTranspose
    sdfg, state, node = _make([6, 8], [Permute((1, 0))])
    nested = ExpandCuTensor.expansion(node, state, sdfg)
    tts = [n for st in nested.states() for n in st.nodes() if isinstance(n, TensorTranspose)]
    assert len(tts) == 1
    assert tts[0].axes == [1, 0]
    assert tts[0].implementation == "cuTENSOR"


def test_cutensor_dispatch_block_falls_back_to_pure():
    """A block (not a permutation) has no cuTENSOR path -> falls back to the pure relayout map."""
    from dace.libraries.linalg import TensorTranspose
    sdfg, state, node = _make([64], [Block(0, 16)])
    nested = ExpandCuTensor.expansion(node, state, sdfg)
    tts = [n for st in nested.states() for n in st.nodes() if isinstance(n, TensorTranspose)]
    assert len(tts) == 0
    tasklets = [n for st in nested.states() for n in st.nodes() if isinstance(n, nd.Tasklet)]
    assert len(tasklets) == 1


if __name__ == "__main__":
    test_node_permute_is_transpose()
    test_node_block_is_reshape()
    test_node_block_unblock_folds_to_copy()
    test_two_chained_nodes_run()
    test_fold_inverse_pair_is_identity()
    test_op_sequence_serialization_roundtrip()
    test_sdfg_json_roundtrip_preserves_ops()
    test_cutensor_dispatch_permute_uses_tensortranspose()
    test_cutensor_dispatch_block_falls_back_to_pure()
    print("LayoutChange node tests PASS")
