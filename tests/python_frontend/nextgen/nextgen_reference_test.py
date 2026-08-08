# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for Reference support in the next-generation frontend: annotated
reference declarations (``ref: dace.data.ArrayReference(...) = A``), bare
declarations, runtime re-pointing via :class:`RefSetNode`, pointer-copy
semantics for reference-to-reference assignment (double-buffering swaps), and
unannotated branch-divergent aliases merging through references.
"""
import numpy as np
import pytest

import dace
from dace import data
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


@dace.program
def _branch_refs(A: dace.float64[20], B: dace.float64[20], i: dace.int32[1], out: dace.float64[20]):
    if i[0] < 5:
        ref = A
    else:
        ref = B
    out[:] = ref


@dace.program
def _dblbuf(A: dace.float64[N], B: dace.float64[N], T: dace.int32, out: dace.float64[N]):
    ref1: dace.data.ArrayReference(A.dtype, A.shape) = A
    ref2: dace.data.ArrayReference(A.dtype, A.shape) = B
    tmp: dace.data.ArrayReference(A.dtype, A.shape)
    for i in range(T):
        ref2[:] = ref1[:] + i * ref2[:]
        # Swap references
        tmp = ref1
        ref1 = ref2
        ref2 = tmp

    out[:] = ref1[:]


@dace.program
def _dblbuf_tuple_swap(A: dace.float64[N], B: dace.float64[N], T: dace.int32, out: dace.float64[N]):
    ref1: dace.data.ArrayReference(A.dtype, A.shape) = A
    ref2: dace.data.ArrayReference(A.dtype, A.shape) = B
    for i in range(T):
        ref2[:] = ref1[:] + i * ref2[:]
        # Swap references
        ref1, ref2 = ref2, ref1

    out[:] = ref1[:]


def _dblbuf_expected(A, B, T):
    r1, r2 = A.copy(), B.copy()
    for i in range(T):
        r2[:] = r1 + i * r2
        r1, r2 = r2, r1
    return r1


def test_unannotated_branch_reference():
    tree = nextgen.parse_program(_branch_refs)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    refsets = _nodes_of_type(tree, tn.RefSetNode)
    assert len(refsets) == 2
    assert {refset.memlet.data for refset in refsets} == {'A', 'B'}
    merged = refsets[0].target
    assert refsets[1].target == merged
    assert isinstance(tree.containers[merged], data.Reference)


@pytest.mark.parametrize('selector', [3, 7])
def test_unannotated_branch_reference_execution(selector):
    tree = nextgen.parse_program(_branch_refs)
    sdfg = tree.as_sdfg()
    A = np.random.rand(20)
    B = np.random.rand(20)
    out = np.zeros(20)
    sdfg(A=A, B=B, i=np.array([selector], np.int32), out=out)
    assert np.allclose(out, A if selector < 5 else B)


def test_annotated_reference_swap():
    tree = nextgen.parse_program(_dblbuf)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # Two initializations and three swap refsets per iteration body
    refsets = _nodes_of_type(tree, tn.RefSetNode)
    assert len(refsets) == 5
    for name in ('ref1', 'ref2', 'tmp'):
        assert isinstance(tree.containers[name], data.Reference)
    assert len(_nodes_of_type(tree, tn.ForScope)) == 1


@pytest.mark.parametrize('trip_count', [3, 4])
def test_annotated_reference_swap_execution(trip_count):
    tree = nextgen.parse_program(_dblbuf)
    sdfg = tree.as_sdfg()
    A0 = np.random.rand(16)
    B0 = np.random.rand(16)
    A, B = A0.copy(), B0.copy()
    out = np.zeros(16)
    sdfg(A=A, B=B, T=np.int32(trip_count), out=out, N=16)
    assert np.allclose(out, _dblbuf_expected(A0, B0, trip_count))


def test_tuple_swap_reference():
    tree = nextgen.parse_program(_dblbuf_tuple_swap)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # The tuple swap desugars into reference temporaries (pointer copies)
    refsets = _nodes_of_type(tree, tn.RefSetNode)
    assert len(refsets) == 6
    references = [name for name, desc in tree.containers.items() if isinstance(desc, data.Reference)]
    assert len(references) == 4  # ref1, ref2, and two unpack temporaries


@pytest.mark.parametrize('trip_count', [3, 4])
def test_tuple_swap_reference_execution(trip_count):
    tree = nextgen.parse_program(_dblbuf_tuple_swap)
    sdfg = tree.as_sdfg()
    A0 = np.random.rand(16)
    B0 = np.random.rand(16)
    A, B = A0.copy(), B0.copy()
    out = np.zeros(16)
    sdfg(A=A, B=B, T=np.int32(trip_count), out=out, N=16)
    assert np.allclose(out, _dblbuf_expected(A0, B0, trip_count))


def test_reference_set_to_computed_value():

    @dace.program
    def computed(A: dace.float64[10], out: dace.float64[10]):
        ref: dace.data.ArrayReference(A.dtype, A.shape) = A
        ref = A + 1.0
        out[:] = ref[:]

    tree = nextgen.parse_program(computed)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # The computed value materializes into a container the reference points to
    refsets = _nodes_of_type(tree, tn.RefSetNode)
    assert len(refsets) == 2

    sdfg = tree.as_sdfg()
    A = np.random.rand(10)
    out = np.zeros(10)
    sdfg(A=A, out=out)
    assert np.allclose(out, A + 1.0)


def test_write_through_merged_reference():
    # After an unannotated branch join, writing through the merged name must
    # reach the branch's actual container (Python aliasing), not a copy.

    @dace.program
    def write_through(A: dace.float64[8], B: dace.float64[8], flag: dace.int32):
        if flag > 0:
            ref = A
        else:
            ref = B
        ref[:] = 42.0

    tree = nextgen.parse_program(write_through)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    sdfg = tree.as_sdfg()
    A = np.zeros(8)
    B = np.zeros(8)
    sdfg(A=A, B=B, flag=np.int32(1))
    assert np.allclose(A, 42.0)
    assert np.allclose(B, 0.0)


if __name__ == '__main__':
    test_unannotated_branch_reference()
    test_unannotated_branch_reference_execution(3)
    test_unannotated_branch_reference_execution(7)
    test_annotated_reference_swap()
    test_annotated_reference_swap_execution(3)
    test_annotated_reference_swap_execution(4)
    test_tuple_swap_reference()
    test_tuple_swap_reference_execution(3)
    test_tuple_swap_reference_execution(4)
    test_reference_set_to_computed_value()
    test_write_through_merged_reference()
