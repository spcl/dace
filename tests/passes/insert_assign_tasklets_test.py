# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the assign-tasklet insertion cleanup passes.

``InsertAssignTaskletsForUnitCopies`` rewrites single-element
``AccessNode -> AccessNode`` copies into ``_out = _in`` tasklets;
``InsertAssignTaskletsAtMapBoundary`` does the same for map-boundary staging
edges and ``other_subset`` copies. Every SDFG is built with the constructor
API, asserted valid before and after, and compared end-to-end against a
deep-copied pre-pass reference run.
"""
import copy

import numpy as np
import pytest

import dace
from dace import nodes
from dace.transformation.passes.insert_unit_copy_assign_tasklets import InsertAssignTaskletsForUnitCopies
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary


def _an_to_an_edges(sdfg: dace.SDFG):
    out = []
    for state in sdfg.states():
        for e in state.edges():
            if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode):
                out.append(e)
    return out


def _assign_tasklets(sdfg: dace.SDFG):
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Tasklet) and n.code.as_string.strip() == "_out = _in"
    ]


# --- InsertAssignTaskletsForUnitCopies ---------------------------------------


def test_unit_copy_with_other_subset_is_split():
    """``A[5] -[B[2]]-> B`` is a single element on both sides: rewritten."""
    sdfg = dace.SDFG('unit_copy_othersub')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    b = state.add_access('B')
    state.add_edge(a, None, b, None, dace.Memlet(data='A', subset='5', other_subset='2'))
    sdfg.validate()

    ref = copy.deepcopy(sdfg)
    A = np.random.rand(10)
    B0 = np.full(10, -1.0)
    pre = B0.copy()
    ref(A=A, B=pre)

    changed = InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {})
    assert changed == 1
    sdfg.validate()
    assert not _an_to_an_edges(sdfg)
    assert len(_assign_tasklets(sdfg)) == 1

    post = B0.copy()
    sdfg(A=A, B=post)
    assert np.allclose(post, pre)
    assert post[2] == A[5]


def test_multi_element_copy_is_left_unchanged():
    """``A[0:5] -> B[0:5]`` moves five elements: must not be rewritten."""
    sdfg = dace.SDFG('multi_copy')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    b = state.add_access('B')
    state.add_edge(a, None, b, None, dace.Memlet(data='A', subset='0:5', other_subset='0:5'))
    sdfg.validate()

    changed = InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {})
    assert changed is None
    assert len(_an_to_an_edges(sdfg)) == 1
    assert not _assign_tasklets(sdfg)


def test_scalar_copy_without_other_subset_is_split():
    """A plain single-element copy (no ``other_subset``) is rewritten."""
    sdfg = dace.SDFG('scalar_copy')
    sdfg.add_array('S1', [1], dace.float64)
    sdfg.add_array('S2', [1], dace.float64)
    state = sdfg.add_state()
    s1 = state.add_access('S1')
    s2 = state.add_access('S2')
    state.add_edge(s1, None, s2, None, dace.Memlet(data='S1', subset='0'))
    sdfg.validate()

    ref = copy.deepcopy(sdfg)
    S1 = np.random.rand(1)
    pre = np.full(1, -1.0)
    ref(S1=S1, S2=pre)

    changed = InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {})
    assert changed == 1
    sdfg.validate()
    assert not _an_to_an_edges(sdfg)

    post = np.full(1, -1.0)
    sdfg(S1=S1, S2=post)
    assert np.allclose(post, pre)
    assert post[0] == S1[0]


def test_symbolic_extent_copy_is_left_unchanged():
    """A copy whose extent is the symbol ``N`` is not provably unit: skipped."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('symbolic_copy')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    b = state.add_access('B')
    state.add_edge(a, None, b, None, dace.Memlet(data='A', subset='0:N', other_subset='0:N'))
    sdfg.validate()

    changed = InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {})
    assert changed is None
    assert len(_an_to_an_edges(sdfg)) == 1


# --- InsertAssignTaskletsAtMapBoundary ---------------------------------------


def _staging_sdfg() -> dace.SDFG:
    """``A -> map -> tmp -> (+1) -> tmp2 -> mapexit -> B`` with explicit
    in-scope staging AccessNodes (the stage-in / stage-out boundary)."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('map_staging')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('tmp', [1], dace.float64)
    sdfg.add_transient('tmp2', [1], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    b = state.add_access('B')
    me, mx = state.add_map('m', dict(i='0:N'))
    tin = state.add_access('tmp')
    tout = state.add_access('tmp2')
    t = state.add_tasklet('add', {'_i'}, {'_o'}, '_o = _i + 1.0')
    state.add_memlet_path(a, me, tin, memlet=dace.Memlet(data='A', subset='i', other_subset='0'))
    state.add_edge(tin, None, t, '_i', dace.Memlet(data='tmp', subset='0'))
    state.add_edge(t, '_o', tout, None, dace.Memlet(data='tmp2', subset='0'))
    state.add_memlet_path(tout, mx, b, memlet=dace.Memlet(data='B', subset='i', other_subset='0'))
    sdfg.validate()
    return sdfg


def test_map_boundary_staging_is_split():
    sdfg = _staging_sdfg()
    ref = copy.deepcopy(sdfg)
    A = np.random.rand(16)
    pre = np.full(16, -1.0)
    ref(A=A, B=pre, N=16)

    changed = InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    assert changed is not None and changed >= 2
    sdfg.validate()

    post = np.full(16, -1.0)
    sdfg(A=A, B=post, N=16)
    assert np.allclose(post, pre)
    assert np.allclose(post, A + 1.0)


def test_map_boundary_other_subset_an_edge_is_split():
    """A bare ``other_subset`` ``AccessNode -> AccessNode`` copy is split."""
    sdfg = dace.SDFG('othersub_an')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    b = state.add_access('B')
    state.add_edge(a, None, b, None, dace.Memlet(data='A', subset='5', other_subset='2'))
    sdfg.validate()

    # No reference run here: DaCe codegen for a bare ``other_subset`` AN->AN
    # copy is the unreliable shape this pass exists to remove, so the pre-pass
    # SDFG is not a valid oracle. Assert analytically.
    A = np.random.rand(10)

    changed = InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    assert changed == 1
    sdfg.validate()
    assert not _an_to_an_edges(sdfg)

    post = np.full(10, -1.0)
    sdfg(A=A, B=post)
    exp = np.full(10, -1.0)
    exp[2] = A[5]
    assert np.allclose(post, exp)


def test_an_to_an_with_other_subset_preserves_wcr():
    """``AccessNode -[wcr]-> AccessNode`` with ``other_subset`` set must keep
    its WCR semantics across the split: the inserted tasklet's output edge
    carries the WCR so the write into the destination still accumulates.

    Regression: dropping the WCR here turned a sequential reduction
    accumulator into a plain overwrite -- LoopToReduce(wcr-scalar) emits
    exactly this shape (privatised ``_priv_X`` source -> ``_priv_X``
    AccessNode WCR write) and the lost WCR corrupted ``c[0] = sum(a*b)``
    style canonicalize-pipeline kernels.
    """
    sdfg = dace.SDFG("wcr_an_to_an")
    sdfg.add_array("S", [4], dace.float64, transient=False)
    sdfg.add_scalar("acc", dace.float64, transient=True)
    state = sdfg.add_state()
    src = state.add_access("S")
    dst = state.add_access("acc")
    state.add_edge(src, None, dst, None, dace.Memlet(data="S", subset="2", other_subset="0", wcr="lambda a, b: a + b"))
    sdfg.validate()

    changed = InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    assert changed == 1
    sdfg.validate()
    assert not _an_to_an_edges(sdfg)

    # Exactly one WCR edge must remain -- on the OUTPUT side of the inserted
    # tasklet (tasklet -> dst_an), preserving the accumulator semantics.
    wcr_edges = [e for st in sdfg.states() for e in st.edges() if e.data is not None and e.data.wcr is not None]
    assert len(wcr_edges) == 1, f'expected one preserved WCR edge, got {len(wcr_edges)}'
    (we, ) = wcr_edges
    assert isinstance(we.src, nodes.Tasklet), 'WCR must move to the tasklet output side'
    assert isinstance(we.dst, nodes.AccessNode) and we.dst.data == "acc"


def test_map_exit_stage_out_preserves_wcr():
    """``AccessNode -[wcr]-> MapExit`` stage-out edge must keep its WCR
    after the split: the inserted tasklet's output edge to the MapExit
    carries the WCR, so each thread's contribution still accumulates at
    the outer write target.

    Regression: this is the shape ``LoopToReduce(prefer='wcr-scalar')``
    followed by ``LoopToMap`` lands at the canonicalize pipeline boundary
    -- the wcr-scalar emit's WCR write to ``_priv_X`` ends up as
    ``_priv_X -[wcr]-> MapExit -> outer _priv_X`` after parallelisation,
    and ``InsertAssignTaskletsAtMapBoundary`` previously stripped the WCR
    from the stage-out path.
    """
    sdfg = dace.SDFG("wcr_stage_out")
    sdfg.add_array("A", [16], dace.float64, transient=False)
    sdfg.add_scalar("acc", dace.float64, transient=True)
    sdfg.add_array("_priv_acc", [1], dace.float64, transient=True)
    state = sdfg.add_state()
    a_an = state.add_access("A")
    priv_an = state.add_access("_priv_acc")
    me, mx = state.add_map("m", {"i": "0:16"})
    t = state.add_tasklet("compute", {"_in"}, {"_out"}, "_out = _in")
    state.add_memlet_path(a_an, me, t, dst_conn="_in", memlet=dace.Memlet("A[i]"))
    # The local scalar accumulator inside the map.
    acc_local = state.add_access("acc")
    state.add_edge(t, "_out", acc_local, None, dace.Memlet("acc[0]"))
    # Stage-out edge with WCR -- this is what must survive the split.
    state.add_memlet_path(acc_local, mx, priv_an, memlet=dace.Memlet("_priv_acc[0]", wcr="lambda a, b: a + b"))
    sdfg.validate()

    InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    sdfg.validate()

    # WCR must be preserved on the tasklet -> MapExit output side.
    wcr_to_map_exit = [
        e for st in sdfg.states() for e in st.edges()
        if e.data is not None and e.data.wcr is not None and isinstance(e.dst, nodes.MapExit)
    ]
    assert len(wcr_to_map_exit) == 1, (f'stage-out split must keep exactly one WCR edge into MapExit; '
                                       f'got {len(wcr_to_map_exit)}')
    we = wcr_to_map_exit[0]
    assert isinstance(we.src, nodes.Tasklet), 'WCR must move to the tasklet output side of the split'


def test_wcr_stage_out_executes_correctly_after_split():
    """End-to-end numerical regression. Build a Map that accumulates ``A[i]``
    into a transient scalar via WCR (the canonical wcr-scalar shape
    ``LoopToReduce(prefer='wcr-scalar') + LoopToMap`` lands), apply
    ``InsertAssignTaskletsAtMapBoundary`` to split the stage-out path, and
    assert the executed SDFG still computes ``sum(A)``. If the WCR semantics
    were dropped during the split (the regression this commit's pass fix
    closes), the result would be a single ``A[last]`` write instead of the
    full sum.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG("wcr_stage_out_exec")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("acc", [1], dace.float64)
    state = sdfg.add_state()
    a_an = state.add_access("A")
    acc_an = state.add_access("acc")
    me, mx = state.add_map("m", {"i": "0:N"})
    t = state.add_tasklet("inc", {"_in"}, {"_out"}, "_out = _in")
    state.add_memlet_path(a_an, me, t, dst_conn="_in", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(t, mx, acc_an, src_conn="_out", memlet=dace.Memlet("acc[0]", wcr="lambda a, b: a + b"))
    sdfg.validate()

    InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    sdfg.validate()

    A = np.random.rand(16)
    out = np.zeros(1)
    sdfg(A=A, acc=out, N=16)
    assert np.isclose(out[0], A.sum()), (f'WCR semantics lost across the split: got {out[0]}, '
                                         f'expected sum(A)={A.sum()}')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
