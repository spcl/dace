# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for AN <-> AN copy cleanup patterns the existing cleaners miss.

Each test builds a minimal SDFG that exhibits a specific pattern observed
in cloudsc / icon kernels, where the existing cleanup family
(:class:`CleanAccessNodeToScalarSliceToTaskletPattern`,
:class:`CleanTaskletToScalarSliceToAccessNodePattern`,
:class:`RemoveRedundantAssignmentTasklets`,
:class:`ResolveOtherSubsetANEdges`) cannot fold the AN <-> AN copy with
``other_subset`` into an explicit assign tasklet.

The fixtures are hand-built (not lowered from ``@dace.program``) so the
shape is exactly the pattern the descent has to handle. Each test:

1. Builds the SDFG.
2. Runs it once to record the reference output.
3. Runs the cleanup chain.
4. Validates the SDFG.
5. Runs the cleaned SDFG and asserts numerical equivalence with the
   reference.

A test that ``xfail``-s flags a real cleanup gap to be addressed; once
the gap is closed (cleaner extended), the ``xfail`` becomes ``xpass``
and the test is flipped to a strict pass.
"""
import copy
import numpy as np
import pytest

import dace
from dace import subsets
from dace.memlet import Memlet


def _ref_run(sdfg, **arrays):
    """Run the SDFG once on a fresh deep copy of the arrays. Returns the mutated copies."""
    work = {k: v.copy() for k, v in arrays.items()}
    sdfg_copy = copy.deepcopy(sdfg)
    sdfg_copy(**work)
    return work


def _assert_equal(reference, result, *, rtol=1e-12):
    """All entries of ``result`` must match ``reference`` element-wise."""
    for k in reference:
        assert np.allclose(reference[k], result[k], rtol=rtol), (
            f"{k} mismatch after cleanup: max abs diff "
            f"{np.max(np.abs(reference[k] - result[k]))}")


def _build_an_to_an_with_other_subset_cross_state():
    """Build: state1 writes ``a -> a_slice`` (with other_subset); state2 reads
    ``a_slice`` via a tasklet that writes ``b``.

    Pattern observed in cloudsc_one: ``zqx -> zqx_index`` in one state
    (out_degree=0 in that state because the read is in a later state).
    ``CleanAccessNodeToScalarSliceToTaskletPattern`` only matches the
    single-state shape, so the multi-state pattern survives uncleaned.

    SDFG shape (one a-array of size (4,), one b-scalar element written):

        state1: a[idx] -> a_slice[0]    (other_subset preserves a's subset)
        state2: a_slice[0] -> tasklet -> b[0]
    """
    sdfg = dace.SDFG("an_to_an_cross_state")
    sdfg.add_array("a", [4], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_scalar("a_slice", dace.float64, transient=True)

    state1 = sdfg.add_state("write_a_slice")
    a1 = state1.add_access("a")
    a_slice1 = state1.add_access("a_slice")
    # AN -> AN copy. ``data`` names the destination (a_slice), subset
    # describes a_slice's [0], ``other_subset`` carries the source's
    # subset on ``a`` ([2]).
    mem = Memlet(data="a_slice", subset=subsets.Range([(0, 0, 1)]))
    mem.other_subset = subsets.Range([(2, 2, 1)])
    state1.add_edge(a1, None, a_slice1, None, mem)

    state2 = sdfg.add_state_after(state1, "read_a_slice")
    a_slice2 = state2.add_access("a_slice")
    b2 = state2.add_access("b")
    tasklet = state2.add_tasklet("scale", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    state2.add_edge(a_slice2, None, tasklet, "_in", Memlet("a_slice[0]"))
    state2.add_edge(tasklet, "_out", b2, None, Memlet("b[0]"))

    sdfg.validate()
    return sdfg


def _build_an_to_an_with_other_subset_single_state():
    """Single-state ``a -> a_slice -> tasklet -> b`` — the SHAPE the
    :class:`CleanAccessNodeToScalarSliceToTaskletPattern` cleaner DOES
    handle. Baseline that the cleanup family already supports."""
    sdfg = dace.SDFG("an_to_an_single_state")
    sdfg.add_array("a", [4], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_scalar("a_slice", dace.float64, transient=True)

    state = sdfg.add_state("compute")
    a = state.add_access("a")
    a_slice = state.add_access("a_slice")
    b = state.add_access("b")
    mem = Memlet(data="a_slice", subset=subsets.Range([(0, 0, 1)]))
    mem.other_subset = subsets.Range([(2, 2, 1)])
    state.add_edge(a, None, a_slice, None, mem)
    tasklet = state.add_tasklet("scale", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    state.add_edge(a_slice, None, tasklet, "_in", Memlet("a_slice[0]"))
    state.add_edge(tasklet, "_out", b, None, Memlet("b[0]"))

    sdfg.validate()
    return sdfg


def _build_multidim_point_an_to_an_cross_state():
    """Build the cloudsc_one ``zqx -> zqx_index`` shape.

    A 3D global ``zqx[2, 3, 4]`` read at the point ``[1, 1, 2]`` into a
    (1,) scalar transient, written in state1 (the transient has
    out_degree=0 here) and consumed by a tasklet in state2 (the read
    state). The multi-dim point subset means the AN -> AN edge carries
    ``other_subset`` but ``CleanAccessNodeToScalarSliceToTaskletPattern``
    (single-state, requires out_degree=1 in the same state) cannot
    match it. The fallback must detect the pure box (only constants
    and symbols in indices — no data-dependent gather) and lift it via
    an assign tasklet.
    """
    sdfg = dace.SDFG("multidim_point_an_to_an")
    sdfg.add_array("zqx", [2, 3, 4], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_scalar("zqx_index", dace.float64, transient=True)

    state1 = sdfg.add_state("write_zqx_index")
    zqx1 = state1.add_access("zqx")
    zqx_index1 = state1.add_access("zqx_index")
    mem = Memlet(data="zqx", subset=subsets.Range([(1, 1, 1), (1, 1, 1), (2, 2, 1)]))
    mem.other_subset = subsets.Range([(0, 0, 1)])
    state1.add_edge(zqx1, None, zqx_index1, None, mem)

    state2 = sdfg.add_state_after(state1, "read_zqx_index")
    zqx_index2 = state2.add_access("zqx_index")
    b2 = state2.add_access("b")
    tasklet = state2.add_tasklet("scale", {"_in"}, {"_out"}, "_out = 3.0 * _in")
    state2.add_edge(zqx_index2, None, tasklet, "_in", Memlet("zqx_index[0]"))
    state2.add_edge(tasklet, "_out", b2, None, Memlet("b[0]"))

    sdfg.validate()
    return sdfg


def _build_tasklet_to_global_with_other_subset_cross_state():
    """Build: state1 has ``tasklet -> a_slice -> a`` with other_subset on
    the AN-AN edge; ``a_slice`` is read in state2.

    Pattern observed in cloudsc loopnests where a write goes through a
    scalar slice that is also read elsewhere. The
    :class:`CleanTaskletToScalarSliceToAccessNodePattern` matcher
    requires single-state in/out_degree=1 on the slice; multi-state
    reuse falls through.
    """
    sdfg = dace.SDFG("tasklet_to_global_cross_state")
    sdfg.add_array("a", [4], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_scalar("a_slice", dace.float64, transient=True)

    state1 = sdfg.add_state("write_through_slice")
    tasklet1 = state1.add_tasklet("produce", {}, {"_out"}, "_out = 7.0")
    a_slice1 = state1.add_access("a_slice")
    a1 = state1.add_access("a")
    state1.add_edge(tasklet1, "_out", a_slice1, None, Memlet("a_slice[0]"))
    # AN -> AN copy: a_slice -> a with other_subset preserving a's
    # destination position.
    mem = Memlet(data="a", subset=subsets.Range([(1, 1, 1)]))
    mem.other_subset = subsets.Range([(0, 0, 1)])
    state1.add_edge(a_slice1, None, a1, None, mem)

    state2 = sdfg.add_state_after(state1, "read_slice_again")
    a_slice2 = state2.add_access("a_slice")
    b2 = state2.add_access("b")
    tasklet2 = state2.add_tasklet("scale", {"_in"}, {"_out"}, "_out = 2.0 * _in")
    state2.add_edge(a_slice2, None, tasklet2, "_in", Memlet("a_slice[0]"))
    state2.add_edge(tasklet2, "_out", b2, None, Memlet("b[0]"))

    sdfg.validate()
    return sdfg


def _run_cleanup(sdfg):
    """Apply the cleanup chain to ``sdfg`` in place."""
    from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
    from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
        CleanAccessNodeToScalarSliceToTaskletPattern, )
    from dace.transformation.passes.clean_tasklet_to_scalar_slice_to_access_node_pattern import (
        CleanTaskletToScalarSliceToAccessNodePattern, )
    from dace.transformation.passes.remove_redundant_assignment_tasklets import RemoveRedundantAssignmentTasklets
    from dace.transformation.passes.vectorization.eliminate_dead_copies import EliminateDeadCopies
    from dace.transformation.passes.vectorization.resolve_other_subset_an_edges import ResolveOtherSubsetANEdges

    sdfg.apply_transformations_repeated(TrivialTaskletElimination, permissive=False, validate=False)
    RemoveRedundantAssignmentTasklets().apply_pass(sdfg, {})
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    CleanTaskletToScalarSliceToAccessNodePattern().apply_pass(sdfg, {})
    EliminateDeadCopies().apply_pass(sdfg, {})
    # Last-resort: resolve any remaining ``other_subset`` AN <-> AN
    # edges by inserting an ``_out = _in`` tasklet so the descent can
    # see the read / write. Walks every SDFG (top-level + nested).
    # Multi-state writers (the ``A -> A_slice`` in state1 + ``A_slice
    # -> tasklet`` in state2 cloudsc shape) are handled here too — the
    # inserted tasklet on the writer side gives the descent's classify
    # / promote walker a visible dataflow without any value
    # substitution.
    ResolveOtherSubsetANEdges().apply_pass(sdfg, {})


def _other_subset_remains(sdfg) -> bool:
    """True iff any memlet anywhere still carries ``other_subset``."""
    for s in sdfg.all_sdfgs_recursive():
        for state in s.states():
            for edge in state.edges():
                if edge.data is not None and edge.data.other_subset is not None:
                    return True
    return False


def test_baseline_single_state_an_to_tasklet_cleans():
    """Single-state ``a -> a_slice -> tasklet -> b`` — the cleaner SHOULD
    fold this (this is the documented happy path)."""
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    sdfg = _build_an_to_an_with_other_subset_single_state()
    ref = _ref_run(sdfg, a=a, b=b)
    _run_cleanup(sdfg)
    sdfg.validate()
    assert not _other_subset_remains(sdfg), "Single-state AN -> AN cleanup should fold the slice"
    a_run, b_run = a.copy(), b.copy()
    sdfg(a=a_run, b=b_run)
    _assert_equal(ref, {"a": a_run, "b": b_run})


def test_cross_state_an_to_tasklet_cleanup_fallback():
    """``a -> a_slice`` in state1, ``a_slice -> tasklet -> b`` in state2.

    Single-state CleanAccessNode does not match (the dest is read in a
    later state). The fallback ResolveOtherSubsetANEdges must still
    insert an assign tasklet so no ``other_subset`` remains and the
    cleaned SDFG produces the same output.
    """
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    sdfg = _build_an_to_an_with_other_subset_cross_state()
    ref = _ref_run(sdfg, a=a, b=b)
    _run_cleanup(sdfg)
    sdfg.validate()
    assert not _other_subset_remains(sdfg), (
        "Cross-state AN -> AN with other_subset must be lowered by the cleanup chain "
        "(fallback to ResolveOtherSubsetANEdges with an inserted assign tasklet)")
    a_run, b_run = a.copy(), b.copy()
    sdfg(a=a_run, b=b_run)
    _assert_equal(ref, {"a": a_run, "b": b_run})


def test_cross_state_multidim_point_pure_box_cleanup_fallback():
    """``zqx -> zqx_index`` (3D source point, 1D dest) cross-state.

    Mirrors cloudsc_one's ``zqx[z1, j, i] -> zqx_index`` pattern. The
    fallback ResolveOtherSubsetANEdges must detect the pure box and
    insert an assign tasklet.
    """
    zqx = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    b = np.zeros(1, dtype=np.float64)
    sdfg = _build_multidim_point_an_to_an_cross_state()
    ref = _ref_run(sdfg, zqx=zqx, b=b)
    _run_cleanup(sdfg)
    sdfg.validate()
    assert not _other_subset_remains(sdfg), (
        "Multi-dim pure-box AN -> AN with other_subset must be lowered "
        "via an assign tasklet by ResolveOtherSubsetANEdges")
    zqx_run, b_run = zqx.copy(), b.copy()
    sdfg(zqx=zqx_run, b=b_run)
    _assert_equal(ref, {"zqx": zqx_run, "b": b_run})


def test_cross_state_tasklet_to_global_cleanup_fallback():
    """``tasklet -> a_slice -> a`` in state1 + ``a_slice -> tasklet -> b`` in state2.

    Mirror of the previous test on the write side. Same fallback
    contract: the cleanup chain must remove the residual
    ``other_subset``.
    """
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    sdfg = _build_tasklet_to_global_with_other_subset_cross_state()
    ref = _ref_run(sdfg, a=a, b=b)
    _run_cleanup(sdfg)
    sdfg.validate()
    assert not _other_subset_remains(sdfg), (
        "Cross-state tasklet -> AN -> AN with other_subset must be lowered by the cleanup chain "
        "(fallback to ResolveOtherSubsetANEdges with an inserted assign tasklet)")
    a_run, b_run = a.copy(), b.copy()
    sdfg(a=a_run, b=b_run)
    _assert_equal(ref, {"a": a_run, "b": b_run})


if __name__ == "__main__":
    test_baseline_single_state_an_to_tasklet_cleans()
    test_cross_state_an_to_tasklet_cleanup_fallback()
    test_cross_state_multidim_point_pure_box_cleanup_fallback()
    test_cross_state_tasklet_to_global_cleanup_fallback()
    print("OK")
