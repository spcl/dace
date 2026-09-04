# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``NormalizeLoopAndMapOrigin``: every Map range / ``LoopRegion``
    counter is rebased to a 0-based begin while KEEPING its stride (unlike
    ``NormalizeLoopsAndMaps``, which folds the stride into the index).
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin

N = dace.symbol('N')


@dace.program
def strided_map(A: dace.float64[40], B: dace.float64[40]):
    for i in dace.map[2:10:3]:
        B[i] = A[i] * 2.0 + 1.0


@dace.program
def loop_bounds(A: dace.float64[N]):
    for i in range(1, N - 2):
        A[i] = A[i] * 3.0 + 1.0


@dace.program
def already_zero_based(A: dace.float64[N]):
    for i in range(0, N):
        A[i] = A[i] + 1.0


@dace.program
def triangular_nest(A: dace.float64[20, 20]):
    for i in range(1, N):
        for j in range(i, N):
            A[i, j] = A[i, j] + i * 10.0 + j


def _map_ranges(sdfg: dace.SDFG):
    out = []
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            out.extend(n.map.range.ranges)
    return out


def _loops(sdfg: dace.SDFG):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable]


def test_map_rebased_keeping_stride():
    """``2:10:3`` (internal inclusive tuple ``(2, 9, 3)``) becomes ``0:8:3``
    (``(0, 7, 3)``): begin 0, SAME stride, subscripts shifted by +2."""
    sdfg = strided_map.to_sdfg(simplify=False)
    changed = NormalizeLoopAndMapOrigin().apply_pass(sdfg, {})
    assert changed is not None
    sdfg.validate()
    (b, e, s), = _map_ranges(sdfg)
    assert str(b) == "0", b
    assert str(e) == "7", e
    assert str(s) == "3", s
    assert str(dace.subsets.Range([(b, e, s)])) == "0:8:3"


def test_map_value_preserving():
    a = np.random.rand(40)
    b_pre = np.full(40, -1.0)
    b_post = b_pre.copy()

    ref = strided_map.to_sdfg(simplify=False)
    ref(A=a.copy(), B=b_pre)

    sdfg = strided_map.to_sdfg(simplify=False)
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    sdfg(A=a.copy(), B=b_post)

    assert np.allclose(b_pre, b_post)


def test_loop_rebased_keeping_stride():
    """``1:N-2`` becomes ``0:N-3`` (begin 0, unit stride kept unit)."""
    sdfg = loop_bounds.to_sdfg(simplify=False)
    changed = NormalizeLoopAndMapOrigin().apply_pass(sdfg, {})
    assert changed is not None
    sdfg.validate()
    loops = _loops(sdfg)
    assert len(loops) == 1
    loop = loops[0]
    assert str(loop_analysis.get_init_assignment(loop)) == "0"
    assert str(loop_analysis.get_loop_stride(loop)) == "1"
    # Inclusive end of ``range(1, N - 2)`` is ``N - 3``; after the rebase (-1)
    # the inclusive end becomes ``N - 4`` -- i.e. the SAME trip count, shown as
    # the python-slice-style exclusive stop ``N - 3``.
    assert str(loop_analysis.get_loop_end(loop)) == "N - 4"


def test_loop_value_preserving():
    n = 30
    a = np.random.rand(n)

    ref = loop_bounds.to_sdfg(simplify=False)
    a_pre = a.copy()
    ref(A=a_pre, N=n)

    sdfg = loop_bounds.to_sdfg(simplify=False)
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    a_post = a.copy()
    sdfg(A=a_post, N=n)

    assert np.allclose(a_pre, a_post)


def test_already_zero_based_is_untouched_and_returns_none():
    sdfg = already_zero_based.to_sdfg(simplify=False)
    ref_hash = sdfg.hash_sdfg()
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) is None
    # No mutation: content hash (``guid``/name-independent) is unchanged.
    assert sdfg.hash_sdfg() == ref_hash


def test_idempotent():
    """A second application on an already-normalized SDFG is a pure no-op
    (``None``) -- required so a ``FixedPointPipeline`` running this pass
    converges instead of spinning forever."""
    sdfg = loop_bounds.to_sdfg(simplify=False)
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) is not None
    ref_hash = sdfg.hash_sdfg()
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == ref_hash


def test_triangular_nest_inner_bound_rewritten_and_value_preserving():
    """``for i in 1:N: for j in i:N`` -- the inner loop's begin is the literal
    token ``i``. Both loops must end up 0-based, with the inner bound
    correctly re-expressed in terms of the (also rebased) outer iterator, and
    the rewrite must be value-preserving (verified by EXECUTION, not by
    inspecting the symbolic form).

    This is the case that the traversal order has to get right in ONE pass: the
    lazy walk rebases the outer loop, which rewrites the inner header to
    ``j = i + 1``, and only then reads the inner begin -- so it sees ``i + 1``,
    not the stale ``i``.
    """
    n = 8
    rng = np.random.default_rng(0)
    a0 = rng.random((20, 20))

    ref = triangular_nest.to_sdfg(simplify=False)
    a_pre = a0.copy()
    ref(A=a_pre, N=n)

    sdfg = triangular_nest.to_sdfg(simplify=False)
    changed = NormalizeLoopAndMapOrigin().apply_pass(sdfg, {})
    assert changed == 2  # both loops, one pass -- no residual, no second round
    sdfg.validate()

    loops = {loop.loop_variable: loop for loop in _loops(sdfg)}
    assert set(loops) == {"i", "j"}
    for loop in loops.values():
        assert str(loop_analysis.get_init_assignment(loop)) == "0", loop.loop_variable
        assert str(loop_analysis.get_loop_stride(loop)) == "1", loop.loop_variable

    # ``i`` ran ``1 .. N-1``, so rebased it runs ``0 .. N-2`` (inclusive end).
    assert str(loop_analysis.get_loop_end(loops["i"])) == "N - 2"
    # ``j`` ran ``i .. N-1`` with the OLD ``i``; with the rebased ``i`` (old ``i+1``)
    # its begin was ``i+1``, so rebased it runs ``0 .. N-i-2`` -- the same trip count,
    # now expressed against the rebased outer iterator.
    assert str(loop_analysis.get_loop_end(loops["j"])) == "N - i - 2"

    a_post = a0.copy()
    sdfg(A=a_post, N=n)
    assert np.allclose(a_pre, a_post)


def test_triangular_nest_idempotent():
    """The nested-begin case converges in one pass: re-applying is a pure no-op."""
    sdfg = triangular_nest.to_sdfg(simplify=False)
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) == 2
    ref_hash = sdfg.hash_sdfg()
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == ref_hash


def test_two_sided_copy_memlet_is_rebased_not_crashed():
    """Regression: a two-sided (``other_subset``) copy memlet used to raise
    ``TODO: Other subset not supported``. Both sides are indexed by the loop
    counter, so BOTH have to be shifted."""
    sdfg = dace.SDFG("two_sided_copy")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    loop = LoopRegion("loop", condition_expr="i < 8", loop_var="i", initialize_expr="i = 2", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_nedge(body.add_read("A"), body.add_write("B"), dace.Memlet(data="A", subset="i", other_subset="i"))
    sdfg.validate()

    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert str(loop_analysis.get_init_assignment(loop)) == "0"
    edge, = body.edges()
    assert str(edge.data.subset) == "i + 2", edge.data.subset
    assert str(edge.data.other_subset) == "i + 2", edge.data.other_subset


def test_while_shaped_loop_is_refused_not_crashed():
    """Regression: a ``LoopRegion`` with no init/update statement (a ``while``)
    used to crash on ``init_statement.as_string``. It carries no counter to
    rebase, so it is left alone -- and, being the only region, the pass reports
    'nothing changed'."""
    sdfg = dace.SDFG("while_shaped")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_symbol("flag", dace.int64)
    loop = LoopRegion("wloop", condition_expr="flag < 8")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("bump", set(), {"out"}, "out = 1.0")
    body.add_edge(tasklet, "out", body.add_write("A"), None, dace.Memlet("A[0]"))
    sdfg.validate()

    ref_hash = sdfg.hash_sdfg()
    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == ref_hash


def test_nested_sdfg_map_is_rebased_after_its_enclosing_map():
    """``scope_children`` restarts at every NestedSDFG, so the walk recurses into
    one as its own root -- from the scope level it sits at, i.e. after every Map
    enclosing it has already been shifted."""
    inner = dace.SDFG("inner")
    inner.add_array("A", [16, 16], dace.float64)
    istate = inner.add_state("ibody", is_start_block=True)
    ime, imx = istate.add_map("inner_map", {"j": "3:9"})
    tasklet = istate.add_tasklet("set", set(), {"y"}, "y = 1.0")
    istate.add_nedge(ime, tasklet, dace.Memlet())
    istate.add_edge(tasklet, "y", imx, "IN_A", dace.Memlet("A[i, j]"))
    imx.add_in_connector("IN_A")
    imx.add_out_connector("OUT_A")
    istate.add_edge(imx, "OUT_A", istate.add_write("A"), None, dace.Memlet("A[i, 3:9]"))

    sdfg = dace.SDFG("outer")
    sdfg.add_array("A", [16, 16], dace.float64)
    state = sdfg.add_state("body", is_start_block=True)
    ome, omx = state.add_map("outer_map", {"i": "2:10"})
    nsdfg = state.add_nested_sdfg(inner, {}, {"A"}, {"i": "i"})
    state.add_nedge(ome, nsdfg, dace.Memlet())
    state.add_edge(nsdfg, "A", omx, "IN_A", dace.Memlet("A[0:16, 0:16]"))
    omx.add_in_connector("IN_A")
    omx.add_out_connector("OUT_A")
    state.add_edge(omx, "OUT_A", state.add_write("A"), None, dace.Memlet("A[0:16, 0:16]"))
    sdfg.validate()

    assert NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}) == 2
    sdfg.validate()
    assert str(ome.map.range) == "0:8", ome.map.range
    assert str(ime.map.range) == "0:6", ime.map.range
    # ``A[i, j]`` under both shifts: the outer ``i`` came from the enclosing map (+2), the
    # inner ``j`` from the map in the nested SDFG (+3).
    write, = [e for e in istate.edges() if e.dst is imx]
    assert str(write.data.subset) == "i + 2, j + 3", write.data.subset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
