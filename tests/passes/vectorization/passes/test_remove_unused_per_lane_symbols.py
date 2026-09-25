# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`RemoveUnusedPerLaneSymbols`.

Per user direction 2026-06-10: indirect-access gather lowering emits many
per-lane SDFG symbols as an intermediate stage. After the downstream gather is
materialised, the per-lane symbols may have no remaining use -- this post-clean
pass sweeps them.
"""
import dace

from dace.transformation.passes.vectorization.remove_unused_per_lane_symbols import (RemoveUnusedPerLaneSymbols)
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def _build_sdfg_with_per_lane_symbols(use_in_memlet=False, use_in_iedge_other=False, use_in_tasklet_body=False):
    """Build an SDFG that has 4 per-lane symbols (lane0id_0..3) declared.

    Knobs control whether those symbols are referenced anywhere downstream:
    * ``use_in_memlet``: reference one per-lane symbol from a memlet subset.
    * ``use_in_iedge_other``: reference one from an interstate-edge ELSEWHERE
      (not its own defining edge).
    * ``use_in_tasklet_body``: reference one from a tasklet body.
    """
    sdfg = dace.SDFG("per_lane_sym_fixture")
    for l in range(4):
        sdfg.add_symbol(LaneIdScheme.make_dim("idx", 0, l), dace.int64)
    sdfg.add_symbol("ii", dace.int64)
    sdfg.add_array("A", (16, ), dace.int64, transient=False)
    sdfg.add_array("B", (16, ), dace.int64, transient=False)
    init_state = sdfg.add_state("init", is_start_block=True)
    body_state = sdfg.add_state("body")
    # The "defining" interstate edge -- 4 per-lane symbol assignments.
    sdfg.add_edge(
        init_state, body_state,
        dace.InterstateEdge(assignments={LaneIdScheme.make_dim("idx", 0, l): f"A[ii + {l}]"
                                         for l in range(4)}))
    # Optionally add a downstream USE of one per-lane symbol so it's NOT cleaned.
    if use_in_memlet:
        # Use idx_lane0id_2 in a memlet subset's free symbol.
        used_sym = LaneIdScheme.make_dim("idx", 0, 2)
        a = body_state.add_access("A")
        b = body_state.add_access("B")
        body_state.add_edge(a, None, b, None, dace.Memlet(f"A[{used_sym}]"))
    if use_in_iedge_other:
        used_sym = LaneIdScheme.make_dim("idx", 0, 1)
        sdfg.add_edge(body_state, sdfg.add_state("end"),
                      dace.InterstateEdge(assignments={"some_other_sym": f"{used_sym} + 1"}))
    if use_in_tasklet_body:
        used_sym = LaneIdScheme.make_dim("idx", 0, 3)
        t = body_state.add_tasklet("uses_sym", set(), {"out"}, f"out = {used_sym}", language=dace.dtypes.Language.CPP)
        b = body_state.add_access("B")
        body_state.add_edge(t, "out", b, None, dace.Memlet("B[0]"))
    return sdfg


def test_sweeps_unused_per_lane_symbols():
    """All 4 per-lane symbols are declared but UNUSED downstream -- all should
    be swept, along with their defining interstate-edge assignments."""
    sdfg = _build_sdfg_with_per_lane_symbols()
    # Pre-condition: 4 per-lane symbols declared.
    pre_symbols = {s for s in sdfg.symbols if LaneIdScheme.is_laneid(s)}
    assert len(pre_symbols) == 4

    removed = RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
    assert removed == 4, f"expected 4 removed, got {removed}"

    # Post-condition: no per-lane symbols remain.
    post_symbols = {s for s in sdfg.symbols if LaneIdScheme.is_laneid(s)}
    assert post_symbols == set()
    # The defining interstate-edge assignments are also dropped.
    for edge in sdfg.all_interstate_edges():
        for assigned_sym in edge.data.assignments:
            assert not LaneIdScheme.is_laneid(assigned_sym), \
                f"per-lane defining assignment {assigned_sym!r} should have been dropped"


def test_preserves_symbol_referenced_by_memlet():
    """A per-lane symbol referenced by a memlet subset must be PRESERVED -- the
    pass only sweeps symbols with NO downstream use."""
    sdfg = _build_sdfg_with_per_lane_symbols(use_in_memlet=True)
    used_sym = LaneIdScheme.make_dim("idx", 0, 2)

    removed = RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
    # 3 symbols unused + 1 used = 3 removed, 1 preserved.
    assert removed == 3, f"expected 3 removed, got {removed}"
    assert used_sym in sdfg.symbols
    # The defining assignment for the preserved symbol must also stay.
    found_used_assign = False
    for edge in sdfg.all_interstate_edges():
        if used_sym in edge.data.assignments:
            found_used_assign = True
    assert found_used_assign, f"defining assignment for preserved {used_sym!r} should remain"


def test_preserves_symbol_referenced_by_other_iedge():
    """A per-lane symbol referenced by an interstate edge OTHER than its own
    defining edge must be preserved."""
    sdfg = _build_sdfg_with_per_lane_symbols(use_in_iedge_other=True)
    used_sym = LaneIdScheme.make_dim("idx", 0, 1)

    removed = RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
    assert removed == 3, f"expected 3 removed (3 unused, 1 used by other iedge), got {removed}"
    assert used_sym in sdfg.symbols


def test_preserves_symbol_referenced_in_tasklet_body():
    """A per-lane symbol referenced in a tasklet body must be preserved."""
    sdfg = _build_sdfg_with_per_lane_symbols(use_in_tasklet_body=True)
    used_sym = LaneIdScheme.make_dim("idx", 0, 3)

    removed = RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
    assert removed == 3, f"expected 3 removed (1 used by tasklet body), got {removed}"
    assert used_sym in sdfg.symbols


def test_idempotent():
    """A second invocation after a successful sweep is a no-op."""
    sdfg = _build_sdfg_with_per_lane_symbols()
    first = RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
    assert first == 4
    second = RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
    assert second is None, f"second invocation must be a no-op, got {second}"


def test_returns_none_when_no_lane_symbols_present():
    """An SDFG without any per-lane symbols returns None (idempotent on a clean
    SDFG)."""
    sdfg = dace.SDFG("clean")
    sdfg.add_symbol("ii", dace.int64)
    sdfg.add_array("A", (16, ), dace.int64, transient=False)
    sdfg.add_state("s")
    assert RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {}) is None


def test_recurses_into_nested_sdfg():
    """The sweep recurses into every NestedSDFG and removes unused per-lane
    symbols inside."""
    outer = dace.SDFG("outer")
    outer.add_array("A", (16, ), dace.int64, transient=False)
    outer_state = outer.add_state("s")
    a_outer = outer_state.add_access("A")

    inner = dace.SDFG("inner")
    inner.add_array("A_conn", (16, ), dace.int64)
    inner.add_symbol("ii", dace.int64)
    for l in range(4):
        inner.add_symbol(LaneIdScheme.make_dim("idx", 0, l), dace.int64)
    inner_init = inner.add_state("init", is_start_block=True)
    inner_body = inner.add_state("body")
    inner.add_edge(
        inner_init, inner_body,
        dace.InterstateEdge(assignments={LaneIdScheme.make_dim("idx", 0, l): f"A_conn[ii + {l}]"
                                         for l in range(4)}))

    nsdfg_node = outer_state.add_nested_sdfg(inner, {"A_conn"}, set(), symbol_mapping={"ii": dace.symbol("ii")})
    outer_state.add_edge(a_outer, None, nsdfg_node, "A_conn", dace.Memlet("A[0:16]"))
    outer.add_symbol("ii", dace.int64)

    # 4 inside the inner SDFG, 0 in the outer.
    pre_inner = {s for s in inner.symbols if LaneIdScheme.is_laneid(s)}
    assert len(pre_inner) == 4

    total = RemoveUnusedPerLaneSymbols().apply_pass(outer, {})
    assert total == 4, f"expected 4 symbols swept from inner, got {total}"
    post_inner = {s for s in inner.symbols if LaneIdScheme.is_laneid(s)}
    assert post_inner == set()
