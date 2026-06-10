# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`GatherLift`.

The lift expands a lane-dep gather placeholder (every lane reads the same bare
``__sym``) into a proper per-lane fan-out: each lane reads its OWN per-lane
symbol via :meth:`LaneIdScheme.make_dim`. Tests verify:

* Placeholder bodies matching the K=1 single-loop shape get rewritten.
* The fan-out emits W new SDFG symbols + W new interstate-edge assignments.
* The original lane-dep symbol survives (consumed by the post-clean later).
* Bodies that don't match the placeholder shape are left alone.
"""
import dace

from dace.transformation.passes.vectorization.gather_lift import GatherLift
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def _build_gather_placeholder_sdfg(W=4):
    """Build an SDFG mimicking the walker's post-staging gather placeholder."""
    sdfg = dace.SDFG("gather_lift_fixture")
    sdfg.add_array("idx", (16, ), dace.int64, transient=False)
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("__sym_x", dace.int64)
    state = sdfg.add_state("s", is_start_block=True)
    # Outer Map (innermost tile-tagged).
    me, mx = state.add_map("outer", {"i": "0:16"})
    # Inner NSDFG mimicking the body.
    inner = dace.SDFG("body")
    inner.add_array("idx", (16, ), dace.int64)
    inner.add_array("_idx_tile_0", (W, ), dace.int64, transient=True, storage=dace.dtypes.StorageType.Register)
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("__sym_x", dace.int64)
    init_state = inner.add_state("init", is_start_block=True)
    body_state = inner.add_state("body")
    # The defining iedge: ``__sym_x = idx[i]``.
    inner.add_edge(init_state, body_state, dace.InterstateEdge(assignments={"__sym_x": "idx[i]"}))
    # The placeholder populate tasklet, matching the K=1 shape that
    # ``materialise_per_lane_index_tile`` emits.
    body_lines = [
        f"for (std::size_t __l0 = 0; __l0 < {W}; ++__l0) {{",
        "    _out[__l0] = (int64_t)(__sym_x);",
        "}",
    ]
    placeholder_tasklet = body_state.add_tasklet(
        name="materialise__idx_tile_0",
        inputs=set(),
        outputs={"_out"},
        code="\n".join(body_lines),
        language=dace.dtypes.Language.CPP,
    )
    out_an = body_state.add_access("_idx_tile_0")
    body_state.add_edge(placeholder_tasklet, "_out", out_an, None, dace.Memlet(f"_idx_tile_0[0:{W}]"))
    nsdfg_node = state.add_nested_sdfg(inner, {"idx"}, set(), symbol_mapping={"i": dace.symbol("i")})
    idx_outer = state.add_access("idx")
    state.add_memlet_path(idx_outer, me, nsdfg_node, dst_conn="idx", memlet=dace.Memlet("idx[0:16]"))
    state.add_nedge(nsdfg_node, mx, dace.Memlet())
    return sdfg, inner, placeholder_tasklet


def test_lifts_placeholder_and_emits_per_lane_symbols():
    """The placeholder body ``_out[__l0] = (int64_t)(__sym_x)`` gets rewritten to
    W unrolled writes, each sourcing from a DIFFERENT per-lane symbol."""
    W = 4
    sdfg, inner, tasklet = _build_gather_placeholder_sdfg(W=W)

    removed = GatherLift(widths=(W, )).apply_pass(sdfg, {})
    assert removed == 1, f"expected 1 lift, got {removed}"

    body_after = tasklet.code.as_string
    # The body must contain W writes, each sourcing from ``__sym_x_lane0id_<l>``.
    for lane in range(W):
        sym_name = LaneIdScheme.make_dim("__sym_x", 0, lane)
        assert f"_out[{lane}]" in body_after, f"expected unrolled write to _out[{lane}]"
        assert sym_name in body_after, f"expected source from {sym_name!r} in body, got {body_after!r}"
    # The body must NOT contain the for-loop anymore.
    assert "for (" not in body_after, f"expected unrolled body, got {body_after!r}"


def test_emits_per_lane_iedge_assignments():
    """Per-lane SDFG symbols and per-lane interstate-edge assignments must be
    declared after the lift."""
    W = 4
    sdfg, inner, _ = _build_gather_placeholder_sdfg(W=W)

    GatherLift(widths=(W, )).apply_pass(sdfg, {})

    # W per-lane SDFG symbols.
    for lane in range(W):
        sym_name = LaneIdScheme.make_dim("__sym_x", 0, lane)
        assert sym_name in inner.symbols, f"missing per-lane symbol {sym_name!r}"
    # W per-lane iedge assignments via the iedge that defined ``__sym_x``.
    defining_iedge = None
    for edge in inner.all_interstate_edges():
        if "__sym_x" in edge.data.assignments:
            defining_iedge = edge
            break
    assert defining_iedge is not None
    for lane in range(W):
        sym_name = LaneIdScheme.make_dim("__sym_x", 0, lane)
        assert sym_name in defining_iedge.data.assignments, \
            f"missing per-lane iedge assignment for {sym_name!r}"
        rhs = defining_iedge.data.assignments[sym_name]
        # The RHS must reference (i + <lane>) for lane > 0; sympy simplifies the
        # lane-0 form ``i + 0`` to bare ``i``, so check the non-zero lanes
        # explicitly and the lane-0 expression as the unshifted RHS.
        if lane == 0:
            assert rhs == "idx[i]", f"expected unshifted RHS at lane 0, got {rhs!r}"
        else:
            assert f"i + {lane}" in rhs, f"expected (i + {lane}) substitution in RHS, got {rhs!r}"


def test_original_symbol_survives():
    """The original lane-dep symbol ``__sym_x`` must remain in the iedge
    assignments + SDFG symbols after the lift (post-clean sweeps it later)."""
    W = 4
    sdfg, inner, _ = _build_gather_placeholder_sdfg(W=W)

    GatherLift(widths=(W, )).apply_pass(sdfg, {})

    assert "__sym_x" in inner.symbols
    defining_iedge = next(
        (edge for edge in inner.all_interstate_edges() if "__sym_x" in edge.data.assignments),
        None,
    )
    assert defining_iedge is not None
    assert defining_iedge.data.assignments["__sym_x"] == "idx[i]", \
        "original iedge RHS must stay intact"


def test_non_placeholder_body_left_alone():
    """A tasklet whose body doesn't match the placeholder shape must be skipped
    (returns ``None`` from apply_pass when nothing to lift)."""
    sdfg = dace.SDFG("non_placeholder_fixture")
    sdfg.add_array("idx", (16, ), dace.int64, transient=False)
    sdfg.add_symbol("i", dace.int64)
    state = sdfg.add_state("s", is_start_block=True)
    me, mx = state.add_map("outer", {"i": "0:16"})
    inner = dace.SDFG("body")
    inner.add_array("idx", (16, ), dace.int64)
    inner.add_array("_idx_tile_0", (4, ), dace.int64, transient=True, storage=dace.dtypes.StorageType.Register)
    inner.add_symbol("i", dace.int64)
    body_state = inner.add_state("body", is_start_block=True)
    # A body that doesn't match the placeholder shape -- a comment + unrelated write.
    tasklet = body_state.add_tasklet(
        name="not_a_placeholder",
        inputs=set(),
        outputs={"_out"},
        code="_out[0] = 42;",
        language=dace.dtypes.Language.CPP,
    )
    out_an = body_state.add_access("_idx_tile_0")
    body_state.add_edge(tasklet, "_out", out_an, None, dace.Memlet("_idx_tile_0[0:4]"))
    nsdfg_node = state.add_nested_sdfg(inner, {"idx"}, set(), symbol_mapping={"i": dace.symbol("i")})
    idx_outer = state.add_access("idx")
    state.add_memlet_path(idx_outer, me, nsdfg_node, dst_conn="idx", memlet=dace.Memlet("idx[0:16]"))
    state.add_nedge(nsdfg_node, mx, dace.Memlet())

    assert GatherLift(widths=(4, )).apply_pass(sdfg, {}) is None
