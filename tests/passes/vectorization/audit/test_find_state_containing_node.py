# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression guards for ``find_state_containing_node`` (formerly
``find_state_of_nsdfg_node``).

Two latent issues fixed:
1. The function used to return the root ``SDFG`` (not the state) despite
   the name and return annotation. Locked-in here: callers receive a
   ``SDFGState`` they can call ``.scope_dict()`` / ``.in_edges()`` on.
2. The function name implied "NSDFG node only" but the only caller
   (``map_predicates``) passes a Tasklet. Locked-in here: works on any
   node type, with a clear exception if the node isn't found or isn't
   in a state-container.
"""
import pytest

import dace
from dace import Memlet
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import find_state_containing_node


def _build_outer_with_nested_tasklet():
    """outer SDFG with a single state containing a nested SDFG whose
    inner state has a tasklet. Returns (outer, outer_state, nsdfg_node,
    inner_state, inner_tasklet)."""
    outer = dace.SDFG("test_find_state_outer")
    outer.add_scalar("acc", dace.float64, transient=False)
    outer_state = outer.add_state()

    inner = dace.SDFG("test_find_state_inner")
    inner.add_scalar("acc", dace.float64, transient=False)
    inner_state = inner.add_state()
    inner_src = inner_state.add_access("acc")
    inner_sink = inner_state.add_access("acc")
    pass_t = inner_state.add_tasklet("inner_pass", {"_i"}, {"_o"}, "_o = _i")
    inner_state.add_edge(inner_src, None, pass_t, "_i", Memlet("acc[0]"))
    inner_state.add_edge(pass_t, "_o", inner_sink, None, Memlet("acc[0]"))

    outer_acc_in = outer_state.add_access("acc")
    nsdfg = outer_state.add_nested_sdfg(inner, {"acc"}, {"acc"})
    outer_state.add_edge(outer_acc_in, None, nsdfg, "acc", Memlet("acc[0]"))
    outer_acc_out = outer_state.add_access("acc")
    outer_state.add_edge(nsdfg, "acc", outer_acc_out, None, Memlet("acc[0]"))

    return outer, outer_state, nsdfg, inner_state, pass_t


def test_returns_state_for_nsdfg_node_not_sdfg():
    """The NSDFG node lives in ``outer_state``; the function must
    return that state, NOT the outer SDFG. Catches the historical
    name-vs-behaviour mismatch where it returned the SDFG."""
    outer, outer_state, nsdfg, _inner_state, _inner_tasklet = _build_outer_with_nested_tasklet()
    result = find_state_containing_node(outer, nsdfg)
    assert isinstance(result, dace.SDFGState), \
        f"Expected SDFGState, got {type(result).__name__} — the 'return state' contract is broken"
    assert result is outer_state, \
        f"Expected the outer state, got a different state ({result.label} vs {outer_state.label})"


def test_returns_state_for_tasklet_inside_nsdfg():
    """The only caller (``map_predicates``) passes a Tasklet, not an
    NSDFG node. The function must walk into nested SDFGs and return the
    INNER state that contains the tasklet — not the outer state."""
    outer, _outer_state, _nsdfg, inner_state, inner_tasklet = _build_outer_with_nested_tasklet()
    result = find_state_containing_node(outer, inner_tasklet)
    assert isinstance(result, dace.SDFGState)
    assert result is inner_state, \
        f"Tasklet lives in {inner_state.label} but function returned {result.label}"
    # Confirm the caller's downstream usage works:
    # scope_dict() is the operation map_predicates.py:386 performs.
    _ = result.scope_dict()


def test_returns_state_for_access_node():
    """Works for plain AccessNodes too — the function is general."""
    outer, outer_state, _nsdfg, _inner_state, _inner_tasklet = _build_outer_with_nested_tasklet()
    # Pick an outer access node and confirm we get the outer state back.
    acc_nodes_outer = [n for n in outer_state.nodes() if isinstance(n, dace.nodes.AccessNode)]
    assert acc_nodes_outer
    result = find_state_containing_node(outer, acc_nodes_outer[0])
    assert result is outer_state


def test_raises_when_node_not_found():
    """A node that isn't anywhere in the SDFG → raises a clear
    exception (not silent None / silent wrong-state return)."""
    outer, _outer_state, _nsdfg, _inner_state, _inner_tasklet = _build_outer_with_nested_tasklet()
    # Build a stray tasklet in a DIFFERENT SDFG.
    stray = dace.SDFG("stray")
    stray_state = stray.add_state()
    stray_tasklet = stray_state.add_tasklet("stray", {}, {"_o"}, "_o = 0")

    with pytest.raises(Exception, match="not found in the root SDFG"):
        find_state_containing_node(outer, stray_tasklet)


def test_caller_pattern_map_predicates_post_rename():
    """End-to-end smoke test of the caller pattern. Mirrors line 385 of
    map_predicates.py: takes a Tasklet inside an NSDFG, walks to the
    inner state, calls ``scope_dict()`` to look up the parent scope.
    If the function ever regresses to returning the SDFG, the scope_dict()
    call would raise AttributeError on the SDFG object."""
    outer, _outer_state, _nsdfg, inner_state, inner_tasklet = _build_outer_with_nested_tasklet()
    parent_state = find_state_containing_node(outer, inner_tasklet)
    # This is the operation the caller does next:
    scope = parent_state.scope_dict()[inner_tasklet]
    # inner_tasklet is at top level of inner_state (no enclosing map), so scope is None.
    assert scope is None
