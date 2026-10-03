# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for ``assert_connector_role_matches_edges`` — the topology-based
audit helper used by the upcoming branch-normalization passes (M3.1+) to
catch malformed tasklet/nested-SDFG wiring at end-of-pass.

The helper drives correctness from edge direction, not from connector name
suffixes. These tests pin the four invariants:
- in-connector ↔ in-edge match
- out-connector ↔ out-edge match
- no overlap between the two roles
- no orphan declared connector
"""
import dace
import pytest
from dace.sdfg.construction_utils import assert_connector_role_matches_edges


def _scalar_sdfg(name: str):
    sdfg = dace.SDFG(name)
    for n in ("a", "b", "c"):
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    return sdfg


def _add_state_with_tasklet(sdfg: dace.SDFG):
    s = sdfg.add_state("body", is_start_block=True)
    a = s.add_access("a")
    b = s.add_access("b")
    c = s.add_access("c")
    t = s.add_tasklet("add", {"_a", "_b"}, {"_c"}, "_c = _a + _b")
    s.add_edge(a, None, t, "_a", dace.Memlet("a[0]"))
    s.add_edge(b, None, t, "_b", dace.Memlet("b[0]"))
    s.add_edge(t, "_c", c, None, dace.Memlet("c[0]"))
    return s, t


def test_correct_wiring_passes():
    sdfg = _scalar_sdfg("ok")
    state, _ = _add_state_with_tasklet(sdfg)
    assert_connector_role_matches_edges(state)


def test_in_connector_with_no_incoming_edge_is_rejected():
    """Declared in-connector with zero incoming edges is the orphan case."""
    sdfg = _scalar_sdfg("orphan_in")
    state, tasklet = _add_state_with_tasklet(sdfg)
    tasklet.add_in_connector("_d")  # no edge for it
    with pytest.raises(AssertionError, match="_d.*no incoming edge"):
        assert_connector_role_matches_edges(state)


def test_out_connector_with_no_outgoing_edge_is_rejected():
    sdfg = _scalar_sdfg("orphan_out")
    state, tasklet = _add_state_with_tasklet(sdfg)
    tasklet.add_out_connector("_z")
    with pytest.raises(AssertionError, match="_z.*no outgoing edge"):
        assert_connector_role_matches_edges(state)


def test_connector_name_in_both_sets_is_rejected():
    """A single connector name cannot be declared as both input and output."""
    sdfg = _scalar_sdfg("both")
    state, tasklet = _add_state_with_tasklet(sdfg)
    # Force-add an out-connector with the same name as an existing in-connector.
    tasklet.add_out_connector("_a", force=True)
    with pytest.raises(AssertionError, match="both input and output"):
        assert_connector_role_matches_edges(state)


def test_edge_landing_on_out_connector_is_rejected():
    """Simulates the s441_v2 frontend bug: a read edge wired to a write-side
    connector. The helper's job is to catch this."""
    sdfg = dace.SDFG("bad_edge")
    for n in ("a", "out"):
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    src = state.add_access("a")
    dst = state.add_access("out")
    tasklet = state.add_tasklet("t", set(), {"_o"}, "_o = 1.0")
    # Correct write edge:
    state.add_edge(tasklet, "_o", dst, None, dace.Memlet("out[0]"))
    # Buggy read edge that lands on the out-connector:
    state.add_edge(src, None, tasklet, "_o", dace.Memlet("a[0]"))
    with pytest.raises(AssertionError, match="lands on out-connector"):
        assert_connector_role_matches_edges(state)


def test_edge_leaving_in_connector_is_rejected():
    sdfg = dace.SDFG("bad_edge2")
    for n in ("a", "out"):
        sdfg.add_array(n, shape=(1, ), dtype=dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    src = state.add_access("a")
    dst = state.add_access("out")
    tasklet = state.add_tasklet("t", {"_a"}, set(), "pass")
    state.add_edge(src, None, tasklet, "_a", dace.Memlet("a[0]"))
    # Buggy write edge that leaves an in-connector:
    state.add_edge(tasklet, "_a", dst, None, dace.Memlet("out[0]"))
    with pytest.raises(AssertionError, match="leaves in-connector"):
        assert_connector_role_matches_edges(state)


def test_helper_ignores_access_nodes_and_map_nodes():
    """Only Tasklet / NestedSDFG roles are checked; the helper must not trip
    on AccessNodes, MapEntries, or other node kinds that legitimately have
    both in and out connectors with shared bases (``IN_x`` / ``OUT_x``)."""
    sdfg = dace.SDFG("ignored_kinds")
    sdfg.add_array("arr", shape=(8, ), dtype=dace.float64)
    sdfg.add_array("out", shape=(8, ), dtype=dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    an = state.add_access("arr")
    on = state.add_access("out")
    me, mx = state.add_map("m", {"i": "0:8"})
    t = state.add_tasklet("copy", {"_a"}, {"_o"}, "_o = _a")
    state.add_edge(an, None, me, "IN_arr", dace.Memlet.from_array("arr", sdfg.arrays["arr"]))
    me.add_in_connector("IN_arr")
    state.add_edge(me, "OUT_arr", t, "_a", dace.Memlet("arr[i]"))
    me.add_out_connector("OUT_arr")
    state.add_edge(t, "_o", mx, "IN_out", dace.Memlet("out[i]"))
    mx.add_in_connector("IN_out")
    state.add_edge(mx, "OUT_out", on, None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    mx.add_out_connector("OUT_out")
    assert_connector_role_matches_edges(state)


def test_passthrough_connector_on_nested_sdfg_is_validated():
    sdfg = dace.SDFG("nsdfg")
    sdfg.add_array("a", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("c", shape=(1, ), dtype=dace.float64)
    state = sdfg.add_state("s", is_start_block=True)

    inner = dace.SDFG("inner")
    inner.add_array("a", shape=(1, ), dtype=dace.float64)
    inner.add_array("c", shape=(1, ), dtype=dace.float64)
    inner_state = inner.add_state("is", is_start_block=True)
    ia = inner_state.add_access("a")
    ic = inner_state.add_access("c")
    it = inner_state.add_tasklet("t", {"_a"}, {"_c"}, "_c = _a + 1")
    inner_state.add_edge(ia, None, it, "_a", dace.Memlet("a[0]"))
    inner_state.add_edge(it, "_c", ic, None, dace.Memlet("c[0]"))

    n = state.add_nested_sdfg(inner, {"a"}, {"c"})
    ra = state.add_access("a")
    rc = state.add_access("c")
    state.add_edge(ra, None, n, "a", dace.Memlet("a[0]"))
    state.add_edge(n, "c", rc, None, dace.Memlet("c[0]"))

    assert_connector_role_matches_edges(state)
