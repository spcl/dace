"""Tests for ``PerfLoopNesting``.

Each case builds a candidate SDFG (via the Python frontend where
possible) and checks whether the transformation fissions the parent map
into K copies, where K equals the number of top-level children in the
inner state's dataflow. Each positive case is compiled and run against
a NumPy reference to verify the transform is semantically sound.
"""
import numpy as np
import dace
from dace import dtypes, memlet as mm, nodes
from dace.sdfg import SDFG, SDFGState
from dace.transformation.interstate import LoopToMap, StateFusion

from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting


def _force_sequential_maps(sdfg: dace.SDFG):
    """Avoid OpenMP codegen so the test binary loads without ``libgomp``."""
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            n.map.schedule = dtypes.ScheduleType.Sequential


M = dace.symbol("M")
N = dace.symbol("N")


def _outer_state(sdfg: dace.SDFG) -> SDFGState:
    for s in sdfg.all_states():
        if any(isinstance(n, nodes.MapEntry) and s.entry_node(n) is None for n in s.nodes()):
            return s
    return sdfg.start_state


def _top_level_map_entries(state: SDFGState):
    return [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]


def _prepare_nested_single_state(sdfg: dace.SDFG):
    """Lower to the shape PerfLoopNesting matches: each inner LoopRegion
    becomes a Map, and all sibling inner states are fused into one."""
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(StateFusion)


def test_three_parallel_inner_maps_fission_into_three_parents():
    """
    for j:
        for i: x[i,j] = 1.0
        for i: y[i,j] = 2.0
        for i: z[i,j] = 3.0
    """

    @dace.program
    def kernel(x: dace.float64[N, M], y: dace.float64[N, M], z: dace.float64[N, M]):
        for j in dace.map[0:M]:
            for i in range(N):
                x[i, j] = 1.0
            for i in range(N):
                y[i, j] = 2.0
            for i in range(N):
                z[i, j] = 3.0

    sdfg: dace.SDFG = kernel.to_sdfg(simplify=False)
    _prepare_nested_single_state(sdfg)

    state = _outer_state(sdfg)
    assert len(_top_level_map_entries(state)) == 1

    applied = sdfg.apply_transformations_repeated(PerfLoopNesting)
    sdfg.validate()

    assert applied == 1
    assert len(_top_level_map_entries(state)) == 3

    # Numerical verification.
    _force_sequential_maps(sdfg)
    m_val, n_val = 7, 5
    x = np.zeros((n_val, m_val), dtype=np.float64)
    y = np.zeros((n_val, m_val), dtype=np.float64)
    z = np.zeros((n_val, m_val), dtype=np.float64)
    sdfg(x=x, y=y, z=z, M=m_val, N=n_val)
    assert np.allclose(x, 1.0)
    assert np.allclose(y, 2.0)
    assert np.allclose(z, 3.0)


def test_mixed_map_and_tasklet_wraps_tasklet_with_trivial_map():
    """
    for j:
        for i: x[i,j] = 1.0
        z[j] = 3.0                  # top-level tasklet in nested state
        for i: y[i,j] = 2.0
    """

    @dace.program
    def kernel(x: dace.float64[N, M], y: dace.float64[N, M], z: dace.float64[M]):
        for j in dace.map[0:M]:
            for i in range(N):
                x[i, j] = 1.0
            z[j] = 3.0
            for i in range(N):
                y[i, j] = 2.0

    sdfg: dace.SDFG = kernel.to_sdfg(simplify=False)
    _prepare_nested_single_state(sdfg)

    state = _outer_state(sdfg)
    assert len(_top_level_map_entries(state)) == 1

    applied = sdfg.apply_transformations_repeated(PerfLoopNesting)
    sdfg.validate()

    assert applied == 1
    # Three duplicated parents: two wrap an inner i-map, one wraps
    # the tasklet-in-trivial-map.
    assert len(_top_level_map_entries(state)) == 3

    _force_sequential_maps(sdfg)
    m_val, n_val = 6, 4
    x = np.zeros((n_val, m_val), dtype=np.float64)
    y = np.zeros((n_val, m_val), dtype=np.float64)
    z = np.zeros(m_val, dtype=np.float64)
    sdfg(x=x, y=y, z=z, M=m_val, N=n_val)
    assert np.allclose(x, 1.0)
    assert np.allclose(y, 2.0)
    assert np.allclose(z, 3.0)


def test_two_state_nested_sdfg_is_rejected():
    """When the NestedSDFG inside the parent map has 2 states, the
    transformation must not apply."""

    inner = SDFG("two_state_inner")
    inner.add_array("a_in", [N], dace.float64)
    inner.add_array("a_out", [N], dace.float64)
    inner.add_array("b_in", [N], dace.float64)
    inner.add_array("b_out", [N], dace.float64)

    s1 = inner.add_state("s1", is_start_block=True)
    s2 = inner.add_state("s2")
    inner.add_edge(s1, s2, dace.InterstateEdge())

    r1 = s1.add_read("a_in")
    w1 = s1.add_write("a_out")
    me1, mx1 = s1.add_map("m1", {"i": "0:N"})
    me1.add_in_connector("IN_a")
    me1.add_out_connector("OUT_a")
    mx1.add_in_connector("IN_a")
    mx1.add_out_connector("OUT_a")
    t1 = s1.add_tasklet("t1", {"x"}, {"y"}, "y = x + 1")
    s1.add_edge(r1, None, me1, "IN_a", mm.Memlet("a_in[0:N]"))
    s1.add_edge(me1, "OUT_a", t1, "x", mm.Memlet("a_in[i]"))
    s1.add_edge(t1, "y", mx1, "IN_a", mm.Memlet("a_out[i]"))
    s1.add_edge(mx1, "OUT_a", w1, None, mm.Memlet("a_out[0:N]"))

    r2 = s2.add_read("b_in")
    w2 = s2.add_write("b_out")
    me2, mx2 = s2.add_map("m2", {"i": "0:N"})
    me2.add_in_connector("IN_b")
    me2.add_out_connector("OUT_b")
    mx2.add_in_connector("IN_b")
    mx2.add_out_connector("OUT_b")
    t2 = s2.add_tasklet("t2", {"x"}, {"y"}, "y = x * 2")
    s2.add_edge(r2, None, me2, "IN_b", mm.Memlet("b_in[0:N]"))
    s2.add_edge(me2, "OUT_b", t2, "x", mm.Memlet("b_in[i]"))
    s2.add_edge(t2, "y", mx2, "IN_b", mm.Memlet("b_out[i]"))
    s2.add_edge(mx2, "OUT_b", w2, None, mm.Memlet("b_out[0:N]"))

    outer = SDFG("outer")
    for name in ("A_in", "A_out", "B_in", "B_out"):
        outer.add_array(name, [M, N], dace.float64)
    ostate = outer.add_state("ostate", is_start_block=True)
    Ar = ostate.add_read("A_in")
    Br = ostate.add_read("B_in")
    Aw = ostate.add_write("A_out")
    Bw = ostate.add_write("B_out")

    pe, px = ostate.add_map("parent", {"j": "0:M"})
    for c in ("a_in", "b_in"):
        pe.add_in_connector("IN_" + c)
        pe.add_out_connector("OUT_" + c)
    for c in ("a_out", "b_out"):
        px.add_in_connector("IN_" + c)
        px.add_out_connector("OUT_" + c)

    ns = ostate.add_nested_sdfg(inner, {"a_in", "b_in"}, {"a_out", "b_out"})
    ostate.add_edge(Ar, None, pe, "IN_a_in", mm.Memlet("A_in[0:M, 0:N]"))
    ostate.add_edge(Br, None, pe, "IN_b_in", mm.Memlet("B_in[0:M, 0:N]"))
    ostate.add_edge(pe, "OUT_a_in", ns, "a_in", mm.Memlet("A_in[j, 0:N]"))
    ostate.add_edge(pe, "OUT_b_in", ns, "b_in", mm.Memlet("B_in[j, 0:N]"))
    ostate.add_edge(ns, "a_out", px, "IN_a_out", mm.Memlet("A_out[j, 0:N]"))
    ostate.add_edge(ns, "b_out", px, "IN_b_out", mm.Memlet("B_out[j, 0:N]"))
    ostate.add_edge(px, "OUT_a_out", Aw, None, mm.Memlet("A_out[0:M, 0:N]"))
    ostate.add_edge(px, "OUT_b_out", Bw, None, mm.Memlet("B_out[0:M, 0:N]"))
    outer.validate()

    applied = outer.apply_transformations_repeated(PerfLoopNesting)
    assert applied == 0
    # The outer MapEntry must still be there, singular.
    assert len(_top_level_map_entries(ostate)) == 1


if __name__ == "__main__":
    test_three_parallel_inner_maps_fission_into_three_parents()
    test_mixed_map_and_tasklet_wraps_tasklet_with_trivial_map()
    test_two_state_nested_sdfg_is_rejected()
    print("OK")
