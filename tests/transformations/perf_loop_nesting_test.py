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


def _build_velocity_for_it_35_pattern():
    """Build the CFL-clipping kernel's ``_for_it_35`` shape as a standalone SDFG.

    Outer MapEntry (``_for_it_35``) wraps a NestedSDFG that has a single
    state with two sibling inner maps (``_for_it_36``, ``_for_it_37``)
    flanked by AccessNodes. Three arrays (``cfl_clipping``, ``z_w_con_c``,
    ``maxvcfl``) are both read and written by the NSDFG, so the outer
    parent has 6 in-connectors and 4 out-connectors.

    Returns (outer_sdfg, outer_state).
    """
    NPROMA = dace.symbol("NPROMA")
    KLEV = dace.symbol("KLEV")
    NB = dace.symbol("NB")
    DD0 = dace.symbol("DD0")
    DD2 = dace.symbol("DD2")
    LEV_LO = dace.symbol("LEV_LO")
    LEV_HI = dace.symbol("LEV_HI")
    IST = dace.symbol("IST")
    IEN = dace.symbol("IEN")

    # -- Inner NSDFG (single state, two sibling inner maps) -------------
    inner = SDFG("loop_body")
    inner.add_array("cfl_clipping", [NPROMA, KLEV], dace.int32)
    inner.add_array("z_w_con_c", [NPROMA, KLEV], dace.float64)
    inner.add_scalar("cfl_w_limit", dace.float64)
    inner.add_scalar("dtime", dace.float64)
    inner.add_array("maxvcfl", [KLEV, NPROMA], dace.float64)
    inner.add_array("levmask", [NB, KLEV - 1], dace.int32)
    inner.add_array("__CG_p_metrics__m_ddqz_z_half", [DD0, KLEV, DD2], dace.float64,
                    storage=dtypes.StorageType.CPU_Heap)
    inner.add_scalar("tmp_call_7", dace.float64, transient=True)

    ist = inner.add_state("single_state_body", is_start_block=True)

    r_zwcon = ist.add_read("z_w_con_c")
    r_cfllim = ist.add_read("cfl_w_limit")
    r_ddqz_1 = ist.add_read("__CG_p_metrics__m_ddqz_z_half")
    w_cflclip = ist.add_access("cfl_clipping")
    r_cflclip = w_cflclip  # one AN reused: _for_it_36 writes, _for_it_37 reads.
    r_dtime = ist.add_read("dtime")
    r_maxvcfl = ist.add_read("maxvcfl")
    r_zwcon_2 = ist.add_read("z_w_con_c")
    r_ddqz_2 = ist.add_read("__CG_p_metrics__m_ddqz_z_half")
    w_levmask = ist.add_write("levmask")
    w_maxvcfl = ist.add_write("maxvcfl")
    w_zwcon = ist.add_write("z_w_con_c")

    # Inner map 1: _for_it_36 - computes cfl_clipping.
    me36, mx36 = ist.add_map("single_state_body_map", {"_for_it_36": "IST:IEN + 1"})
    for c in ("z_w_con_c", "cfl_w_limit", "ddqz"):
        me36.add_in_connector("IN_" + c); me36.add_out_connector("OUT_" + c)
    mx36.add_in_connector("IN_cfl_clipping"); mx36.add_out_connector("OUT_cfl_clipping")

    ist.add_edge(r_zwcon, None, me36, "IN_z_w_con_c",
                 mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ist.add_edge(r_cfllim, None, me36, "IN_cfl_w_limit", mm.Memlet("cfl_w_limit[0]"))
    ist.add_edge(r_ddqz_1, None, me36, "IN_ddqz",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))

    t36a = ist.add_tasklet("T_abs", {"z_in"}, {"t_out"}, "t_out = abs(z_in)")
    tmp7 = ist.add_access("tmp_call_7")
    t36b = ist.add_tasklet("T_cmp", {"t_in", "lim_in", "ddqz_in"}, {"c_out"},
                           "c_out = 1 if (t_in > (lim_in * ddqz_in)) else 0")

    ist.add_edge(me36, "OUT_z_w_con_c", t36a, "z_in", mm.Memlet("z_w_con_c[0, 0]"))
    ist.add_edge(t36a, "t_out", tmp7, None, mm.Memlet("tmp_call_7[0]"))
    ist.add_edge(tmp7, None, t36b, "t_in", mm.Memlet("tmp_call_7[0]"))
    ist.add_edge(me36, "OUT_cfl_w_limit", t36b, "lim_in", mm.Memlet("cfl_w_limit[0]"))
    ist.add_edge(me36, "OUT_ddqz", t36b, "ddqz_in",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0, 0, 0]"))
    ist.add_edge(t36b, "c_out", mx36, "IN_cfl_clipping", mm.Memlet("cfl_clipping[0, 0]"))
    ist.add_edge(mx36, "OUT_cfl_clipping", w_cflclip, None,
                 mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))

    # Inner map 2: _for_it_37 - consumes cfl_clipping + others, writes levmask/maxvcfl/z_w_con_c.
    me37, mx37 = ist.add_map("single_state_body_0_map", {"_for_it_37": "IST:IEN + 1"})
    for c in ("cfl_clipping", "dtime", "z_w_con_c", "maxvcfl", "ddqz"):
        me37.add_in_connector("IN_" + c); me37.add_out_connector("OUT_" + c)
    for c in ("levmask", "maxvcfl", "z_w_con_c"):
        mx37.add_in_connector("IN_" + c); mx37.add_out_connector("OUT_" + c)

    ist.add_edge(r_cflclip, None, me37, "IN_cfl_clipping",
                 mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))
    ist.add_edge(r_dtime, None, me37, "IN_dtime", mm.Memlet("dtime[0]"))
    ist.add_edge(r_zwcon_2, None, me37, "IN_z_w_con_c",
                 mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ist.add_edge(r_maxvcfl, None, me37, "IN_maxvcfl",
                 mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ist.add_edge(r_ddqz_2, None, me37, "IN_ddqz",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))

    t37 = ist.add_tasklet(
        "T_update",
        {"c_in", "zw_in", "mv_in", "dt_in", "ddqz_in"},
        {"lv_out", "mv_out", "zw_out"},
        "lv_out = c_in\n"
        "mv_out = mv_in + dt_in\n"
        "zw_out = zw_in * ddqz_in",
    )
    ist.add_edge(me37, "OUT_cfl_clipping", t37, "c_in", mm.Memlet("cfl_clipping[0, 0]"))
    ist.add_edge(me37, "OUT_z_w_con_c", t37, "zw_in", mm.Memlet("z_w_con_c[0, 0]"))
    ist.add_edge(me37, "OUT_maxvcfl", t37, "mv_in", mm.Memlet("maxvcfl[0, 0]"))
    ist.add_edge(me37, "OUT_dtime", t37, "dt_in", mm.Memlet("dtime[0]"))
    ist.add_edge(me37, "OUT_ddqz", t37, "ddqz_in",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0, 0, 0]"))

    ist.add_edge(t37, "lv_out", mx37, "IN_levmask", mm.Memlet("levmask[0, 0]"))
    ist.add_edge(t37, "mv_out", mx37, "IN_maxvcfl", mm.Memlet("maxvcfl[0, 0]"))
    ist.add_edge(t37, "zw_out", mx37, "IN_z_w_con_c", mm.Memlet("z_w_con_c[0, 0]"))
    ist.add_edge(mx37, "OUT_levmask", w_levmask, None,
                 mm.Memlet("levmask[0:NB, 0:KLEV - 1]"))
    ist.add_edge(mx37, "OUT_maxvcfl", w_maxvcfl, None,
                 mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ist.add_edge(mx37, "OUT_z_w_con_c", w_zwcon, None,
                 mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))

    # -- Outer SDFG: parent _for_it_35 around the NSDFG ------------------
    outer = SDFG("velocity_for_it_35")
    outer.add_array("cfl_clipping", [NPROMA, KLEV], dace.int32)
    outer.add_array("z_w_con_c", [NPROMA, KLEV], dace.float64)
    outer.add_scalar("cfl_w_limit", dace.float64)
    outer.add_scalar("dtime", dace.float64)
    outer.add_array("maxvcfl", [KLEV, NPROMA], dace.float64)
    outer.add_array("levmask", [NB, KLEV - 1], dace.int32)
    outer.add_array("__CG_p_metrics__m_ddqz_z_half", [DD0, KLEV, DD2], dace.float64,
                    storage=dtypes.StorageType.CPU_Heap)

    ostate = outer.add_state("outer_state", is_start_block=True)

    pe, px = ostate.add_map("single_state_body_4_map", {"_for_it_35": "LEV_LO:LEV_HI"})
    for c in ("cfl_clipping", "z_w_con_c", "cfl_w_limit", "dtime", "maxvcfl", "ddqz"):
        pe.add_in_connector("IN_" + c); pe.add_out_connector("OUT_" + c)
    for c in ("levmask", "maxvcfl", "z_w_con_c", "cfl_clipping"):
        px.add_in_connector("IN_" + c); px.add_out_connector("OUT_" + c)

    nsdfg = ostate.add_nested_sdfg(
        inner,
        inputs={"cfl_clipping", "z_w_con_c", "cfl_w_limit", "dtime", "maxvcfl",
                "__CG_p_metrics__m_ddqz_z_half"},
        outputs={"levmask", "maxvcfl", "z_w_con_c", "cfl_clipping"},
        symbol_mapping={"NPROMA": NPROMA, "KLEV": KLEV, "NB": NB, "DD0": DD0, "DD2": DD2,
                        "IST": 0, "IEN": NPROMA - 1, "_for_it_35": "_for_it_35"},
    )

    r_cflclip_o = ostate.add_read("cfl_clipping")
    r_zwcon_o = ostate.add_read("z_w_con_c")
    r_cfllim_o = ostate.add_read("cfl_w_limit")
    r_dtime_o = ostate.add_read("dtime")
    r_maxvcfl_o = ostate.add_read("maxvcfl")
    r_ddqz_o = ostate.add_read("__CG_p_metrics__m_ddqz_z_half")
    w_levmask_o = ostate.add_write("levmask")
    w_maxvcfl_o = ostate.add_write("maxvcfl")
    w_zwcon_o = ostate.add_write("z_w_con_c")
    w_cflclip_o = ostate.add_write("cfl_clipping")

    ostate.add_edge(r_cflclip_o, None, pe, "IN_cfl_clipping",
                    mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(r_zwcon_o, None, pe, "IN_z_w_con_c",
                    mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(r_cfllim_o, None, pe, "IN_cfl_w_limit", mm.Memlet("cfl_w_limit[0]"))
    ostate.add_edge(r_dtime_o, None, pe, "IN_dtime", mm.Memlet("dtime[0]"))
    ostate.add_edge(r_maxvcfl_o, None, pe, "IN_maxvcfl",
                    mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(r_ddqz_o, None, pe, "IN_ddqz",
                    mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))

    ostate.add_edge(pe, "OUT_cfl_clipping", nsdfg, "cfl_clipping",
                    mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(pe, "OUT_z_w_con_c", nsdfg, "z_w_con_c",
                    mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(pe, "OUT_cfl_w_limit", nsdfg, "cfl_w_limit", mm.Memlet("cfl_w_limit[0]"))
    ostate.add_edge(pe, "OUT_dtime", nsdfg, "dtime", mm.Memlet("dtime[0]"))
    ostate.add_edge(pe, "OUT_maxvcfl", nsdfg, "maxvcfl",
                    mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(pe, "OUT_ddqz", nsdfg, "__CG_p_metrics__m_ddqz_z_half",
                    mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))

    ostate.add_edge(nsdfg, "levmask", px, "IN_levmask",
                    mm.Memlet("levmask[0:NB, 0:KLEV - 1]"))
    ostate.add_edge(nsdfg, "maxvcfl", px, "IN_maxvcfl",
                    mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(nsdfg, "z_w_con_c", px, "IN_z_w_con_c",
                    mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(nsdfg, "cfl_clipping", px, "IN_cfl_clipping",
                    mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))

    ostate.add_edge(px, "OUT_levmask", w_levmask_o, None,
                    mm.Memlet("levmask[0:NB, 0:KLEV - 1]"))
    ostate.add_edge(px, "OUT_maxvcfl", w_maxvcfl_o, None,
                    mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(px, "OUT_z_w_con_c", w_zwcon_o, None,
                    mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(px, "OUT_cfl_clipping", w_cflclip_o, None,
                    mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))

    outer.validate()
    return outer, ostate


def test_velocity_for_it_35_pattern_fissions_into_two_parents():
    """Reconstruction of the CFL-clipping kernel's ``_for_it_35`` shape
    from the ICON velocity-tendencies pipeline. Exercises PLN on a
    realistic match: outer parent with dual read/write on three arrays
    (``cfl_clipping``, ``z_w_con_c``, ``maxvcfl``) and two sibling inner
    maps in the NSDFG's only state.
    """
    outer, ostate = _build_velocity_for_it_35_pattern()

    assert len(_top_level_map_entries(ostate)) == 1
    parent = _top_level_map_entries(ostate)[0]
    assert PerfLoopNesting().can_be_applied_to(outer, parent_entry=parent)

    applied = outer.apply_transformations_repeated(PerfLoopNesting)
    outer.validate()

    assert applied == 1, f"expected 1 PLN application, got {applied}"
    # Two top-level parents afterwards -- one per sibling inner map.
    assert len(_top_level_map_entries(ostate)) == 2


def test_pln_on_parent_inside_nested_sdfg_must_use_owning_sdfg():
    """``can_be_applied_to`` / ``apply_to`` look up the containing state
    via ``sdfg.states()`` on the SDFG argument. If the parent MapEntry
    lives inside a NestedSDFG but the caller passes the *top-level*
    SDFG, the lookup fails with ``StopIteration``.

    This test wraps the velocity pattern inside a NestedSDFG and
    verifies:
      1. passing the top-level SDFG raises ``StopIteration``;
      2. passing the *owning* (nested) SDFG matches and applies cleanly,
         yielding two duplicated parents inside the nested state.
    """
    import pytest

    inner_sdfg, inner_state = _build_velocity_for_it_35_pattern()

    wrapper = SDFG("wrapper")
    for name, desc in inner_sdfg.arrays.items():
        wrapper.add_datadesc(name, desc.clone())
    wstate = wrapper.add_state("wstate", is_start_block=True)
    nested = wstate.add_nested_sdfg(
        inner_sdfg,
        inputs={"cfl_clipping", "z_w_con_c", "cfl_w_limit", "dtime", "maxvcfl",
                "__CG_p_metrics__m_ddqz_z_half"},
        outputs={"levmask", "maxvcfl", "z_w_con_c", "cfl_clipping"},
    )
    for name in ("cfl_clipping", "z_w_con_c", "cfl_w_limit", "dtime", "maxvcfl",
                 "__CG_p_metrics__m_ddqz_z_half"):
        r = wstate.add_read(name)
        wstate.add_edge(r, None, nested, name, mm.Memlet.from_array(name, wrapper.arrays[name]))
    for name in ("levmask", "maxvcfl", "z_w_con_c", "cfl_clipping"):
        w = wstate.add_write(name)
        wstate.add_edge(nested, name, w, None, mm.Memlet.from_array(name, wrapper.arrays[name]))

    wrapper.validate()

    parent = None
    owning_sdfg = None
    for owner in wrapper.all_sdfgs_recursive():
        for st in owner.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.MapEntry) and "_for_it_35" in n.map.params:
                    parent, owning_sdfg = n, owner
    assert parent is not None
    assert owning_sdfg is inner_sdfg, "parent must live in the nested SDFG, not the wrapper"

    # 1. Wrong SDFG -> StopIteration.
    with pytest.raises(StopIteration):
        PerfLoopNesting().can_be_applied_to(wrapper, parent_entry=parent)

    # 2. Owning SDFG -> match + apply.
    assert PerfLoopNesting().can_be_applied_to(owning_sdfg, parent_entry=parent)
    PerfLoopNesting().apply_to(owning_sdfg, parent_entry=parent)
    wrapper.validate()

    top_entries = [n for n in inner_state.nodes()
                   if isinstance(n, nodes.MapEntry) and inner_state.entry_node(n) is None]
    assert len(top_entries) == 2


if __name__ == "__main__":
    test_three_parallel_inner_maps_fission_into_three_parents()
    test_mixed_map_and_tasklet_wraps_tasklet_with_trivial_map()
    test_two_state_nested_sdfg_is_rejected()
    test_velocity_for_it_35_pattern_fissions_into_two_parents()
    test_pln_on_parent_inside_nested_sdfg_must_use_owning_sdfg()
    print("OK")
