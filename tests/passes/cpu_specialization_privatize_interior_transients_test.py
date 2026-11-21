# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``privatize_interior_transients`` renames a shared transient along every edge of its memlet paths, not only the
edges next to its access nodes."""

import dace
from dace.transformation.passes.cpu_specialization.decide_cpu_parallel_axes import privatize_interior_transients


def copy_into_an_inner_map() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    """``map i: s = A[i]; map j: t[j] = s; B[i, j] = t[j]`` with the copy ``s -> t`` named by ``t`` through the inner map's entry,
    and ``t`` also used in a later state, so the outer map's scope shares it."""
    sdfg = dace.SDFG("privatize_interior_copy_path")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [4, 3], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    sdfg.add_array("t", [3], dace.float64, transient=True)
    state = sdfg.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    outer_entry, outer_exit = state.add_map("outer", {"i": "0:4"})
    inner_entry, inner_exit = state.add_map("inner", {"j": "0:3"})
    load = state.add_tasklet("load", {"x"}, {"y"}, "y = x")
    s = state.add_access("s")
    t = state.add_access("t")
    state.add_memlet_path(a, outer_entry, load, dst_conn="x", memlet=dace.Memlet("A[i]"))
    state.add_edge(load, "y", s, None, dace.Memlet("s[0]"))
    # The copy is named by its destination along its whole path: ``t[j]``, with ``s[0]`` as the other subset.
    state.add_memlet_path(s, inner_entry, t, memlet=dace.Memlet(data="t", subset="j", other_subset="0"))
    state.add_memlet_path(t, inner_exit, outer_exit, b, memlet=dace.Memlet(data="B", subset="i, j", other_subset="j"))
    later = sdfg.add_state_after(state)
    later.add_nedge(later.add_read("A"), later.add_write("t"), dace.Memlet("A[0:3]"))
    sdfg.validate()
    return sdfg, state, outer_entry


def test_a_copy_through_an_inner_map_entry_is_renamed_along_its_path():
    sdfg, state, outer_entry = copy_into_an_inner_map()
    assert privatize_interior_transients(sdfg, state, outer_entry)
    sdfg.validate()
    inner_names = {n.data for n in state.data_nodes() if state.entry_node(n) is not None}
    assert "t" not in inner_names and "s" in inner_names
