# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural tests for the "#pragma omp simd" / "#pragma omp parallel for simd" codegen rules.

Rule 1: an innermost Sequential map gets "#pragma omp simd" on its innermost-dimension loop.
Rule 2: an innermost CPU_Multicore map gets "simd" folded into its "parallel for" pragma.
"Innermost" means the map's body holds no Map and no loop at ANY depth. A NestedSDFG is opened
and cleared when it transitively holds neither; it disqualifies only when it hides a Map, a
LoopRegion, or a back edge in a plain state machine. An outer map wrapping an inner map never
gets "simd", whatever its own schedule.
"""

import dace
from dace.config import set_temporary
from dace.dtypes import ScheduleType
from dace.sdfg.state import LoopRegion


def _leaf_map_sdfg(schedule: ScheduleType) -> dace.SDFG:
    sdfg = dace.SDFG(f"simd_leaf_{schedule.name}")
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_array("B", [10], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "leafmap",
        dict(i="0:10"),
        dict(a=dace.Memlet("A[i]")),
        "b = a * 2.0",
        dict(b=dace.Memlet("B[i]")),
        schedule=schedule,
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def _nested_maps_sdfg(outer_schedule: ScheduleType, inner_schedule: ScheduleType) -> dace.SDFG:
    sdfg = dace.SDFG(f"simd_nested_{outer_schedule.name}_{inner_schedule.name}")
    sdfg.add_array("A", [10, 10], dace.float64)
    sdfg.add_array("B", [10, 10], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("outer", dict(i="0:10"), schedule=outer_schedule)
    inner_entry, inner_exit = state.add_map("inner", dict(j="0:10"), schedule=inner_schedule)
    tasklet = state.add_tasklet("t", {"a": None}, {"b": None}, "b = a * 2.0")
    A = state.add_read("A")
    B = state.add_write("B")
    state.add_memlet_path(A, outer_entry, inner_entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i, j]"))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, B, src_conn="b", memlet=dace.Memlet("B[i, j]"))
    sdfg.validate()
    return sdfg


def _double_tasklet(state) -> None:
    """``b_out[0] = a_in[0] * 2`` as plain dataflow in ``state``."""
    t = state.add_tasklet("nt", {"a": None}, {"b": None}, "b = a * 2.0")
    state.add_edge(state.add_read("a_in"), None, t, "a", dace.Memlet("a_in[0]"))
    state.add_edge(t, "b", state.add_write("b_out"), None, dace.Memlet("b_out[0]"))


def _inner_sdfg(payload: str) -> dace.SDFG:
    nsdfg = dace.SDFG(f"inner_{payload}")
    nsdfg.add_array("a_in", [1], dace.float64)
    nsdfg.add_array("b_out", [1], dace.float64)
    if payload == "loopregion":
        loop = LoopRegion("nloop", "k < 1", "k", "k = 0", "k = k + 1")
        nsdfg.add_node(loop, is_start_block=True)
        _double_tasklet(loop.add_state(is_start_block=True))
    elif payload == "backedge":
        s0 = nsdfg.add_state(is_start_block=True)
        s1 = nsdfg.add_state()
        _double_tasklet(s1)
        # Structurally a back edge, never taken at run time: the point is the cycle, not the trip.
        nsdfg.add_edge(s0, s1, dace.InterstateEdge(condition="1 > 0"))
        nsdfg.add_edge(s1, s0, dace.InterstateEdge(condition="0 > 1"))
    elif payload == "map":
        nstate = nsdfg.add_state()
        nstate.add_mapped_tasklet("innermap",
                                  dict(k="0:1"),
                                  dict(a=dace.Memlet("a_in[0]")),
                                  "b = a * 2.0",
                                  dict(b=dace.Memlet("b_out[0]")),
                                  external_edges=True)
    elif payload == "nested":
        nstate = nsdfg.add_state()
        deeper = _inner_sdfg("straight")
        deep_node = nstate.add_nested_sdfg(deeper, {"a_in": None}, {"b_out": None})
        nstate.add_edge(nstate.add_read("a_in"), None, deep_node, "a_in", dace.Memlet("a_in[0]"))
        nstate.add_edge(deep_node, "b_out", nstate.add_write("b_out"), None, dace.Memlet("b_out[0]"))
    else:
        _double_tasklet(nsdfg.add_state())
    return nsdfg


def _nested_sdfg_body_sdfg(outer_schedule: ScheduleType, payload: str = "straight") -> dace.SDFG:
    sdfg = dace.SDFG(f"simd_nsdfg_{payload}_{outer_schedule.name}")
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_array("B", [10], dace.float64)
    state = sdfg.add_state()
    map_entry, map_exit = state.add_map("outermap", dict(i="0:10"), schedule=outer_schedule)

    nsdfg_node = state.add_nested_sdfg(_inner_sdfg(payload), {"a_in": None}, {"b_out": None})
    A = state.add_read("A")
    B = state.add_write("B")
    state.add_memlet_path(A, map_entry, nsdfg_node, dst_conn="a_in", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(nsdfg_node, map_exit, B, src_conn="b_out", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def test_innermost_sequential_map_gets_omp_simd():
    code = _leaf_map_sdfg(ScheduleType.Sequential).generate_code()[0].code
    assert "#pragma omp simd" in code
    assert "#pragma omp parallel for simd" not in code


def test_innermost_cpu_multicore_map_gets_parallel_for_simd():
    code = _leaf_map_sdfg(ScheduleType.CPU_Multicore).generate_code()[0].code
    assert "#pragma omp parallel for simd" in code


def test_cpu_multicore_map_with_inner_map_has_plain_parallel_for():
    # Outer CPU_Multicore wraps an inner map: it is not a leaf, so its own pragma stays a plain
    # "parallel for" with no simd clause folded in.
    code = _nested_maps_sdfg(ScheduleType.CPU_Multicore, ScheduleType.Default).generate_code()[0].code
    assert "#pragma omp parallel for simd" not in code
    assert "#pragma omp parallel for" in code


def test_straight_line_nested_sdfg_body_allows_simd():
    # A NestedSDFG holding only straight-line dataflow is opened and cleared: the outlined
    # "loop_body" the Fortran frontend emits is exactly this shape.
    code = _nested_sdfg_body_sdfg(ScheduleType.Sequential).generate_code()[0].code
    assert "#pragma omp simd" in code
    code = _nested_sdfg_body_sdfg(ScheduleType.CPU_Multicore).generate_code()[0].code
    assert "#pragma omp parallel for simd" in code


def test_nested_sdfg_holding_a_loopregion_blocks_simd():
    for schedule in (ScheduleType.Sequential, ScheduleType.CPU_Multicore):
        code = _nested_sdfg_body_sdfg(schedule, payload="loopregion").generate_code()[0].code
        assert "#pragma omp simd" not in code
        assert "#pragma omp parallel for simd" not in code


def test_nested_sdfg_holding_a_back_edge_blocks_simd():
    # A loop need not be a LoopRegion: a back edge in a plain state machine is one too.
    for schedule in (ScheduleType.Sequential, ScheduleType.CPU_Multicore):
        code = _nested_sdfg_body_sdfg(schedule, payload="backedge").generate_code()[0].code
        assert "#pragma omp simd" not in code
        assert "#pragma omp parallel for simd" not in code


def test_nested_sdfg_holding_a_map_blocks_simd():
    # The inner map is itself a leaf and keeps its own simd, so check the OUTER loop's pragma only.
    for schedule in (ScheduleType.Sequential, ScheduleType.CPU_Multicore):
        lines = _nested_sdfg_body_sdfg(schedule, payload="map").generate_code()[0].code.splitlines()
        outer_idx = next(i for i, l in enumerate(lines) if "for (auto i = 0" in l)
        assert "simd" not in lines[outer_idx - 1]


def test_doubly_nested_straight_line_sdfg_allows_simd():
    code = _nested_sdfg_body_sdfg(ScheduleType.Sequential, payload="nested").generate_code()[0].code
    assert "#pragma omp simd" in code


def test_outer_sequential_map_wrapping_inner_map_gets_no_simd_on_outer_loop():
    code = _nested_maps_sdfg(ScheduleType.Sequential, ScheduleType.Sequential).generate_code()[0].code
    lines = code.splitlines()
    outer_idx = next(i for i, l in enumerate(lines) if "for (auto i = 0" in l)
    inner_idx = next(i for i, l in enumerate(lines) if "for (auto j = 0" in l)
    assert "simd" not in lines[outer_idx - 1]
    assert "#pragma omp simd" in lines[inner_idx - 1]


def test_simd_sequential_maps_config_off_switch():
    with set_temporary('compiler', 'cpu', 'simd_sequential_maps', value=False):
        code = _leaf_map_sdfg(ScheduleType.Sequential).generate_code()[0].code
    assert "#pragma omp simd" not in code


def test_simd_innermost_multicore_maps_config_off_switch():
    with set_temporary('compiler', 'cpu', 'simd_innermost_multicore_maps', value=False):
        code = _leaf_map_sdfg(ScheduleType.CPU_Multicore).generate_code()[0].code
    assert "#pragma omp parallel for simd" not in code
    assert "#pragma omp parallel for" in code


if __name__ == "__main__":
    test_innermost_sequential_map_gets_omp_simd()
    test_innermost_cpu_multicore_map_gets_parallel_for_simd()
    test_cpu_multicore_map_with_inner_map_has_plain_parallel_for()
    test_straight_line_nested_sdfg_body_allows_simd()
    test_nested_sdfg_holding_a_loopregion_blocks_simd()
    test_nested_sdfg_holding_a_back_edge_blocks_simd()
    test_nested_sdfg_holding_a_map_blocks_simd()
    test_doubly_nested_straight_line_sdfg_allows_simd()
    test_outer_sequential_map_wrapping_inner_map_gets_no_simd_on_outer_loop()
    test_simd_sequential_maps_config_off_switch()
    test_simd_innermost_multicore_maps_config_off_switch()
