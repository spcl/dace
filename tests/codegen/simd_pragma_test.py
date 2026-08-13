# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural tests for the "#pragma omp simd" / "#pragma omp parallel for simd" codegen rules.

Rule 1: an innermost Sequential map gets "#pragma omp simd" on its innermost-dimension loop.
Rule 2: an innermost CPU_Multicore map gets "simd" folded into its "parallel for" pragma.
"Innermost" means the map's body holds no inner Map and no NestedSDFG (non-recursive: a
NestedSDFG disqualifies on sight). An outer map wrapping an inner map never gets "simd",
whatever its own schedule.
"""

import dace
from dace.config import set_temporary
from dace.dtypes import ScheduleType


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


def _nested_sdfg_body_sdfg(outer_schedule: ScheduleType) -> dace.SDFG:
    sdfg = dace.SDFG(f"simd_nsdfg_{outer_schedule.name}")
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_array("B", [10], dace.float64)
    state = sdfg.add_state()
    map_entry, map_exit = state.add_map("outermap", dict(i="0:10"), schedule=outer_schedule)

    nsdfg = dace.SDFG("inner")
    nsdfg.add_array("a_in", [1], dace.float64)
    nsdfg.add_array("b_out", [1], dace.float64)
    nstate = nsdfg.add_state()
    t = nstate.add_tasklet("nt", {"a": None}, {"b": None}, "b = a * 2.0")
    ain = nstate.add_read("a_in")
    bout = nstate.add_write("b_out")
    nstate.add_edge(ain, None, t, "a", dace.Memlet("a_in[0]"))
    nstate.add_edge(t, "b", bout, None, dace.Memlet("b_out[0]"))

    nsdfg_node = state.add_nested_sdfg(nsdfg, {"a_in": None}, {"b_out": None})
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


def test_nested_sdfg_body_blocks_simd():
    # A NestedSDFG in the map body disqualifies it on sight, whatever schedule the outer map has.
    for schedule in (ScheduleType.Sequential, ScheduleType.CPU_Multicore):
        code = _nested_sdfg_body_sdfg(schedule).generate_code()[0].code
        assert "#pragma omp simd" not in code
        assert "#pragma omp parallel for simd" not in code


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
    test_nested_sdfg_body_blocks_simd()
    test_outer_sequential_map_wrapping_inner_map_gets_no_simd_on_outer_loop()
    test_simd_sequential_maps_config_off_switch()
    test_simd_innermost_multicore_maps_config_off_switch()
