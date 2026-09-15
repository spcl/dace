# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Two copies of one value into different elements of an array are two writes, and both must survive.

``BypassTrivialAssignTasklets`` collapses duplicate ``AN -> [_out = _in] -> AN`` copies. Keyed on container names
alone, ``c[0, i] = z`` and ``c[1, i] = z`` looked like one copy and the second write was dropped: CloudSC's
``zconvsink[ncldql] = zmfdn; zconvsink[ncldqi] = zmfdn`` lost the ice entry.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import copy

import numpy as np

import dace
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets

LENGTH = 16


def one_value_copied_into_two_elements() -> dace.SDFG:
    body = dace.SDFG("copy_into_two_elements_body")
    body.add_symbol("i", dace.int64)
    body.add_array("X", (LENGTH, ), dace.float64)
    body.add_array("C", (2, LENGTH), dace.float64)
    body.add_scalar("Z", dace.float64, transient=True)
    state = body.add_state("convective_sink")
    value = state.add_access("Z")
    clamp = state.add_tasklet("clamp", {"_in": None}, {"_out": None}, "_out = max(0.0, _in)")
    state.add_edge(state.add_access("X"), None, clamp, "_in", Memlet("X[i]"))
    state.add_edge(clamp, "_out", value, None, Memlet("Z"))
    for element in (0, 1):
        store = state.add_tasklet(f"store_{element}", {"_in": None}, {"_out": None}, "_out = _in")
        state.add_edge(value, None, store, "_in", Memlet("Z"))
        state.add_edge(store, "_out", state.add_access("C"), None, Memlet(f"C[{element}, i]"))

    sdfg = dace.SDFG("bypass_dedup_distinct_elements")
    sdfg.add_array("X", (LENGTH, ), dace.float64)
    sdfg.add_array("C", (2, LENGTH), dace.float64)
    outer = sdfg.add_state("outer")
    map_entry, map_exit = outer.add_map("per_column", {"i": f"0:{LENGTH}"}, schedule=dace.ScheduleType.CPU_Multicore)
    nested = outer.add_nested_sdfg(body, {"X": None}, {"C": None}, symbol_mapping={"i": "i"})
    map_entry.add_in_connector("IN_X")
    map_entry.add_out_connector("OUT_X")
    map_exit.add_in_connector("IN_C")
    map_exit.add_out_connector("OUT_C")
    outer.add_edge(outer.add_read("X"), None, map_entry, "IN_X", Memlet(f"X[0:{LENGTH}]"))
    outer.add_edge(map_entry, "OUT_X", nested, "X", Memlet(f"X[0:{LENGTH}]"))
    outer.add_edge(nested, "C", map_exit, "IN_C", Memlet(f"C[0:2, 0:{LENGTH}]"))
    outer.add_edge(map_exit, "OUT_C", outer.add_write("C"), None, Memlet(f"C[0:2, 0:{LENGTH}]"))
    sdfg.validate()
    return sdfg


def written_elements(body: dace.SDFG) -> list[str]:
    return sorted(
        str(edge.data.subset) for state in body.states() for node in state.data_nodes() if node.data == "C"
        for edge in state.in_edges(node) if isinstance(edge.src, nodes.Tasklet))


def test_copies_of_one_value_into_two_elements_both_stay_written():
    sdfg = one_value_copied_into_two_elements()
    body = next(nsdfg for nsdfg in sdfg.all_sdfgs_recursive() if nsdfg is not sdfg)

    BypassTrivialAssignTasklets().apply_pass(sdfg, {})

    sdfg.validate()
    assert written_elements(body) == ["0, i", "1, i"]
    rng = np.random.default_rng(9)
    arguments = {"X": rng.uniform(-1.0, 1.0, LENGTH), "C": np.full((2, LENGTH), -7.0)}
    got = copy.deepcopy(arguments)
    sdfg.compile()(**got)
    expected = np.maximum(0.0, arguments["X"])
    np.testing.assert_array_equal(got["C"][0], expected)
    np.testing.assert_array_equal(got["C"][1], expected)
