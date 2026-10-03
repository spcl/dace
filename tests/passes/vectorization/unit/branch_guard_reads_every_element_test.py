# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lifted branch guard reads every element it names, not one element per array.

``BranchNormalization`` lifts a guard such as ``(A[0, i] + A[1, i] < 0.5) or (B[i] < 0.25)`` into a tasklet whose
inputs are the array reads. One input per array name folded ``A[0, i] + A[1, i]`` into ``2 * A[0, i]``: CloudSC's
tidy-up guard ``zqx[ncldql] + zqx[ncldqi] < rlmin or za < ramin`` then fired on the wrong columns.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import copy

import numpy as np

import dace
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization

LENGTH = 16


def guard_summing_two_elements() -> dace.SDFG:
    body = dace.SDFG("guard_summing_two_elements_body")
    body.add_symbol("i", dace.int64)
    body.add_array("A", (2, LENGTH), dace.float64)
    body.add_array("B", (LENGTH, ), dace.float64)
    body.add_array("OUT", (LENGTH, ), dace.float64)
    stage = body.add_state("stage", is_start_block=True)
    guarded = ConditionalBlock("tidy_up")
    body.add_node(guarded)
    body.add_edge(stage, guarded, dace.InterstateEdge(assignments={"s": "A[0, i] + A[1, i]", "t": "B[i]"}))
    arm = ControlFlowRegion("tidy_up_arm", sdfg=body)
    write = arm.add_state("write", is_start_block=True)
    zero = write.add_tasklet("zero", {}, {"_out": None}, "_out = 0.0")
    write.add_edge(zero, "_out", write.add_access("OUT"), None, Memlet("OUT[i]"))
    guarded.add_branch(CodeBlock("(s < 0.5) or (t < 0.25)"), arm)
    body.add_edge(guarded, body.add_state("after"), dace.InterstateEdge())

    sdfg = dace.SDFG("branch_guard_reads_every_element")
    sdfg.add_array("A", (2, LENGTH), dace.float64)
    sdfg.add_array("B", (LENGTH, ), dace.float64)
    sdfg.add_array("OUT", (LENGTH, ), dace.float64)
    outer = sdfg.add_state("outer")
    map_entry, map_exit = outer.add_map("per_column", {"i": f"0:{LENGTH}"}, schedule=dace.ScheduleType.CPU_Multicore)
    nested = outer.add_nested_sdfg(body, {"A": None, "B": None}, {"OUT": None}, symbol_mapping={"i": "i"})
    for name, extent in (("A", f"0:2, 0:{LENGTH}"), ("B", f"0:{LENGTH}")):
        map_entry.add_in_connector(f"IN_{name}")
        map_entry.add_out_connector(f"OUT_{name}")
        outer.add_edge(outer.add_read(name), None, map_entry, f"IN_{name}", Memlet(f"{name}[{extent}]"))
        outer.add_edge(map_entry, f"OUT_{name}", nested, name, Memlet(f"{name}[{extent}]"))
    map_exit.add_in_connector("IN_OUT")
    map_exit.add_out_connector("OUT_OUT")
    outer.add_edge(nested, "OUT", map_exit, "IN_OUT", Memlet(f"OUT[0:{LENGTH}]"))
    outer.add_edge(map_exit, "OUT_OUT", outer.add_write("OUT"), None, Memlet(f"OUT[0:{LENGTH}]"))
    sdfg.validate()
    return sdfg


def lifted_guard_reads(sdfg: dace.SDFG) -> list[str]:
    return sorted(
        str(edge.data) for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet) and node.label.startswith("lift_cond") for edge in state.in_edges(node))


def test_guard_summing_two_elements_of_one_array_reads_both():
    sdfg = guard_summing_two_elements()

    BranchNormalization().apply_pass(sdfg, {})

    sdfg.validate()
    assert lifted_guard_reads(sdfg) == ["A[0, i]", "A[1, i]", "B[i]"]
    a = np.empty((2, LENGTH))
    a[0] = np.linspace(0.05, 0.35, LENGTH)
    a[1] = 0.3
    b = np.full(LENGTH, 0.9)
    arguments = {"A": a, "B": b, "OUT": np.full(LENGTH, 4.0)}
    got = copy.deepcopy(arguments)
    sdfg.compile()(**got)
    expected = np.where(a[0] + a[1] < 0.5, 0.0, 4.0)
    np.testing.assert_array_equal(got["OUT"], expected)
