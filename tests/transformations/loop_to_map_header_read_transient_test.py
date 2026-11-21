# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` must not privatize a transient that a header OUTSIDE the loop reads.

A symbol the frontend demotes to a scalar container (fv3_dycore's inlined ``ord``: ``if hord == 10: o = 8 else:
o = hord`` inside one loop, ``if abs(o) == 5`` in the conditions of the next) is written by access nodes in the
first loop and read by nothing but ``ConditionalBlock`` headers of the second. Loop-locality was decided by
access nodes alone, so the first loop's lift moved the scalar into its nested SDFG as a private transient; the
second loop's headers were left naming a symbol nothing defines, and every later ``arglist()`` raised KeyError.
"""

import numpy as np

import dace
from dace.sdfg.state import CodeBlock, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol("N")


def build_sdfg() -> dace.SDFG:
    """``for i: o = 5`` then ``for j: if o == 5: B[j] = 1 else: B[j] = -1`` with ``o`` a transient scalar."""
    sdfg = dace.SDFG("loop_to_map_header_read_transient")
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_scalar("o", dace.int64, transient=True)

    writer = LoopRegion("writer", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(writer, is_start_block=True)
    state = writer.add_state("assign", is_start_block=True)
    tasklet = state.add_tasklet("assign", {}, {"__out"}, "__out = 5")
    state.add_edge(tasklet, "__out", state.add_write("o"), None, dace.Memlet("o[0]"))

    reader = LoopRegion("reader", "j < N", "j", "j = 0", "j = j + 1")
    sdfg.add_node(reader)
    sdfg.add_edge(writer, reader, dace.InterstateEdge())
    guard = ConditionalBlock("guard", sdfg=sdfg)
    reader.add_node(guard, is_start_block=True)
    for label, value, condition in (("then_body", 1.0, CodeBlock("o == 5")), ("else_body", -1.0, None)):
        branch = ControlFlowRegion(label, sdfg=sdfg, parent=guard)
        body = branch.add_state(f"{label}_state", is_start_block=True)
        write = body.add_tasklet(label, {}, {"__out"}, f"__out = {value}")
        body.add_edge(write, "__out", body.add_write("B"), None, dace.Memlet("B[j]"))
        guard.add_branch(condition, branch)

    sdfg.validate()
    return sdfg


def test_a_transient_an_outside_header_reads_stays_in_the_parent():
    sdfg = build_sdfg()
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert "o" in sdfg.arrays, "the scalar the reader's header reads was moved into a nested SDFG"
    assert "o" not in {str(s) for s in sdfg.free_symbols}, sdfg.free_symbols


def test_a_transient_an_outside_header_reads_keeps_its_value():
    sdfg = build_sdfg()
    sdfg.apply_transformations_repeated(LoopToMap)
    out = np.zeros(8)
    sdfg(B=out, N=8)
    assert np.array_equal(out, np.ones(8))


if __name__ == "__main__":
    test_a_transient_an_outside_header_reads_stays_in_the_parent()
    test_a_transient_an_outside_header_reads_keeps_its_value()
