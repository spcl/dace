# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient allocated under a loop-variant condition and freed at the end of the loop body is freed once."""

import re

import numpy as np

import dace
from dace import InterstateEdge
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

N = dace.symbol("N")


def build() -> dace.SDFG:
    """``DO k``: ``IF (flags(k))`` sizes, allocates and fills ``buf``; a second ``IF (flags(k))`` copies it out; the
    free ends the loop body."""
    sdfg = dace.SDFG("allocated_in_a_loop_variant_if")
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_array("flags", [2], dace.int32)
    sdfg.add_symbol("d0", dace.int64)
    sdfg.add_transient("buf", ["d0"], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "k < 2", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, InterstateEdge(assignments={"d0": "0"}))
    start = loop.add_state("start", is_start_block=True)
    previous = start
    for label in ("fill", "cp"):
        cond = ConditionalBlock(f"if_{label}")
        loop.add_node(cond)
        loop.add_edge(previous, cond, InterstateEdge())
        arm = ControlFlowRegion(f"arm_{label}", sdfg)
        cond.add_branch(CodeBlock("flags[k] != 0"), arm)
        first = arm.add_state(f"start_{label}", is_start_block=True)
        state = arm.add_state(label)
        arm.add_edge(first, state, InterstateEdge(assignments={"d0": "N"} if label == "fill" else {}))
        if label == "fill":
            state.add_mapped_tasklet(
                "fill", {"i": "0:d0"}, {}, "o = i + k", {"o": dace.Memlet("buf[i]")}, external_edges=True
            )
        else:
            state.add_mapped_tasklet(
                "cp",
                {"i": "0:N"},
                {"x": dace.Memlet("buf[i]")},
                "o = x",
                {"o": dace.Memlet("out[i]")},
                external_edges=True,
            )
        previous = cond
    return sdfg


# Reproducer: the free at the end of the loop body left the pointer dangling, so an iteration that skipped the
# allocation freed the previous iteration's buffer again (valgrind: "Invalid free()")
def test_a_free_past_a_skipped_allocation_frees_nothing():
    sdfg = build()
    code = sdfg.generate_code()[0].clean_code
    assert re.search(r"delete_array\(buf, 64\);\s*buf = nullptr;", code), code
    for flags, expected in (((1, 0), np.arange(5)), ((0, 1), np.arange(5) + 1), ((1, 1), np.arange(5) + 1)):
        out = np.zeros(5)
        sdfg(out=out, flags=np.array(flags, dtype=np.int32), N=5)
        assert np.allclose(out, expected), flags


if __name__ == "__main__":
    test_a_free_past_a_skipped_allocation_frees_nothing()
