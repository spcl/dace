# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Graphs more than one offloading test builds."""

import dace
from dace import InterstateEdge, Memlet
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

N = dace.symbol("N")


def host_write_then_kernel_arm() -> dace.SDFG:
    """``if c > 0: A[0] = 5; B = A + 1`` (a host state, then a kernel), then ``C = 2 * A`` in a kernel."""
    sdfg = dace.SDFG("host_write_then_kernel_arm")
    for name in "ABC":
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_symbol("c", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    guard = ConditionalBlock("guard")
    sdfg.add_node(guard)
    sdfg.add_edge(start, guard, InterstateEdge())
    arm = ControlFlowRegion("arm", sdfg=sdfg)
    guard.add_branch(CodeBlock("c > 0"), arm)
    write = arm.add_state("host_write", is_start_block=True)
    five = write.add_tasklet("five", {}, {"o"}, "o = 5.0")
    write.add_edge(five, "o", write.add_write("A"), None, Memlet("A[0]"))
    arm.add_state_after(write, "kernel_b").add_mapped_tasklet(
        "b", {"i": "0:N"}, {"a": Memlet("A[i]")}, "o = a + 1", {"o": Memlet("B[i]")}, external_edges=True
    )
    after = sdfg.add_state("kernel_c")
    sdfg.add_edge(guard, after, InterstateEdge())
    after.add_mapped_tasklet(
        "c", {"i": "0:N"}, {"a": Memlet("A[i]")}, "o = a * 2", {"o": Memlet("C[i]")}, external_edges=True
    )
    sdfg.validate()
    return sdfg
