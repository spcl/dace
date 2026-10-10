# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A container only the later arm of a conditional writes on the device is copied back for a host read after it.

Location propagation walked the IR in pre-order, so the conditional's close node forwarded its locations
once its FIRST arm reached it. A transient only the second arm writes then had no location in the blocks
after the conditional, and the host read behind them was renamed to a host twin nothing declared.
"""

import numpy as np
import pytest

import dace
from dace import InterstateEdge, Memlet
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol("N")
#: Below this extent the host arm runs, which leaves ``T`` unwritten.
SMALL = 4


def later_arm_writes_then_gap_then_host_read() -> dace.SDFG:
    """``if N < SMALL: X[0] = 1`` else ``T = 2 * A`` in a kernel; an empty state; then ``out = T[0]`` on the host."""
    sdfg = dace.SDFG("later_arm_writes_then_gap_then_host_read")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("X", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("T", [N], dace.float64)
    start = sdfg.add_state("start", is_start_block=True)

    guard = ConditionalBlock("guard")
    sdfg.add_node(guard)
    sdfg.add_edge(start, guard, InterstateEdge())
    host_arm = ControlFlowRegion("host_arm", sdfg=sdfg)
    guard.add_branch(CodeBlock(f"N < {SMALL}"), host_arm)
    touch = host_arm.add_state("touch_x", is_start_block=True)
    one = touch.add_tasklet("one", {}, {"o"}, "o = 1.0")
    touch.add_edge(one, "o", touch.add_write("X"), None, Memlet("X[0]"))
    device_arm = ControlFlowRegion("device_arm", sdfg=sdfg)
    guard.add_branch(None, device_arm)
    device_arm.add_state("write_T", is_start_block=True).add_mapped_tasklet(
        "double", {"i": "0:N"}, {"a": Memlet("A[i]")}, "t = 2 * a", {"t": Memlet("T[i]")}, external_edges=True
    )

    gap = sdfg.add_state("gap")
    sdfg.add_edge(guard, gap, InterstateEdge())
    read = sdfg.add_state_after(gap, "read_T")
    first = read.add_tasklet("first", {"t"}, {"o"}, "o = t")
    read.add_edge(read.add_read("T"), None, first, "t", Memlet("T[0]"))
    read.add_edge(first, "o", read.add_write("out"), None, Memlet("out[0]"))
    sdfg.validate()
    return sdfg


def offloaded() -> dace.SDFG:
    sdfg = later_arm_writes_then_gap_then_host_read()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def test_the_blocks_after_a_conditional_know_where_its_later_arm_left_the_data():
    sdfg = offloaded()
    undeclared = {
        (state.label, node.data)
        for state in sdfg.states()
        for node in state.data_nodes()
        if node.data not in sdfg.arrays
    }
    assert not undeclared, undeclared
    read = next(state for state in sdfg.states() if state.label == "read_T")
    (read_T,) = [node for node in read.data_nodes() if node.data.startswith("T")]
    assert sdfg.arrays[read_T.data].storage != dace.StorageType.GPU_Global
    copies = [
        (src.data, dst.data)
        for state in sdfg.states()
        for src in state.data_nodes()
        for dst in state.successors(src)
        if isinstance(dst, dace.nodes.AccessNode)
    ]
    assert ("T", read_T.data) in copies, copies


@pytest.mark.gpu
def test_the_host_read_after_the_conditional_sees_the_device_result():
    sdfg = offloaded()
    A = np.arange(1, 2 * SMALL + 1, dtype=np.float64)
    X = np.zeros(1)
    out = np.zeros(1)
    sdfg(A=A, X=X, out=out, N=2 * SMALL)
    np.testing.assert_array_equal(out, [2 * A[0]])
