# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every path through a conditional, and out of a loop, leaves a container where the code after it expects it.

* A conditional without ``else`` has a path that skips the arm, so the arm restores what it moved, and the copies it
  needs stay inside it, where only the path that runs the arm pays for them.
* The blocks after a conditional know the locations every arm leaves, not only the first.
* A loop body ending in a conditional copies back once per iteration, not once per branch.
* A copy in front of an interstate edge is placed after the block the edge leaves, also when that block is a loop.
* The placement visits each arm once, not once per route: 48 conditionals in a row have 2^48 routes.
"""

import collections
import sys

import numpy as np
import pytest

import dace
from dace import InterstateEdge, Memlet
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol("N")
#: Below this extent the host arm of ``later_arm_writes_then_gap_then_host_read`` runs, which leaves ``T`` unwritten.
SMALL = 4
IN_A_ROW = 48
STEPS = 3


def offloaded(sdfg: dace.SDFG) -> dace.SDFG:
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def launch_128_wide(sdfg: dace.SDFG) -> dace.SDFG:
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    return sdfg


def copies(states) -> list[tuple[str, str]]:
    """``(source, destination)`` of every container-to-container copy in ``states``."""
    return [
        (edge.src.data, edge.dst.data)
        for state in states
        if isinstance(state, dace.SDFGState)
        for edge in state.edges()
        if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode)
    ]


def in_a_loop(state: dace.SDFGState) -> bool:
    region = state.parent_graph
    while region is not None and not isinstance(region, (LoopRegion, dace.SDFG)):
        region = region.parent_graph
    return isinstance(region, LoopRegion)


def host_device_copies(sdfg: dace.SDFG, only_in_loops: bool = False) -> collections.Counter:
    """``(source, destination, region)`` of every copy between host and device memory."""
    found = collections.Counter()
    for node, state in sdfg.all_nodes_recursive():
        if not (isinstance(state, dace.SDFGState) and isinstance(node, dace.nodes.AccessNode)):
            continue
        if only_in_loops and not in_a_loop(state):
            continue
        for edge in state.out_edges(node):
            if not isinstance(edge.dst, dace.nodes.AccessNode):
                continue
            on_device = [n.desc(state.sdfg).storage == dace.StorageType.GPU_Global for n in (node, edge.dst)]
            if on_device[0] != on_device[1]:
                found[(node.data, edge.dst.data, state.parent_graph.label)] += 1
    return found


@dace.program
def kernel_arm_then_host_read(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    if c > 0:
        for i in dace.map[0:N]:
            A[i] = A[i] * 2.0
    B[0] = A[0]


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


def test_the_blocks_after_a_conditional_know_where_its_later_arm_left_the_data():
    sdfg = offloaded(later_arm_writes_then_gap_then_host_read())
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
    assert ("T", read_T.data) in copies(sdfg.states()), copies(sdfg.states())


@pytest.mark.gpu
def test_the_host_read_after_the_conditional_sees_the_device_result():
    sdfg = offloaded(later_arm_writes_then_gap_then_host_read())
    A = np.arange(1, 2 * SMALL + 1, dtype=np.float64)
    X = np.zeros(1)
    out = np.zeros(1)
    sdfg(A=A, X=X, out=out, N=2 * SMALL)
    np.testing.assert_array_equal(out, [2 * A[0]])


@dace.program
def loop_ending_in_branches(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    for k in range(4):
        for i in dace.map[0:N]:
            B[i] = A[i] + 1.0
        if c > 2:
            A[0] = A[0] + 1.0
        elif c > 1:
            A[1] = 2.0
        elif c > 0:
            A[2] = 3.0
        else:
            A[3] = 4.0


def loop_ending_in_branches_reference(A: np.ndarray, B: np.ndarray, c: int) -> None:
    for attempt in range(4):
        B[:] = A + 1.0
        if c > 2:
            A[0] = A[0] + 1.0
        elif c > 1:
            A[1] = 2.0
        elif c > 0:
            A[2] = 3.0
        else:
            A[3] = 4.0


def test_a_loop_ending_in_a_conditional_copies_back_to_the_device_once():
    copied = host_device_copies(offloaded(loop_ending_in_branches.to_sdfg(simplify=True)))
    repeated = {copy: n for copy, n in copied.items() if n > 1}
    assert not repeated, repeated


@pytest.mark.gpu
@pytest.mark.parametrize("c", [0, 1, 2, 3])
def test_the_loop_ending_in_a_conditional_computes_what_numpy_computes(c):
    sdfg = launch_128_wide(offloaded(loop_ending_in_branches.to_sdfg(simplify=True)))
    A = np.arange(8, dtype=np.float64)
    B = np.zeros(8)
    want_A, want_B = A.copy(), B.copy()
    loop_ending_in_branches_reference(want_A, want_B, c)
    sdfg(A=A, B=B, c=c, N=8)
    np.testing.assert_array_equal(A, want_A)
    np.testing.assert_array_equal(B, want_B)


@dace.program
def loop_then_host_read(A: dace.float64[N], out: dace.float64[1]):
    for k in range(3):
        for i in dace.map[0:N]:
            A[i] = A[i] + 1.0
    s = A[0]
    if s > 2.0:
        out[0] = s


def test_a_loop_before_a_host_reading_edge_is_followed_by_its_copy_back():
    sdfg = offloaded(loop_then_host_read.to_sdfg(simplify=True))
    loop = next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))
    after = sdfg.successors(loop)
    assert len(after) == 1 and isinstance(after[0], dace.SDFGState), after
    copied = {e.data.data for e in after[0].edges()}
    assert "A_gpu" in copied, copied


@pytest.mark.gpu
def test_the_host_read_after_the_loop_sees_the_device_result():
    sdfg = launch_128_wide(offloaded(loop_then_host_read.to_sdfg(simplify=True)))
    A = np.arange(8, dtype=np.float64)
    out = np.zeros(1)
    sdfg(A=A, out=out, N=8)
    np.testing.assert_array_equal(A, np.arange(8) + 3.0)
    np.testing.assert_array_equal(out, [3.0])


@dace.program
def conditionals_in_a_row(A: dace.float64[N], c: dace.int64):
    for step in range(STEPS):
        for i in dace.map[0:N]:
            A[i] = A[i] * 0.5
        for j in dace.unroll(range(IN_A_ROW)):
            if c > j:
                for i in dace.map[0:N]:
                    A[i] = A[i] + 1.0
            else:
                for i in dace.map[0:N]:
                    A[i] = A[i] - 1.0


def conditionals_in_a_row_reference(A: np.ndarray, c: int) -> None:
    for step in range(STEPS):
        taken = min(max(c, 0), IN_A_ROW)
        A *= 0.5
        A += taken - (IN_A_ROW - taken)


def test_conditionals_in_a_row_are_offloaded_and_keep_the_data_on_the_device():
    """The offload finishes, every map is a kernel, and nothing crosses the bus inside the loop."""
    sdfg = offloaded(conditionals_in_a_row.to_sdfg(simplify=True))
    maps = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert len(maps) == 2 * IN_A_ROW + 1
    assert all(entry.map.schedule == dace.ScheduleType.GPU_Device for entry in maps)
    assert not host_device_copies(sdfg, only_in_loops=True)


@pytest.mark.gpu
@pytest.mark.parametrize("c", [0, 5, IN_A_ROW + 2])
def test_the_offloaded_conditionals_compute_what_numpy_computes(c):
    sdfg = launch_128_wide(offloaded(conditionals_in_a_row.to_sdfg(simplify=True)))
    A = np.arange(16, dtype=np.float64)
    want = A.copy()
    conditionals_in_a_row_reference(want, c)
    sdfg(A=A, c=c, N=16)
    np.testing.assert_array_equal(A, want)


if __name__ == "__main__":
    test_the_blocks_after_a_conditional_know_where_its_later_arm_left_the_data()
    test_a_loop_ending_in_a_conditional_copies_back_to_the_device_once()
    test_a_loop_before_a_host_reading_edge_is_followed_by_its_copy_back()
    test_conditionals_in_a_row_are_offloaded_and_keep_the_data_on_the_device()
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        test_the_host_read_after_the_conditional_sees_the_device_result()
        for c in [0, 1, 2, 3]:
            test_the_loop_ending_in_a_conditional_computes_what_numpy_computes(c)
        test_the_host_read_after_the_loop_sees_the_device_result()
        for c in [0, 5, IN_A_ROW + 2]:
            test_the_offloaded_conditionals_compute_what_numpy_computes(c)
