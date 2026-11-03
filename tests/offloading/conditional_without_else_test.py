# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A conditional without an ``else`` that the program skips leaves its data where it was.

The copy analysis read such a conditional as its arms alone, so the blocks after it assumed the arm ran. An arm
that brings a host-fresh container down and updates it on the device left the device side current; the copy back
after the conditional then ran on the skipped path too and overwrote the host's newer values with the stale
device ones (CLOUDSC's ``zqxfg`` under ``if iphase[jm - 1] == 2`` once canonicalization lowered its maps to loops).
"""

import numpy as np
import pytest

import dace
from dace import InterstateEdge, Memlet
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol("N")


def host_update_then_device_arm() -> dace.SDFG:
    """``A = 1`` in a kernel; ``A += 10`` in a host loop; ``if flag > 0``: peek ``A`` on the host, double it on
    the device."""
    sdfg = dace.SDFG("host_update_then_device_arm")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_symbol("flag", dace.int64)
    init = sdfg.add_state("init", is_start_block=True)
    init.add_mapped_tasklet("ones", {"i": "0:N"}, {}, "a = 1.0", {"a": Memlet("A[i]")}, external_edges=True)

    update = LoopRegion("update", "j < N", "j", "j = 0", "j = j + 1", sdfg=sdfg)
    sdfg.add_node(update)
    sdfg.add_edge(init, update, InterstateEdge())
    add = update.add_state("add_ten", is_start_block=True)
    plus = add.add_tasklet("plus", {"x"}, {"y"}, "y = x + 10.0")
    add.add_edge(add.add_read("A"), None, plus, "x", Memlet("A[j]"))
    add.add_edge(plus, "y", add.add_write("A"), None, Memlet("A[j]"))

    guard = ConditionalBlock("guard")
    sdfg.add_node(guard)
    sdfg.add_edge(update, guard, InterstateEdge())
    arm = ControlFlowRegion("device_arm", sdfg=sdfg)
    guard.add_branch(CodeBlock("flag > 0"), arm)
    peek = LoopRegion("peek", "k < N", "k", "k = 0", "k = k + 1", sdfg=sdfg)
    arm.add_node(peek, is_start_block=True)
    read = peek.add_state("read_A", is_start_block=True)
    move = read.add_tasklet("move", {"x"}, {"y"}, "y = x")
    read.add_edge(read.add_read("A"), None, move, "x", Memlet("A[k]"))
    read.add_edge(move, "y", read.add_write("B"), None, Memlet("B[k]"))
    double = arm.add_state("double")
    arm.add_edge(peek, double, InterstateEdge())
    double.add_mapped_tasklet(
        "double", {"i": "0:N"}, {"x": Memlet("A[i]")}, "y = 2 * x", {"y": Memlet("A[i]")}, external_edges=True
    )
    sdfg.validate()
    return sdfg


def offloaded() -> dace.SDFG:
    sdfg = host_update_then_device_arm()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def copies_in(states) -> list[tuple[str, str]]:
    return [
        (edge.src.data, edge.dst.data)
        for state in states
        for edge in state.edges()
        if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode)
    ]


def test_the_skipped_path_refreshes_what_the_copy_after_the_conditional_reads():
    sdfg = offloaded()
    (guard,) = [block for block in sdfg.nodes() if isinstance(block, ConditionalBlock)]
    (after,) = sdfg.successors(guard)
    copied_back = {src for src, dst in copies_in([after]) if dst == "A"}
    assert copied_back, "the device result of the arm is never copied back"
    skipped = [branch for condition, branch in guard.branches if condition is None]
    assert len(skipped) == 1, "the path that takes no arm has no copies of its own"
    assert ("A", *copied_back) in copies_in(skipped[0].states()), copies_in(skipped[0].states())


@pytest.mark.gpu
def test_the_host_update_survives_a_skipped_device_arm():
    sdfg = offloaded()
    A = np.zeros(8)
    B = np.zeros(8)
    sdfg(A=A, B=B, N=8, flag=0)
    np.testing.assert_array_equal(A, np.full(8, 11.0))


@pytest.mark.gpu
def test_the_device_arm_still_doubles_the_host_update():
    sdfg = offloaded()
    A = np.zeros(8)
    B = np.zeros(8)
    sdfg(A=A, B=B, N=8, flag=1)
    np.testing.assert_array_equal(A, np.full(8, 22.0))
    np.testing.assert_array_equal(B, np.full(8, 11.0))


if __name__ == "__main__":
    test_the_skipped_path_refreshes_what_the_copy_after_the_conditional_reads()
    test_the_host_update_survives_a_skipped_device_arm()
    test_the_device_arm_still_doubles_the_host_update()
