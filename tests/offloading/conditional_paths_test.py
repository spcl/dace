# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every path through a conditional leaves a container where the code after it expects it.

A conditional without ``else`` has a path that skips the arm, so the arm must restore what it moved; the copies it
needs belong inside it, where only the path that runs the arm pays for them.
"""
import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator
from shared_graphs import host_write_then_kernel_arm

N = dace.symbol('N')


@dace.program
def kernel_arm_then_host_read(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    if c > 0:
        for i in dace.map[0:N]:
            A[i] = A[i] * 2.0
    B[0] = A[0]


def offloaded(sdfg: dace.SDFG) -> dace.SDFG:
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def copies(region: ControlFlowRegion) -> list:
    """``(source, destination)`` of every container-to-container copy in the states directly under ``region``."""
    return [(edge.src.data, edge.dst.data) for state in region.nodes() if isinstance(state, dace.SDFGState)
            for edge in state.edges()
            if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode)]


def test_the_path_that_skips_the_arm_copies_what_the_kernel_after_it_reads():
    sdfg = offloaded(host_write_then_kernel_arm())
    guard = next(block for block in sdfg.nodes() if isinstance(block, ConditionalBlock))
    (next_block, ) = sdfg.successors(guard)

    assert ('A', 'A_gpu') in copies(sdfg), 'no copy of A to the device outside the arm: the skipping path has none'
    assert next_block.label.startswith('copy_A_'), next_block.label


def test_the_copies_of_a_one_armed_conditional_stay_inside_its_arm():
    sdfg = offloaded(kernel_arm_then_host_read.to_sdfg(simplify=True))
    guard = next(block for block in sdfg.nodes() if isinstance(block, ConditionalBlock))
    (_, arm), = guard.branches

    assert not [copy for copy in copies(sdfg) if 'A' in copy and 'A_gpu' in copy], 'a copy outside the conditional'
    assert ('A', 'A_gpu') in copies(arm) and ('A_gpu', 'A') in copies(arm), copies(arm)


if __name__ == '__main__':
    test_the_path_that_skips_the_arm_copies_what_the_kernel_after_it_reads()
    test_the_copies_of_a_one_armed_conditional_stay_inside_its_arm()
