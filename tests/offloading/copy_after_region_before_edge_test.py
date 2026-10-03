# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop whose kernel writes an array the next interstate edge reads on the host gets its copy after the loop.

A loop is not a state, and a copy in front of an interstate edge is placed after the block the edge leaves.
"""
import sys

import numpy as np
import pytest

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol('N')


@dace.program
def loop_then_host_read(A: dace.float64[N], out: dace.float64[1]):
    for k in range(3):
        for i in dace.map[0:N]:
            A[i] = A[i] + 1.0
    s = A[0]
    if s > 2.0:
        out[0] = s


def offloaded() -> dace.SDFG:
    sdfg = loop_then_host_read.to_sdfg(simplify=True)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def test_a_loop_before_a_host_reading_edge_is_followed_by_its_copy_back():
    sdfg = offloaded()
    loop = next(b for b in sdfg.nodes() if isinstance(b, dace.sdfg.state.LoopRegion))
    after = sdfg.successors(loop)
    assert len(after) == 1 and isinstance(after[0], dace.SDFGState), after
    copied = {e.data.data for e in after[0].edges()}
    assert 'A_gpu' in copied, copied


@pytest.mark.gpu
def test_the_host_read_after_the_loop_sees_the_device_result():
    sdfg = offloaded()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    A = np.arange(8, dtype=np.float64)
    out = np.zeros(1)
    sdfg(A=A, out=out, N=8)
    np.testing.assert_array_equal(A, np.arange(8) + 3.0)
    np.testing.assert_array_equal(out, [3.0])


if __name__ == '__main__':
    test_a_loop_before_a_host_reading_edge_is_followed_by_its_copy_back()
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        test_the_host_read_after_the_loop_sees_the_device_result()
