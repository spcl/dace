# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop body holding many conditionals in a row is offloaded in time linear in its blocks.

The placement visits each arm once, not once per route: 48 conditionals in a row have 2^48 routes.
"""
import collections
import sys

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol('N')
IN_A_ROW = 48
STEPS = 3


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


def reference(A: np.ndarray, c: int) -> None:
    for step in range(STEPS):
        taken = min(max(c, 0), IN_A_ROW)
        A *= 0.5
        A += taken - (IN_A_ROW - taken)


def offloaded() -> dace.SDFG:
    sdfg = conditionals_in_a_row.to_sdfg(simplify=True)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def copies_inside_loops(sdfg: dace.SDFG) -> collections.Counter:
    """``(source, destination)`` of every copy between host and device memory under a loop."""
    found = collections.Counter()
    for node, state in sdfg.all_nodes_recursive():
        if not (isinstance(state, dace.SDFGState) and isinstance(node, dace.nodes.AccessNode)):
            continue
        region = state.parent_graph
        while region is not None and not isinstance(region, (LoopRegion, dace.SDFG)):
            region = region.parent_graph
        if not isinstance(region, LoopRegion):
            continue
        for edge in state.out_edges(node):
            if not isinstance(edge.dst, dace.nodes.AccessNode):
                continue
            on_device = [n.desc(state.sdfg).storage == dace.StorageType.GPU_Global for n in (node, edge.dst)]
            if on_device[0] != on_device[1]:
                found[(node.data, edge.dst.data)] += 1
    return found


def test_conditionals_in_a_row_are_offloaded_and_keep_the_data_on_the_device():
    """The offload finishes, every map is a kernel, and nothing crosses the bus inside the loop."""
    sdfg = offloaded()
    maps = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert len(maps) == 2 * IN_A_ROW + 1
    assert all(entry.map.schedule == dace.ScheduleType.GPU_Device for entry in maps)
    assert not copies_inside_loops(sdfg)


CONDITIONS = [0, 5, IN_A_ROW + 2]


@pytest.mark.gpu
@pytest.mark.parametrize('c', CONDITIONS)
def test_the_offloaded_conditionals_compute_what_numpy_computes(c):
    sdfg = offloaded()
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    A = np.arange(16, dtype=np.float64)
    want = A.copy()
    reference(want, c)
    sdfg(A=A, c=c, N=16)
    np.testing.assert_array_equal(A, want)


if __name__ == '__main__':
    test_conditionals_in_a_row_are_offloaded_and_keep_the_data_on_the_device()
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        for c in CONDITIONS:
            test_the_offloaded_conditionals_compute_what_numpy_computes(c)
