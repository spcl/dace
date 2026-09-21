# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop whose body ends in a conditional copies its host writes back to the device once per iteration.

Every branch of the conditional is a tail of the loop body and each one resolves to the same ConditionalBlock,
so the offloader stacked one identical ``A -> A_gpu`` copy state after it per branch: 24 in a row after one
CLOUDSC branch, 11704 repeated copies over the whole graph.
"""
import collections

import numpy as np
import pytest

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol('N')


@dace.program
def branches(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
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


def reference(A: np.ndarray, B: np.ndarray, c: int) -> None:
    for _ in range(4):
        B[:] = A + 1.0
        if c > 2:
            A[0] = A[0] + 1.0
        elif c > 1:
            A[1] = 2.0
        elif c > 0:
            A[2] = 3.0
        else:
            A[3] = 4.0


def offloaded() -> dace.SDFG:
    sdfg = branches.to_sdfg(simplify=True)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def host_device_copies(sdfg: dace.SDFG) -> collections.Counter:
    """``(source, destination, region)`` of every copy between host and device memory."""
    found = collections.Counter()
    for node, state in sdfg.all_nodes_recursive():
        if not (isinstance(state, dace.SDFGState) and isinstance(node, dace.nodes.AccessNode)):
            continue
        for edge in state.out_edges(node):
            if not isinstance(edge.dst, dace.nodes.AccessNode):
                continue
            on_device = [n.desc(state.sdfg).storage == dace.StorageType.GPU_Global for n in (node, edge.dst)]
            if on_device[0] != on_device[1]:
                found[(node.data, edge.dst.data, state.parent_graph.label)] += 1
    return found


def test_a_loop_ending_in_a_conditional_copies_back_to_the_device_once():
    repeated = {copy: n for copy, n in host_device_copies(offloaded()).items() if n > 1}
    assert not repeated, repeated


@pytest.mark.gpu
@pytest.mark.parametrize('c', [0, 1, 2, 3])
def test_the_offloaded_loop_computes_what_numpy_computes(c):
    sdfg = offloaded()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    A = np.arange(8, dtype=np.float64)
    B = np.zeros(8)
    want_A, want_B = A.copy(), B.copy()
    reference(want_A, want_B, c)
    sdfg(A=A, B=B, c=c, N=8)
    np.testing.assert_array_equal(A, want_A)
    np.testing.assert_array_equal(B, want_B)
