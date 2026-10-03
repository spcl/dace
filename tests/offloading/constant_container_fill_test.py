# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A container nothing writes is filled on the other side once, at the program's entry.

The copy analysis records WHERE a container is needed at each program point, not whether its data
changed, so a device array the host reads inside a loop is moved to the host and back around every
iteration: polybench nussinov read ``seq`` on the host inside its ``j`` loop and paid a
``seq -> seq_host`` fill in the loop body and a ``seq_host -> seq`` copy at the end of each
iteration, 6.4 million pageable copies per call at N=3591 (400 s against a 22 s kernel). Neither
side ever writes ``seq``, so every copy after the first carries the bytes already there.
"""
import collections

import numpy as np
import pytest

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol('N')


@dace.program
def match(b1: dace.int32, b2: dace.int32):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


@dace.program
def nussinov(seq: dace.int32[N], table: dace.int32[N, N]):
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1] + match(seq[i], seq[j]))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])


def reference(seq: np.ndarray, table: np.ndarray) -> None:
    n = seq.shape[0]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < n:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < n:
                if i < j - 1:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1] + (1 if seq[i] + seq[j] == 3 else 0))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])


def offloaded() -> dace.SDFG:
    """``nussinov`` with its arguments resident on the device, the way a GPU column hands them over."""
    sdfg = nussinov.to_sdfg(simplify=True)
    for desc in sdfg.arrays.values():
        if not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def seq_copies(sdfg: dace.SDFG) -> collections.Counter:
    """``(source, destination, enclosing region)`` of every copy between ``seq`` and its twin."""
    found = collections.Counter()
    for node, state in sdfg.all_nodes_recursive():
        if not (isinstance(state, dace.SDFGState) and isinstance(node, dace.nodes.AccessNode)):
            continue
        for edge in state.out_edges(node):
            if isinstance(edge.dst, dace.nodes.AccessNode) and {node.data, edge.dst.data} == {'seq', 'seq_host'}:
                region = state.parent_graph
                found[(node.data, edge.dst.data, 'top' if region is sdfg else region.label)] += 1
    return found


def test_a_container_nothing_writes_is_filled_once_at_the_entry():
    assert seq_copies(offloaded()) == collections.Counter({('seq', 'seq_host', 'top'): 1})


@pytest.mark.gpu
def test_the_offloaded_recurrence_computes_what_numpy_computes():
    import cupy
    sdfg = offloaded()
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 4, size=40).astype(np.int32)
    table = np.zeros((40, 40), dtype=np.int32)
    want = table.copy()
    reference(seq, want)
    d_seq, d_table = cupy.asarray(seq), cupy.asarray(table)
    sdfg(seq=d_seq, table=d_table, N=40)
    np.testing.assert_array_equal(cupy.asnumpy(d_table), want)
    np.testing.assert_array_equal(cupy.asnumpy(d_seq), seq)
