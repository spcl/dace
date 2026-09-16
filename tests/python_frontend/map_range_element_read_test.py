# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Data-dependent map ranges read the element they were given, not a copy of it.

A range like ``dace.map[start:end]`` built from ``start = A[i]`` reads ``A[i]`` itself, so that no
scalar sits between the two maps and keeps them from being seen as one nested directly inside the
other. That shortcut only holds while the element still has the value that was read out of it.
"""
import numpy as np

import dace


def _dynamic_inputs(sdfg: dace.SDFG):
    """Map the dynamic-range connectors of every map in ``sdfg`` to the memlets feeding them."""
    inputs = {}
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            for edge in dace.sdfg.dynamic_map_inputs(state, node):
                inputs[edge.dst_conn] = edge.data
    return inputs


def test_range_reads_the_element_itself():
    N = dace.symbol('N')

    @dace.program
    def rowsum(indptr: dace.int32[N + 1], vals: dace.float64[N * N], out: dace.float64[N]):
        for i in range(N):
            start = indptr[i]
            end = indptr[i + 1]
            for j in dace.map[start:end]:
                out[i] += vals[j]

    inputs = _dynamic_inputs(rowsum.to_sdfg(simplify=False))
    assert len(inputs) == 2, inputs
    assert all(memlet.data == 'indptr' for memlet in inputs.values()), inputs
    assert {str(memlet.subset) for memlet in inputs.values()} == {'i', 'i + 1'}, inputs


def test_range_keeps_the_value_it_was_given():
    """A write to the array in between makes the copy and the element differ; the copy is right."""

    @dace.program
    def shifted(A: dace.int32[10], out: dace.float64[10]):
        start = A[0]
        A[0] = 5
        for j in dace.map[start:10]:
            out[j] = 1.0

    inputs = _dynamic_inputs(shifted.to_sdfg(simplify=False))
    assert all(memlet.data != 'A' for memlet in inputs.values()), inputs

    A = np.zeros(10, dtype=np.int32)
    A[0] = 2
    out = np.zeros(10)
    shifted(A, out)
    expected = np.zeros(10)
    expected[2:] = 1.0
    assert np.allclose(out, expected), out


if __name__ == '__main__':
    test_range_reads_the_element_itself()
    test_range_keeps_the_value_it_was_given()
