# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Data-dependent map ranges read array elements in their bounds directly.

A range like ``dace.map[A[i]:A[i + 1]]`` reads ``A[i]`` and ``A[i + 1]`` as dynamic map inputs, so that no
scalar copy sits between the map and the scope enclosing it. A bound given through a variable reads the value the
variable holds, which is a copy of the element.
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
    N = dace.symbol("N")

    @dace.program
    def rowsum(indptr: dace.int32[N + 1], vals: dace.float64[N * N], out: dace.float64[N]):
        for i in range(N):
            for j in dace.map[indptr[i] : indptr[i + 1] - 1]:
                out[i] += vals[j]

    sdfg = rowsum.to_sdfg(simplify=False)
    inputs = _dynamic_inputs(sdfg)
    assert len(inputs) == 2, inputs
    assert all(memlet.data == "indptr" for memlet in inputs.values()), inputs
    assert {str(memlet.subset) for memlet in inputs.values()} == {"i", "i + 1"}, inputs

    # The rest of the bound expression is kept
    ranges = [str(n.map.range) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.nodes.MapEntry)]
    assert any(r.endswith(" - 1") for r in ranges), ranges

    indptr = np.array([0, 3, 3, 7], dtype=np.int32)
    vals = np.random.rand(9)
    out = np.zeros(3)
    sdfg(indptr=indptr, vals=vals, out=out, N=3)
    expected = np.array([vals[0:2].sum(), 0.0, vals[3:6].sum()])
    assert np.allclose(out, expected), out


def test_range_reads_variables():
    """Bounds given through variables read the scalars that hold the elements, which in turn copy them."""
    N = dace.symbol("N")

    @dace.program
    def rowsum(indptr: dace.int32[N + 1], vals: dace.float64[N * N], out: dace.float64[N]):
        for i in range(N):
            start = indptr[i]
            end = indptr[i + 1]
            for j in dace.map[start:end]:
                out[i] += vals[j]

    sdfg = rowsum.to_sdfg(simplify=False)
    inputs = _dynamic_inputs(sdfg)
    assert len(inputs) == 2, inputs

    # Every scalar a bound reads is a copy of an element of ``indptr``
    copied = set()
    for state in sdfg.all_states():
        for node in state.data_nodes():
            if any(memlet.data == node.data for memlet in inputs.values()):
                for edge in state.in_edges(node):
                    assert edge.data.data == "indptr", edge.data
                    copied.add(str(edge.data.subset))
    assert copied == {"i", "i + 1"}, copied

    indptr = np.array([0, 3, 3, 7], dtype=np.int32)
    vals = np.random.rand(9)
    expected = np.array([vals[0:3].sum(), 0.0, vals[3:7].sum()])
    for simplify in (False, True):
        out = np.zeros(3)
        rowsum.to_sdfg(simplify=simplify)(indptr=indptr, vals=vals, out=out, N=3)
        assert np.allclose(out, expected), (simplify, out)


def test_range_keeps_the_value_it_was_given():
    """A write to the array in between makes the copy and the element differ; the copy is right."""

    @dace.program
    def shifted(A: dace.int32[10], out: dace.float64[10]):
        start = A[0]
        A[0] = 5
        for j in dace.map[start:10]:
            out[j] = 1.0

    inputs = _dynamic_inputs(shifted.to_sdfg(simplify=False))
    assert all(memlet.data != "A" for memlet in inputs.values()), inputs

    A = np.zeros(10, dtype=np.int32)
    A[0] = 2
    out = np.zeros(10)
    shifted(A, out)
    expected = np.zeros(10)
    expected[2:] = 1.0
    assert np.allclose(out, expected), out


if __name__ == "__main__":
    test_range_reads_the_element_itself()
    test_range_reads_variables()
    test_range_keeps_the_value_it_was_given()
