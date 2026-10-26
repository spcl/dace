# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Reduce``'s default ``auto`` implementation: the ``pure`` WCR map, or the sequential accumulator inside a parallel
map; both compile and compute, including a mixed-type sum."""

import numpy as np

import dace
from dace.libraries.standard.nodes.reduce import Reduce

N = 64


def reduce_sdfg(name: str, wcr: str, in_dtype, out_dtype, identity) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), in_dtype)
    sdfg.add_array("B", (1,), out_dtype)
    state = sdfg.add_state()
    red = Reduce("reduce", wcr, axes=None, identity=identity)
    state.add_node(red)
    state.add_edge(state.add_read("A"), None, red, None, dace.Memlet(f"A[0:{N}]"))
    state.add_edge(red, None, state.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg


def test_default_cpu_reduce_is_the_wcr_map():
    sdfg = reduce_sdfg("reduce_auto_wcr_map", "lambda a, b: max(a, b)", dace.float64, dace.float64, None)
    sdfg.expand_library_nodes()
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert all("#pragma omp" not in t.code.as_string for t in tasklets)
    assert any(e.data.wcr is not None for e, _ in sdfg.all_edges_recursive())

    # Without an identity the reduction starts from what B holds
    a, b = np.random.rand(N), np.full(1, -np.inf)
    sdfg(A=a, B=b)
    assert b[0] == a.max()


def test_mixed_type_sum_keeps_the_input_type_until_the_store():
    """A Fortran ``SUM`` of an integer array into a real: the accumulator stays integer and a tasklet casts."""
    sdfg = reduce_sdfg("reduce_mixed_seq", "lambda a, b: a + b", dace.int32, dace.float64, 0)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Reduce):
            node.implementation = "pure-seq"
    a, b = np.arange(N, dtype=np.int32), np.zeros(1)
    sdfg(A=a, B=b)
    assert b[0] == float(a.sum())


def test_reduction_inside_a_parallel_map_is_sequential():
    @dace.program
    def rowsum(A: dace.float64[8, N], B: dace.float64[8]):
        for i in dace.map[0:8]:
            B[i] = np.sum(A[i])

    sdfg = rowsum.to_sdfg(simplify=True)
    a, b = np.random.rand(8, N), np.zeros(8)
    sdfg(A=a, B=b)
    assert np.allclose(b, a.sum(axis=1))
