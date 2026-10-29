# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The sequential ``Reduce`` expansion that ``auto`` picks inside a parallel map keeps the input type until the
store."""

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


def test_mixed_type_sum_keeps_the_input_type_until_the_store():
    """A Fortran ``SUM`` of an integer array into a real: the accumulator stays integer and a tasklet casts."""
    sdfg = reduce_sdfg("reduce_mixed_seq", "lambda a, b: a + b", dace.int32, dace.float64, 0)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Reduce):
            node.implementation = "pure-seq"
    a, b = np.arange(N, dtype=np.int32), np.zeros(1)
    sdfg(A=a, B=b)
    assert b[0] == float(a.sum())
