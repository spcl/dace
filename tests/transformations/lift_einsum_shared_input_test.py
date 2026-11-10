# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LiftEinsum`` on a map whose two operands read ONE map input: ``np.sum(c * c, axis=1)`` (kmeans).

The operands were keyed by the map connector they enter through, so the second overwrote the first
and the lifted einsum kept one of its two inputs wired: ``Dangling in-connector __in1``.
"""

import numpy as np

import dace
from dace.libraries.blas import Einsum
from dace.transformation.dataflow.lift_einsum import LiftEinsum

M, N = (dace.symbol(s) for s in "MN")


def row_square_norms(zero_first: bool = False) -> dace.SDFG:
    """``out[i] += c[i, j] * c[i, j]`` with both tasklet inputs fed by the map's single ``IN_c``; ``zero_first``
    zeroes ``out`` in a map ordered before it by an empty edge only, as kmeans' ``c_sqnorm``."""
    sdfg = dace.SDFG(f"lift_einsum_shared_input_{zero_first}")
    sdfg.add_array("c", [M, N], dace.float64)
    sdfg.add_array("out", [M], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("sq", {"i": "0:M", "j": "0:N"})
    tasklet = state.add_tasklet("mul", {"__in1", "__in2"}, {"__out"}, "__out = __in1 * __in2")
    entry.add_in_connector("IN_c")
    entry.add_out_connector("OUT_c")
    state.add_edge(state.add_read("c"), None, entry, "IN_c", dace.Memlet("c[0:M, 0:N]"))
    if zero_first:
        zero = state.add_mapped_tasklet(
            "zero", {"i": "0:M"}, {}, "o = 0", {"o": dace.Memlet("out[i]")}, external_edges=True
        )
        state.add_nedge(state.out_edges(zero[2])[0].dst, entry, dace.Memlet())
    for operand in ("__in1", "__in2"):
        state.add_edge(entry, "OUT_c", tasklet, operand, dace.Memlet("c[i, j]"))
    state.add_memlet_path(
        tasklet,
        exit_node,
        state.add_write("out"),
        src_conn="__out",
        memlet=dace.Memlet("out[i]", wcr="lambda a, b: a + b"),
    )
    sdfg.validate()
    return sdfg


def test_both_operands_of_one_map_input_are_wired():
    sdfg = row_square_norms()
    assert sdfg.apply_transformations(LiftEinsum) == 1
    sdfg.validate()
    m, n = 5, 7
    c = np.random.default_rng(0).random((m, n))
    out = np.zeros(m)
    sdfg(c=c, out=out, M=m, N=n)
    assert np.allclose(out, (c * c).sum(axis=1))


def test_ordering_edge_into_the_map_orders_the_einsum():
    sdfg = row_square_norms(zero_first=True)
    state = sdfg.start_state
    zeroed = next(e.src for e in state.edges() if e.data.is_empty() and isinstance(e.src, dace.nodes.AccessNode))
    assert sdfg.apply_transformations(LiftEinsum) == 1
    sdfg.validate()
    einsum = next(n for n in state.nodes() if isinstance(n, Einsum))
    assert dace.symbolic.equal_valued(1, einsum.beta)
    # the einsum folds onto the zeroed ``out``: without the edge nothing orders the zeroing before it
    assert [e.src for e in state.in_edges(einsum) if e.data.is_empty()] == [zeroed]
    m, n = 5, 7
    c = np.random.default_rng(0).random((m, n))
    out = np.full(m, 3.0)
    sdfg(c=c, out=out, M=m, N=n)
    assert np.allclose(out, (c * c).sum(axis=1))


if __name__ == "__main__":
    test_both_operands_of_one_map_input_are_wired()
    test_ordering_edge_into_the_map_orders_the_einsum()
