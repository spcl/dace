# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``WavefrontSkew`` reads each access on its own side of a copy edge: a store ``w -> A[u, v]`` whose memlet names ``w``
writes ``A`` at its ``other_subset``, not at ``w``'s ``[0]``."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops
from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew

N = 12


def stencil_through_copy() -> dace.SDFG:
    """``A[u, v] = A[u - 1, v] + A[u, v - 1]`` over ``1 <= u, v < N``, stored through a scalar and a copy edge."""
    sdfg = dace.SDFG("stencil_through_copy")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_scalar("w", dace.float64, transient=True)
    outer = LoopRegion("outer", f"u < {N}", "u", "u = 1", "u = u + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", f"v < {N}", "v", "v = 1", "v = v + 1")
    outer.add_node(inner, is_start_block=True)
    state = inner.add_state("body", is_start_block=True)
    tasklet = state.add_tasklet("add", {"up", "left"}, {"out"}, "out = up + left")
    state.add_edge(state.add_read("A"), None, tasklet, "up", dace.Memlet("A[u - 1, v]"))
    state.add_edge(state.add_read("A"), None, tasklet, "left", dace.Memlet("A[u, v - 1]"))
    w = state.add_access("w")
    state.add_edge(tasklet, "out", w, None, dace.Memlet("w[0]"))
    state.add_edge(w, None, state.add_write("A"), None, dace.Memlet(data="w", subset="0", other_subset="u, v"))
    sdfg.validate()
    return sdfg


def test_a_store_through_a_copy_edge_naming_the_scalar_is_skewed():
    sdfg = stencil_through_copy()
    assert WavefrontSkew().apply_pass(sdfg, {}), "the copy edge's write to A must be read at A[u, v]"
    ParallelizeLoops().apply_pass(sdfg, {})
    assert any(isinstance(node, dace.nodes.MapEntry) for node, parent in sdfg.all_nodes_recursive())

    a = np.random.default_rng(0).random((N, N))
    ref = a.copy()
    for u in range(1, N):
        for v in range(1, N):
            ref[u, v] = ref[u - 1, v] + ref[u, v - 1]
    sdfg(A=a)
    assert np.allclose(a, ref)


if __name__ == "__main__":
    test_a_store_through_a_copy_edge_naming_the_scalar_is_skewed()
