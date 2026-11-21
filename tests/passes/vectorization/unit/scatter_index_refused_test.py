# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scatter whose index has no tile-op lowering leaves its map scalar, as a gather does (vloc_psi_k_acc's
``psic[nl[igk[i]]] = ...``, bound through interstate symbols, raised NotImplementedError out of the vectorizer)."""

import warnings

import numpy as np

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")
ARRAYS = (("A", dace.float64), ("B", dace.float64), ("igk", dace.int64), ("nl", dace.int64))


def scatter_through_symbols() -> dace.SDFG:
    """``for i in map(0:N): j = igk[i]; k = nl[j]; B[k] = A[i] + 1`` with ``j`` and ``k`` interstate symbols."""
    inner = dace.SDFG("scatter_body")
    for name, dtype in ARRAYS:
        inner.add_array(name, [N], dtype)
    for name in ("i", "j", "k"):
        inner.add_symbol(name, dace.int64)
    s0 = inner.add_state("s0", is_start_block=True)
    s1 = inner.add_state("s1")
    s2 = inner.add_state("s2")
    inner.add_edge(s0, s1, dace.InterstateEdge(assignments={"j": "igk[i]"}))
    inner.add_edge(s1, s2, dace.InterstateEdge(assignments={"k": "nl[j]"}))
    tasklet = s2.add_tasklet("inc", {"a"}, {"b"}, "b = a + 1")
    s2.add_edge(s2.add_access("A"), None, tasklet, "a", dace.Memlet("A[i]"))
    s2.add_edge(tasklet, "b", s2.add_access("B"), None, dace.Memlet("B[k]"))

    sdfg = dace.SDFG("scatter_through_symbols")
    for name, dtype in ARRAYS:
        sdfg.add_array(name, [N], dtype)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:N"})
    node = state.add_nested_sdfg(inner, {"A", "igk", "nl"}, {"B"}, symbol_mapping={"N": "N", "i": "i"})
    for name in ("A", "igk", "nl"):
        state.add_memlet_path(state.add_access(name), entry, node, dst_conn=name, memlet=dace.Memlet(f"{name}[0:N]"))
    state.add_memlet_path(node, exit_, state.add_access("B"), src_conn="B", memlet=dace.Memlet("B[0:N]"))
    return sdfg


def test_unlowerable_scatter_index_leaves_the_map_scalar():
    sdfg = scatter_through_symbols()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VectorizeCPUMultiDim(VectorizeConfig(widths=(4,), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    assert any("un-tiled" in str(w.message) and "scatter index" in str(w.message) for w in caught)
    n = 13
    a = np.arange(n, dtype=np.float64)
    igk = np.arange(n)[::-1].copy()
    nl = (np.arange(n) * 5) % n
    b = np.zeros(n)
    sdfg(A=a, B=b, igk=igk, nl=nl, N=n)
    ref = np.zeros(n)
    ref[nl[igk]] = a + 1
    np.testing.assert_array_equal(b, ref)


if __name__ == "__main__":
    test_unlowerable_scatter_index_leaves_the_map_scalar()
    print("ok")
