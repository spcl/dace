# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Subset bounds that are SymExprs (a main and an approximate expression) read as their main expression."""

import dace
from dace import symbolic
from dace.transformation.passes.minimize_stride_permutation import score_indexed_strides


def symexpr_map() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    """``map i in 0:K`` writing ``A[i]`` through SymExpr bounds."""
    sdfg = dace.SDFG("symexpr_map")
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_array("A", ["K"], dace.float64)
    state = sdfg.add_state()
    tasklet, entry, exit_node = state.add_mapped_tasklet(
        "w", {"i": "0:K"}, {}, "o = 1", {"o": dace.Memlet("A[i]")}, external_edges=True
    )
    K = symbolic.symbol("K")
    entry.map.range = dace.subsets.Range([(symbolic.SymExpr(K - K, 0), symbolic.SymExpr(K - 1, K), 1)])
    for edge in state.edges():
        if edge.data.data == "A":
            i = symbolic.symbol("i")
            edge.data.subset = dace.subsets.Range([(symbolic.SymExpr(i, i), symbolic.SymExpr(i, i), 1)])
    return sdfg, state, entry


def test_a_symexpr_index_is_scored_by_its_main_expression():
    sdfg, state, entry = symexpr_map()
    scores = score_indexed_strides(state.edges(), sdfg, ["i"])
    assert scores["i"][0] == 1
