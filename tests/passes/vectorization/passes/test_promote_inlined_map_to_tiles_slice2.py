# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for slice 2 of :class:`PromoteInlinedMapToTiles` -- the binop /
unop tasklet -> :class:`TileBinop` / :class:`TileUnop` rewrite.

These tests are pre-design-freeze: they pin the helper's current contract
(``_operand_kind`` resolves to Tile or Symbol; refuses everything else with
NotImplementedError) so the upcoming vocabulary rename + G3 partial-index
extension can be measured against a baseline. End-to-end numerical parity
is the slice-4 deliverable.
"""
import pytest

import dace
from dace import data, nodes
from dace.libraries.tileops import TileBinop, TileUnop
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.promote_inlined_map_to_tiles import (
    promote_binop_tasklet_to_tile_binop,
    promote_tasklets_to_tile_ops,
    promote_unop_tasklet_to_tile_unop,
    widen_body_scalars_to_tile,
)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


def _spec(widths=(4, 8)):
    return TileDimSpec(iter_vars=("i", "j"), widths=tuple(widths), global_ubs=("M", "N"))


def _count(state, cls):
    return sum(1 for n in state.nodes() if isinstance(n, cls))


def _build_binop_fixture(rhs="_in1 + _in2", widths=(4, 8)):
    """Build a Map whose body has a binop tasklet between two tile-shaped
    transients."""
    sdfg = dace.SDFG("binop_outer")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    sdfg.add_array("B", (16, 32), dace.float64, transient=False)
    sdfg.add_array("C", (16, 32), dace.float64, transient=False)
    sdfg.add_array("t1", widths, dace.float64, transient=True)
    sdfg.add_array("t2", widths, dace.float64, transient=True)
    sdfg.add_array("t3", widths, dace.float64, transient=True)
    state = sdfg.add_state("s")
    me, mx = state.add_map("m", {"i": "0:16:4", "j": "0:32:8"})
    a, b, c = state.add_access("A"), state.add_access("B"), state.add_access("C")
    t1, t2, t3 = state.add_access("t1"), state.add_access("t2"), state.add_access("t3")
    # Load into t1/t2 (the global side will be picked up by slice 4); for this test
    # we wire them as direct AN->AN edges that mimic the post-widening shape.
    state.add_memlet_path(a, me, t1, memlet=Memlet("A[i:i+4, j:j+8]"))
    state.add_memlet_path(b, me, t2, memlet=Memlet("B[i:i+4, j:j+8]"))
    binop = state.add_tasklet("compute", {"_in1", "_in2"}, {"_out"}, f"_out = {rhs}")
    state.add_edge(t1, None, binop, "_in1", Memlet("t1[0:4, 0:8]"))
    state.add_edge(t2, None, binop, "_in2", Memlet("t2[0:4, 0:8]"))
    state.add_edge(binop, "_out", t3, None, Memlet("t3[0:4, 0:8]"))
    state.add_memlet_path(t3, mx, c, memlet=Memlet("C[i:i+4, j:j+8]"))
    return sdfg, state, me, binop


def test_binop_tile_tile_rewrites_to_tilebinop():
    """Both operands are tile-shaped transients -> TileBinop with kind_a=kind_b=Tile."""
    sdfg, state, me, binop = _build_binop_fixture(rhs="_in1 + _in2")
    assert promote_binop_tasklet_to_tile_binop(state, binop, widths=(4, 8))
    assert _count(state, TileBinop) == 1
    (tb, ) = [n for n in state.nodes() if isinstance(n, TileBinop)]
    assert tb.kind_a == "Tile" and tb.kind_b == "Tile"
    assert tb.op == "+"


def test_binop_tile_symbol_literal_rewrites_with_symbol_kind():
    """Numeric literal RHS -> kind_b=Symbol, expr_b='1.0'."""
    sdfg, state, me, binop = _build_binop_fixture(rhs="_in1 + 1.0")
    assert promote_binop_tasklet_to_tile_binop(state, binop, widths=(4, 8))
    (tb, ) = [n for n in state.nodes() if isinstance(n, TileBinop)]
    assert tb.kind_a == "Tile"
    assert tb.kind_b == "Symbol"
    assert tb.expr_b == "1.0"


def test_binop_refuses_non_tile_output():
    """Output AN's descriptor must match widths -- otherwise refuse (slice 4 handles globals)."""
    sdfg, state, me, binop = _build_binop_fixture()
    # Re-target output to a global Array instead of a tile transient.
    out_edge = next(e for e in state.out_edges(binop) if e.src_conn == "_out")
    c_node = next(n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == "t3")
    # Replace t3's descriptor with a non-tile shape.
    sdfg.remove_data("t3", validate=False)
    sdfg.add_array("t3", (16, 32), dace.float64, transient=True)
    with pytest.raises(NotImplementedError, match="output"):
        promote_binop_tasklet_to_tile_binop(state, binop, widths=(4, 8))


def test_binop_refuses_no_tile_operand():
    """All-Symbol binop refuses -- TileBinop requires at least one Tile."""
    sdfg = dace.SDFG("all_symbol")
    sdfg.add_array("C", (16, 32), dace.float64, transient=False)
    sdfg.add_array("t3", (4, 8), dace.float64, transient=True)
    state = sdfg.add_state("s")
    me, mx = state.add_map("m", {"i": "0:16:4", "j": "0:32:8"})
    t3 = state.add_access("t3")
    c = state.add_access("C")
    binop = state.add_tasklet("compute", set(), {"_out"}, "_out = 1.0 + 2.0")
    state.add_edge(me, None, binop, None, Memlet())
    state.add_edge(binop, "_out", t3, None, Memlet("t3[0:4, 0:8]"))
    state.add_memlet_path(t3, mx, c, memlet=Memlet("C[i:i+4, j:j+8]"))
    with pytest.raises(NotImplementedError, match="no Tile"):
        promote_binop_tasklet_to_tile_binop(state, binop, widths=(4, 8))


def test_unop_tile_neg_rewrites_to_tileunop():
    """``_out = -_in`` with Tile input -> TileUnop op='neg'."""
    sdfg = dace.SDFG("unop_outer")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    sdfg.add_array("C", (16, 32), dace.float64, transient=False)
    sdfg.add_array("t1", (4, 8), dace.float64, transient=True)
    sdfg.add_array("t3", (4, 8), dace.float64, transient=True)
    state = sdfg.add_state("s")
    me, mx = state.add_map("m", {"i": "0:16:4", "j": "0:32:8"})
    a, c = state.add_access("A"), state.add_access("C")
    t1, t3 = state.add_access("t1"), state.add_access("t3")
    state.add_memlet_path(a, me, t1, memlet=Memlet("A[i:i+4, j:j+8]"))
    unop = state.add_tasklet("neg", {"_in"}, {"_out"}, "_out = -_in")
    state.add_edge(t1, None, unop, "_in", Memlet("t1[0:4, 0:8]"))
    state.add_edge(unop, "_out", t3, None, Memlet("t3[0:4, 0:8]"))
    state.add_memlet_path(t3, mx, c, memlet=Memlet("C[i:i+4, j:j+8]"))
    assert promote_unop_tasklet_to_tile_unop(state, unop, widths=(4, 8))
    assert _count(state, TileUnop) == 1
    (tu, ) = [n for n in state.nodes() if isinstance(n, TileUnop)]
    assert tu.op == "neg"


def test_promote_tasklets_walks_scope_and_counts():
    """The walker rewrites both binop and unop tasklets in one sweep."""
    sdfg, state, me, _ = _build_binop_fixture(rhs="_in1 * _in2")
    counts = promote_tasklets_to_tile_ops(state, me, _spec())
    assert counts == {"binop": 1, "unop": 0}
