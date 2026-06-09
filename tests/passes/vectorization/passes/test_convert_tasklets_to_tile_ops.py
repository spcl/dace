# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`ConvertTaskletsToTileOps` (first slice: binary Tile+Tile)."""
import pytest

import dace
from dace.libraries.tileops import TileBinop
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import (ConvertTaskletsToTileOps)


def _build_inner_body_with_binop(op="+"):
    """Build an SDFG with one tile-tagged Map containing a body NSDFG whose state
    has a single binary tasklet ``_o = _a <op> _b``."""
    sdfg = dace.SDFG("binop_fixture")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    sdfg.add_array("B", (8, ), dace.float64, transient=False)
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("B", (8, ), dace.float64, transient=False)
    inner.add_array("C", (8, ), dace.float64, transient=False)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    b_inner = instate.add_access("B")
    c_inner = instate.add_access("C")
    if op in ("min", "max"):
        body_str = f"_o = {op}(_a, _b)"
    else:
        body_str = f"_o = _a {op} _b"
    tasklet = instate.add_tasklet("body_tasklet", {"_a", "_b"}, {"_o"}, body_str)
    instate.add_edge(a_inner, None, tasklet, "_a", Memlet("A[ii]"))
    instate.add_edge(b_inner, None, tasklet, "_b", Memlet("B[ii]"))
    instate.add_edge(tasklet, "_o", c_inner, None, Memlet("C[ii]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "B"}, {"C"}, symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    b_outer = state.add_access("B")
    c_outer = state.add_access("C")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(b_outer, me, nsdfg, dst_conn="B", memlet=Memlet("B[0:8]"))
    state.add_memlet_path(nsdfg, mx, c_outer, src_conn="C", memlet=Memlet("C[0:8]"))
    return sdfg, inner


@pytest.mark.parametrize("op", ["+", "-", "*", "/", "min", "max"])
def test_converter_replaces_binary_tasklet_with_tilebinop(op):
    """Each supported op gets converted; the inner state has exactly one TileBinop after."""
    sdfg, inner = _build_inner_body_with_binop(op=op)
    result = ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    body_state = next(s for s in inner.states())
    binops = [n for n in body_state.nodes() if isinstance(n, TileBinop)]
    assert len(binops) == 1, f"expected exactly one TileBinop, got {len(binops)}"
    assert binops[0].op == op


def test_converter_preserves_memlets_on_rewired_edges():
    """Rewired ``_a`` / ``_b`` / ``_c`` memlets keep their per-iteration subset shape."""
    sdfg, inner = _build_inner_body_with_binop(op="+")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    a_edge = next(e for e in body_state.in_edges(binop) if e.dst_conn == "_a")
    b_edge = next(e for e in body_state.in_edges(binop) if e.dst_conn == "_b")
    c_edge = next(e for e in body_state.out_edges(binop) if e.src_conn == "_c")
    assert str(a_edge.data) == "A[ii]"
    assert str(b_edge.data) == "B[ii]"
    assert str(c_edge.data) == "C[ii]"


def test_converter_skips_non_recognised_tasklet():
    """A tasklet with an unsupported op (e.g. ``//``) is left intact."""
    sdfg, inner = _build_inner_body_with_binop(op="+")
    # Mutate the tasklet body to an unrecognised form post-construction.
    body_state = next(s for s in inner.states())
    tasklet = next(n for n in body_state.nodes() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code = dace.properties.CodeBlock("_o = _a // _b")
    result = ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    assert result is None
    # Tasklet still present, no TileBinop.
    body_state = next(s for s in inner.states())
    assert any(isinstance(n, dace.nodes.Tasklet) for n in body_state.nodes())
    assert not any(isinstance(n, TileBinop) for n in body_state.nodes())


def test_converter_empty_sdfg_returns_none():
    """SDFG with no tile-tagged map yields zero conversions -> ``None``."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    assert ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {}) is None


def test_converter_refuses_invalid_widths():
    """Constructor refuses K outside {1, 2, 3}."""
    with pytest.raises(ValueError, match=r"widths length"):
        ConvertTaskletsToTileOps(widths=())
    with pytest.raises(ValueError, match=r"widths length"):
        ConvertTaskletsToTileOps(widths=(8, 8, 8, 8))


# ---- unary tasklet conversion (TileUnop) ----------------------------------


def _build_inner_body_with_unop(op="abs"):
    """Build an SDFG with one tile-tagged Map containing a body NSDFG whose state
    has a single unary tasklet ``_o = <op>(_a)`` (or ``_o = -_a`` for ``neg``)."""
    sdfg = dace.SDFG(f"unop_{op}_fixture")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("C", (8, ), dace.float64, transient=False)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    c_inner = instate.add_access("C")
    if op == "neg":
        body_str = "_o = -_a"
    else:
        body_str = f"_o = math.{op}(_a)"
    tasklet = instate.add_tasklet("body_tasklet", {"_a"}, {"_o"}, body_str)
    instate.add_edge(a_inner, None, tasklet, "_a", Memlet("A[ii]"))
    instate.add_edge(tasklet, "_o", c_inner, None, Memlet("C[ii]"))

    nsdfg = state.add_nested_sdfg(inner, {"A"}, {"C"}, symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    c_outer = state.add_access("C")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(nsdfg, mx, c_outer, src_conn="C", memlet=Memlet("C[0:8]"))
    return sdfg, inner


@pytest.mark.parametrize("op", ["neg", "abs", "exp", "log", "sqrt", "floor", "ceil", "tanh"])
def test_converter_replaces_unary_tasklet_with_tileunop(op):
    """Each supported unary op gets converted to a TileUnop."""
    from dace.libraries.tileops import TileUnop
    sdfg, inner = _build_inner_body_with_unop(op=op)
    result = ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    body_state = next(s for s in inner.states())
    unops = [n for n in body_state.nodes() if isinstance(n, TileUnop)]
    assert len(unops) == 1
    assert unops[0].op == op


def test_converter_unop_preserves_memlets_on_rewired_edges():
    """Rewired ``_a`` / ``_c`` memlets keep their per-iteration subset shape."""
    from dace.libraries.tileops import TileUnop
    sdfg, inner = _build_inner_body_with_unop(op="abs")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    unop = next(n for n in body_state.nodes() if isinstance(n, TileUnop))
    a_edge = next(e for e in body_state.in_edges(unop) if e.dst_conn == "_a")
    c_edge = next(e for e in body_state.out_edges(unop) if e.src_conn == "_c")
    assert str(a_edge.data) == "A[ii]"
    assert str(c_edge.data) == "C[ii]"


# ---- ternary if-then-else (TileITE) -----------------------------------------


def _build_inner_body_with_ite():
    """Build an SDFG whose body NSDFG has a ternary ``_o = _t if _cond else _e`` tasklet."""
    sdfg = dace.SDFG("ite_fixture")
    sdfg.add_array("Cond", (8, ), dace.bool_, transient=False)
    sdfg.add_array("T", (8, ), dace.float64, transient=False)
    sdfg.add_array("E", (8, ), dace.float64, transient=False)
    sdfg.add_array("O", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("Cond", (8, ), dace.bool_, transient=False)
    inner.add_array("T", (8, ), dace.float64, transient=False)
    inner.add_array("E", (8, ), dace.float64, transient=False)
    inner.add_array("O", (8, ), dace.float64, transient=False)
    instate = inner.add_state("body")
    c_inner = instate.add_access("Cond")
    t_inner = instate.add_access("T")
    e_inner = instate.add_access("E")
    o_inner = instate.add_access("O")
    tasklet = instate.add_tasklet("ite_tasklet", {"_cond", "_t", "_e"}, {"_o"}, "_o = _t if _cond else _e")
    instate.add_edge(c_inner, None, tasklet, "_cond", Memlet("Cond[ii]"))
    instate.add_edge(t_inner, None, tasklet, "_t", Memlet("T[ii]"))
    instate.add_edge(e_inner, None, tasklet, "_e", Memlet("E[ii]"))
    instate.add_edge(tasklet, "_o", o_inner, None, Memlet("O[ii]"))

    nsdfg = state.add_nested_sdfg(inner, {"Cond", "T", "E"}, {"O"}, symbol_mapping={"ii": "ii"})
    for name, an in (("Cond", state.add_access("Cond")), ("T", state.add_access("T")), ("E", state.add_access("E"))):
        state.add_memlet_path(an, me, nsdfg, dst_conn=name, memlet=Memlet(f"{name}[0:8]"))
    o_outer = state.add_access("O")
    state.add_memlet_path(nsdfg, mx, o_outer, src_conn="O", memlet=Memlet("O[0:8]"))
    return sdfg, inner


def test_converter_replaces_ternary_tasklet_with_tileite():
    """A Python ternary ``_o = _t if _cond else _e`` gets converted to a TileITE."""
    from dace.libraries.tileops import TileITE
    sdfg, inner = _build_inner_body_with_ite()
    result = ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    body_state = next(s for s in inner.states())
    ites = [n for n in body_state.nodes() if isinstance(n, TileITE)]
    assert len(ites) == 1


def test_converter_ite_wires_cond_t_e_connectors_correctly():
    """``_cond`` / ``_t`` / ``_e`` end up on the right connectors with the right memlets."""
    from dace.libraries.tileops import TileITE
    sdfg, inner = _build_inner_body_with_ite()
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    ite = next(n for n in body_state.nodes() if isinstance(n, TileITE))
    cond_edge = next(e for e in body_state.in_edges(ite) if e.dst_conn == "_cond")
    t_edge = next(e for e in body_state.in_edges(ite) if e.dst_conn == "_t")
    e_edge = next(e for e in body_state.in_edges(ite) if e.dst_conn == "_e")
    out_edge = next(e for e in body_state.out_edges(ite) if e.src_conn == "_o")
    assert str(cond_edge.data) == "Cond[ii]"
    assert str(t_edge.data) == "T[ii]"
    assert str(e_edge.data) == "E[ii]"
    assert str(out_edge.data) == "O[ii]"


# ---- operand-kind detection (Scalar broadcast) ---------------------------


def _build_inner_body_with_tile_plus_scalar(op="+"):
    """Binop with tile-shape left + Scalar right (the post-walker shape for stage_constant)."""
    sdfg = dace.SDFG("mixed_kind_fixture")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("C", (8, ), dace.float64, transient=False)
    inner.add_array("A_tile", (8, ), dace.float64, transient=True)
    inner.add_scalar("S_bridge", dace.float64, transient=True)
    instate = inner.add_state("body")
    tile_an = instate.add_access("A_tile")
    scalar_an = instate.add_access("S_bridge")
    c_inner = instate.add_access("C")
    tasklet = instate.add_tasklet("body_tasklet", {"_a", "_b"}, {"_o"}, f"_o = _a {op} _b")
    instate.add_edge(tile_an, None, tasklet, "_a", Memlet("A_tile[ii]"))
    instate.add_edge(scalar_an, None, tasklet, "_b", Memlet("S_bridge"))
    instate.add_edge(tasklet, "_o", c_inner, None, Memlet("C[ii]"))

    nsdfg = state.add_nested_sdfg(inner, {"A"}, {"C"}, symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    c_outer = state.add_access("C")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(nsdfg, mx, c_outer, src_conn="C", memlet=Memlet("C[0:8]"))
    return sdfg, inner


def test_converter_classifies_scalar_bridge_operand_as_scalar_kind():
    """Binop right-reads a Scalar transient -> TileBinop.kind_b == "Scalar"."""
    sdfg, inner = _build_inner_body_with_tile_plus_scalar(op="+")
    result = ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.kind_a == "Tile"
    assert binop.kind_b == "Scalar"


def test_converter_default_tile_kind_when_both_sources_are_tile_shape():
    """Both reads from tile-shape -> both kinds default to Tile."""
    sdfg, inner = _build_inner_body_with_binop(op="+")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.kind_a == "Tile"
    assert binop.kind_b == "Tile"


def test_converter_unop_with_scalar_source_sets_scalar_kind():
    """Unary op reading a Scalar bridge -> TileUnop.kind_a == "Scalar"."""
    from dace.libraries.tileops import TileUnop
    sdfg = dace.SDFG("unop_scalar_fixture")
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body")
    inner.add_array("C", (8, ), dace.float64, transient=False)
    inner.add_scalar("S_bridge", dace.float64, transient=True)
    instate = inner.add_state("body")
    scalar_an = instate.add_access("S_bridge")
    c_inner = instate.add_access("C")
    tasklet = instate.add_tasklet("body_tasklet", {"_a"}, {"_o"}, "_o = abs(_a)")
    instate.add_edge(scalar_an, None, tasklet, "_a", Memlet("S_bridge"))
    instate.add_edge(tasklet, "_o", c_inner, None, Memlet("C[ii]"))
    nsdfg = state.add_nested_sdfg(inner, set(), {"C"}, symbol_mapping={"ii": "ii"})
    c_outer = state.add_access("C")
    state.add_memlet_path(me, nsdfg, memlet=Memlet())
    state.add_memlet_path(nsdfg, mx, c_outer, src_conn="C", memlet=Memlet("C[0:8]"))

    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    unop = next(n for n in body_state.nodes() if isinstance(n, TileUnop))
    assert unop.kind_a == "Scalar"


# ---- reduction (TileReduce) ---------------------------------------------


def _build_inner_body_with_reduction(op="+"):
    """Build a body NSDFG with an in-place RMW accumulator tasklet ``_acc = _acc <op> _val``."""
    _OP_TAG = {"+": "add", "*": "mul", "min": "min", "max": "max"}
    sdfg = dace.SDFG(f"reduce_{_OP_TAG[op]}_fixture")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    sdfg.add_array("Acc", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("Acc", (1, ), dace.float64, transient=False)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    acc_inner_in = instate.add_access("Acc")
    acc_inner_out = instate.add_access("Acc")
    if op in ("min", "max"):
        body_str = f"_acc = {op}(_acc, _val)"
    else:
        body_str = f"_acc = _acc {op} _val"
    tasklet = instate.add_tasklet("body_tasklet", {"_acc", "_val"}, {"_acc"}, body_str)
    instate.add_edge(a_inner, None, tasklet, "_val", Memlet("A[ii]"))
    instate.add_edge(acc_inner_in, None, tasklet, "_acc", Memlet("Acc[0]"))
    instate.add_edge(tasklet, "_acc", acc_inner_out, None, Memlet("Acc[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "Acc"}, {"Acc"}, symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    acc_outer_in = state.add_access("Acc")
    acc_outer_out = state.add_access("Acc")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(acc_outer_in, me, nsdfg, dst_conn="Acc", memlet=Memlet("Acc[0]"))
    state.add_memlet_path(nsdfg, mx, acc_outer_out, src_conn="Acc", memlet=Memlet("Acc[0]"))
    return sdfg, inner


@pytest.mark.parametrize("op", ["+", "*", "min", "max"])
def test_converter_replaces_inplace_rmw_tasklet_with_tilereduce(op):
    """Each supported associative op gets converted to a TileReduce."""
    from dace.libraries.tileops import TileReduce
    sdfg, inner = _build_inner_body_with_reduction(op=op)
    result = ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    body_state = next(s for s in inner.states())
    reduces = [n for n in body_state.nodes() if isinstance(n, TileReduce)]
    assert len(reduces) == 1
    assert reduces[0].op == op


def test_converter_reduction_wires_src_and_dst_correctly():
    """``_val`` (tile input) -> TileReduce._src; ``_acc`` (accumulator) -> TileReduce._dst."""
    from dace.libraries.tileops import TileReduce
    sdfg, inner = _build_inner_body_with_reduction(op="+")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    reduce_node = next(n for n in body_state.nodes() if isinstance(n, TileReduce))
    src_edge = next(e for e in body_state.in_edges(reduce_node) if e.dst_conn == "_src")
    dst_edge = next(e for e in body_state.out_edges(reduce_node) if e.src_conn == "_dst")
    assert str(src_edge.data) == "A[ii]"
    assert str(dst_edge.data) == "Acc[0]"


def test_converter_skips_non_inplace_binop():
    """Non-RMW binop ``_o = _a + _b`` is NOT recognised as a reduction; stays a TileBinop."""
    from dace.libraries.tileops import TileBinop, TileReduce
    sdfg, inner = _build_inner_body_with_binop(op="+")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    assert not any(isinstance(n, TileReduce) for n in body_state.nodes())
    assert any(isinstance(n, TileBinop) for n in body_state.nodes())
