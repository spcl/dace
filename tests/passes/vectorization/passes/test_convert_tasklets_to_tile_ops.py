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


# ---- forward-analysis output transient widening -------------------------------


def test_widening_done_by_widen_accesses_pre_pass():
    """Per user direction 2026-06-09/11: widening is done PROACTIVELY by the
    unified :class:`WidenAccesses` pre-pass (replaces the prior
    ``InferBodyTransientShapes`` + ``WidenScalarsToTiles`` two-pass split),
    NOT reactively by the converter. The converter no longer widens output
    transients -- it just emits lib nodes and trusts the descriptor shapes
    set up by the pre-pass.
    """
    from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses
    sdfg2 = dace.SDFG("widen_intermediate")
    sdfg2.add_array("A", (8, ), dace.float64, transient=False)
    sdfg2.add_array("B", (8, ), dace.float64, transient=False)
    state = sdfg2.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("B", (8, ), dace.float64, transient=False)
    inner.add_array("mid_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    a_in = instate.add_access("A")
    b_in = instate.add_access("B")
    mid = instate.add_access("mid_t")
    tasklet = instate.add_tasklet("body_t", {"_a", "_b"}, {"_o"}, "_o = _a + _b")
    instate.add_edge(a_in, None, tasklet, "_a", Memlet("A[ii]"))
    instate.add_edge(b_in, None, tasklet, "_b", Memlet("B[ii]"))
    instate.add_edge(tasklet, "_o", mid, None, Memlet("mid_t[0]"))
    nsdfg = state.add_nested_sdfg(inner, {"A", "B"}, set(), symbol_mapping={"ii": "ii"})
    state.add_memlet_path(state.add_access("A"), me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(state.add_access("B"), me, nsdfg, dst_conn="B", memlet=Memlet("B[0:8]"))
    state.add_nedge(nsdfg, mx, Memlet())
    # Run the unified pre-pass -- it widens mid_t based on producer access patterns.
    WidenAccesses(widths=(8, )).apply_pass(sdfg2, {})
    desc = inner.arrays["mid_t"]
    assert tuple(desc.shape) == (8, ), \
        f"WidenAccesses should widen mid_t to (8,); got {tuple(desc.shape)}"
    # Then the converter is a pure tasklet -> lib-node rewriter (no widening, no narrowing).
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg2, {})
    # mid_t still tile-shape.
    assert tuple(inner.arrays["mid_t"].shape) == (8, )


def test_converter_leaves_length1_output_unchanged_for_all_scalar_binop():
    """When BOTH inputs are Scalar, the output transient stays length-1 (no widening).

    Per user direction: scalar-scalar / scalar-symbol / symbol-symbol op -> Scalar output."""
    sdfg = dace.SDFG("no_widen_scalar")
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body")
    inner.add_array("C", (8, ), dace.float64, transient=False)
    # Two Scalar bridges + one length-1 intermediate transient.
    inner.add_scalar("S_a", dace.float64, transient=True)
    inner.add_scalar("S_b", dace.float64, transient=True)
    inner.add_array("mid_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    sa = instate.add_access("S_a")
    sb = instate.add_access("S_b")
    mid = instate.add_access("mid_t")
    c_in = instate.add_access("C")
    binop_tasklet = instate.add_tasklet("body_t", {"_a", "_b"}, {"_o"}, "_o = _a + _b")
    instate.add_edge(sa, None, binop_tasklet, "_a", Memlet("S_a"))
    instate.add_edge(sb, None, binop_tasklet, "_b", Memlet("S_b"))
    instate.add_edge(binop_tasklet, "_o", mid, None, Memlet("mid_t[0]"))
    nsdfg = state.add_nested_sdfg(inner, set(), {"C"}, symbol_mapping={"ii": "ii"})
    state.add_nedge(me, nsdfg, Memlet())
    state.add_memlet_path(nsdfg, mx, state.add_access("C"), src_conn="C", memlet=Memlet("C[0:8]"))
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    desc = inner.arrays["mid_t"]
    assert tuple(desc.shape) == (1, ), \
        f"expected mid_t to stay length-1 for all-Scalar op (Scalar output), got {tuple(desc.shape)}"


# ---- Symbol operand kind --------------------------------------------------


def _build_inner_body_with_symbol_binop(body_str):
    """Body NSDFG with a single in-connector tasklet whose body references an outer symbol."""
    sdfg = dace.SDFG("symbol_binop_fixture")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("C", (8, ), dace.float64, transient=False)
    instate = inner.add_state("body")
    a_in = instate.add_access("A")
    c_in = instate.add_access("C")
    tasklet = instate.add_tasklet("body_t", {"_a"}, {"_o"}, body_str)
    instate.add_edge(a_in, None, tasklet, "_a", Memlet("A[ii]"))
    instate.add_edge(tasklet, "_o", c_in, None, Memlet("C[ii]"))
    nsdfg = state.add_nested_sdfg(inner, {"A"}, {"C"}, symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    c_outer = state.add_access("C")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(nsdfg, mx, c_outer, src_conn="C", memlet=Memlet("C[0:8]"))
    return sdfg, inner


def test_converter_handles_symbol_operand_on_right():
    """``_o = _a + 5`` -> TileBinop(kind_a=Tile, kind_b=Symbol, expr_b="5")."""
    sdfg, inner = _build_inner_body_with_symbol_binop("_o = _a + 5")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.kind_a == "Tile"
    assert binop.kind_b == "Symbol"
    assert binop.expr_b == "5"
    assert binop.op == "+"


def test_converter_handles_symbol_operand_on_left():
    """``_o = 2 * _a`` -> TileBinop(kind_a=Symbol, expr_a="2", kind_b=Tile)."""
    sdfg, inner = _build_inner_body_with_symbol_binop("_o = 2 * _a")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.kind_a == "Symbol"
    assert binop.kind_b == "Tile"
    assert binop.expr_a == "2"
    assert binop.op == "*"


def test_converter_handles_symbol_in_min_function():
    """``_o = min(_a, 0.5)`` -> TileBinop(op=min, kind_a=Tile, kind_b=Symbol, expr_b="0.5")."""
    sdfg, inner = _build_inner_body_with_symbol_binop("_o = min(_a, 0.5)")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.op == "min"
    assert binop.kind_a == "Tile"
    assert binop.kind_b == "Symbol"
    assert binop.expr_b == "0.5"


# ---- mask wiring on converted lib nodes ------------------------------------


def _build_body_with_mask(body_str, n_in_conns=2, has_b_arr=True):
    """Build a body NSDFG with a TileMaskGen + _tile_iter_mask AccessNode in scope,
    plus a tasklet whose body is ``body_str``. Used to verify the converter wires
    ``has_mask=True`` + ``_mask`` on the emitted Tile* lib node."""
    from dace.libraries.tileops import TileMaskGen
    sdfg = dace.SDFG("mask_fixture")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    if has_b_arr:
        sdfg.add_array("B", (8, ), dace.float64, transient=False)
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    if has_b_arr:
        inner.add_array("B", (8, ), dace.float64, transient=False)
    inner.add_array("C", (8, ), dace.float64, transient=False)
    inner.add_array("_tile_iter_mask", (8, ), dace.bool_, transient=True)
    instate = inner.add_state("body")

    # Emit TileMaskGen + mask AN inside the body (mimics GenerateTileIterationMask).
    mask_gen = TileMaskGen(name="_tile_iter_mask_gen", widths=(8, ), iter_vars=("ii", ), global_ubs=("8", ))
    instate.add_node(mask_gen)
    mask_an = instate.add_access("_tile_iter_mask")
    instate.add_edge(mask_gen, "_o", mask_an, None, Memlet("_tile_iter_mask[0:8]"))

    in_conns = {"_a", "_b"} if n_in_conns == 2 else ({"_a", "_b", "_c"} if n_in_conns == 3 else {"_a"})
    tasklet = instate.add_tasklet("body_t", in_conns, {"_o"}, body_str)
    a_in = instate.add_access("A")
    instate.add_edge(a_in, None, tasklet, "_a", Memlet("A[ii]"))
    if has_b_arr and "_b" in in_conns:
        b_in = instate.add_access("B")
        instate.add_edge(b_in, None, tasklet, "_b", Memlet("B[ii]"))
    c_in = instate.add_access("C")
    instate.add_edge(tasklet, "_o", c_in, None, Memlet("C[ii]"))

    inputs = {"A", "B"} if has_b_arr else {"A"}
    nsdfg = state.add_nested_sdfg(inner, inputs, {"C"}, symbol_mapping={"ii": "ii"})
    state.add_memlet_path(state.add_access("A"), me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    if has_b_arr:
        state.add_memlet_path(state.add_access("B"), me, nsdfg, dst_conn="B", memlet=Memlet("B[0:8]"))
    state.add_memlet_path(nsdfg, mx, state.add_access("C"), src_conn="C", memlet=Memlet("C[0:8]"))
    return sdfg, inner


def test_converter_drops_mask_on_pure_arithmetic_binop():
    """Option C (user direction 2026-06-11/12): pure-arithmetic
    :class:`TileBinop` has NO iter-mask -- the downstream :class:`TileStore`
    at the global-write boundary discards inactive lanes. Tile transients
    are register-private."""
    sdfg, inner = _build_body_with_mask("_o = _a + _b")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.has_mask is False
    mask_edges = [e for e in body_state.in_edges(binop) if e.dst_conn == "_mask"]
    assert len(mask_edges) == 0


def test_converter_drops_mask_on_pure_arithmetic_unop():
    """Option C: :class:`TileUnop` (pure arithmetic) has no iter-mask."""
    from dace.libraries.tileops import TileUnop
    sdfg, inner = _build_body_with_mask("_o = abs(_a)", n_in_conns=1, has_b_arr=False)
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    unop = next(n for n in body_state.nodes() if isinstance(n, TileUnop))
    assert unop.has_mask is False
    mask_edges = [e for e in body_state.in_edges(unop) if e.dst_conn == "_mask"]
    assert len(mask_edges) == 0


def test_converter_skips_mask_when_no_mask_in_scope():
    """A body without TileMaskGen produces has_mask=False (the divisible / unmasked case)."""
    sdfg, inner = _build_inner_body_with_binop(op="+")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.has_mask is False
    mask_edges = [e for e in body_state.in_edges(binop) if e.dst_conn == "_mask"]
    assert len(mask_edges) == 0


def test_converter_keeps_mask_producer_an_alive_for_reductions():
    """Option C: pure-arithmetic ops drop the iter-mask, but ``TileMaskGen`` still
    produces the mask transient for the side-effect-boundary consumers
    (:class:`TileReduce`, :class:`TileStore`)."""
    from dace.libraries.tileops import TileMaskGen
    sdfg, inner = _build_body_with_mask("_o = _a + _b")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    # TileMaskGen still emitted (consumed downstream by TileStore at the global write).
    assert any(isinstance(n, TileMaskGen) for n in body_state.nodes())


# ---- Symbol operand: data-independent vs lane-id-dependent -----------------


def _build_body_with_zero_in_conn(body_str, add_symbol_N=True):
    """Build a body NSDFG whose tasklet has 0 input connectors (purely Symbol-driven)."""
    sdfg = dace.SDFG("sym_only_fixture")
    sdfg.add_array("C", (8, ), dace.float64, transient=False)
    if add_symbol_N:
        sdfg.add_symbol("N", dace.int64)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body")
    inner.add_array("C", (8, ), dace.float64, transient=False)
    if add_symbol_N:
        inner.add_symbol("N", dace.int64)
    instate = inner.add_state("body")
    c_in = instate.add_access("C")
    tasklet = instate.add_tasklet("body_t", set(), {"_o"}, body_str)
    instate.add_edge(tasklet, "_o", c_in, None, Memlet("C[ii]"))
    nsdfg = state.add_nested_sdfg(inner,
                                  set(), {"C"},
                                  symbol_mapping={
                                      "ii": "ii",
                                      "N": "N"
                                  } if add_symbol_N else {"ii": "ii"})
    c_outer = state.add_access("C")
    state.add_nedge(me, nsdfg, Memlet())
    state.add_memlet_path(nsdfg, mx, c_outer, src_conn="C", memlet=Memlet("C[0:8]"))
    return sdfg, inner


def test_converter_emits_tileunop_with_symbol_invariant():
    """``_o = abs(N + 1)`` (N invariant) -> TileUnop(kind_a=Symbol, expr_a) with NO _a edge."""
    from dace.libraries.tileops import TileUnop
    sdfg, inner = _build_body_with_zero_in_conn("_o = abs(N + 1)")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    unop = next(n for n in body_state.nodes() if isinstance(n, TileUnop))
    assert unop.kind_a == "Symbol"
    assert "N" in (unop.expr_a or "")
    # No _a in-edge.
    in_edges = [e for e in body_state.in_edges(unop) if e.dst_conn == "_a"]
    assert len(in_edges) == 0


def test_converter_materialises_tile_for_lane_id_dependent_symbol_in_unop():
    """``_o = abs(ii + 1)`` -> the materialiser produces a per-lane tile; TileUnop reads
    from it as a Tile operand. Validates the user's "lane-id-symbol -> tile" path."""
    from dace.libraries.tileops import TileUnop
    sdfg, inner = _build_body_with_zero_in_conn("_o = abs(ii + 1)", add_symbol_N=False)
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    unop = next(n for n in body_state.nodes() if isinstance(n, TileUnop))
    assert unop.kind_a == "Tile", f"expected Tile (materialised lane-id), got {unop.kind_a}"
    in_edges = [e for e in body_state.in_edges(unop) if e.dst_conn == "_a"]
    assert len(in_edges) == 1, "expected an _a edge wired from the materialised tile"
    src = in_edges[0].src
    assert isinstance(src, dace.nodes.AccessNode)


def test_converter_emits_tilebinop_with_two_symbols_invariant():
    """``_o = N + 5`` (both invariant) -> TileBinop(kind_a=Symbol, kind_b=Symbol)."""
    sdfg, inner = _build_body_with_zero_in_conn("_o = N + 5")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.kind_a == "Symbol"
    assert binop.kind_b == "Symbol"


def test_converter_emits_mixed_tilebinop_with_lane_id_and_invariant_symbols():
    """``_o = ii * N`` -> lane-id ``ii`` materialised to Tile; N stays Symbol."""
    sdfg, inner = _build_body_with_zero_in_conn("_o = ii * N")
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert {binop.kind_a, binop.kind_b} == {"Tile", "Symbol"}


# ---- ``**`` / ``pow`` lowering -----------------------------------------------


@pytest.mark.parametrize("body_form", [
    "_o = _a ** _b",
    "_o = pow(_a, _b)",
])
def test_converter_emits_tilebinop_for_power(body_form):
    """Both ``_a ** _b`` and ``pow(_a, _b)`` lower to a TileBinop with op='**'.

    ``PowerOperatorExpansion`` runs upstream and rewrites integer-constant exponents
    (``x ** 2`` -> ``x * x``); only runtime exponents reach this dispatch. The lib node
    lowers ``**`` to ``std::pow`` at expansion time.
    """
    sdfg, inner = _build_inner_body_with_binop(op="+")  # placeholder, body replaced below
    # Replace the tasklet body with the requested form.
    body_state = next(s for s in inner.states())
    tasklet = next(n for n in body_state.nodes() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code = dace.properties.CodeBlock(body_form)
    ConvertTaskletsToTileOps(widths=(8, )).apply_pass(sdfg, {})
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.op == "**", f"expected op='**', got {binop.op!r}"
