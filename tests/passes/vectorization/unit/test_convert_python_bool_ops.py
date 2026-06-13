# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-syntax operator detection in :class:`ConvertTaskletsToTileOps`.

Lift passes (notably ``SameWriteSetIfElseToITECFG``'s ``combine_cond``)
emit raw-Python tasklet bodies — ``_o = (_a or _b)``, ``_o = (not _a)``.
After ``WidenAccesses`` widens the connectors to a register tile, these
bodies must convert to per-lane tile ops; otherwise codegen emits a scalar
``bool`` assigned to a ``bool*`` register tile, a hard C++ compile error.

These tests pin the detector contract:

* ``or`` -> :class:`TileBinop` ``op='||'``
* ``and`` -> :class:`TileBinop` ``op='&&'``
* ``not`` -> :class:`TileUnop` ``op='not'`` (lowers to C ``!``)
* ``@`` (matmul) -> refused (not a per-lane elementwise op)
"""
import dace
import pytest

from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import (ConvertTaskletsToTileOps,
                                                                                   _normalize_python_tasklet_body)


def _tasklet(in_conns, out_conn, code):
    sdfg = dace.SDFG("t")
    state = sdfg.add_state("s", is_start_block=True)
    return state.add_tasklet("tk", set(in_conns), {out_conn}, code)


@pytest.mark.parametrize("body,expected", [
    ("_o = (_c_0 or _c_1)", "_o = (_c_0 || _c_1)"),
    ("_o = _c_0 and _c_1", "_o = _c_0 && _c_1"),
    ("_o = (not _c_0)", "_o = (not _c_0)"),  # not is unary; left for _detect_unop
    ("_o = horizontal_or", "_o = horizontal_or"),  # word-boundary: substring 'or' untouched
])
def test_normalize_python_tasklet_body(body, expected):
    assert _normalize_python_tasklet_body(body) == expected


def test_normalize_refuses_matmul():
    assert _normalize_python_tasklet_body("_o = _a @ _b") is None


def test_detect_binop_or():
    t = _tasklet(["_c_0", "_c_1"], "_o", "_o = (_c_0 or _c_1)")
    res = ConvertTaskletsToTileOps()._detect_binop(t)
    assert res is not None
    _out, _a, _b, op = res
    assert op == "||"


def test_detect_binop_and():
    t = _tasklet(["_c_0", "_c_1"], "_o", "_o = (_c_0 and _c_1)")
    res = ConvertTaskletsToTileOps()._detect_binop(t)
    assert res is not None
    _out, _a, _b, op = res
    assert op == "&&"


@pytest.mark.parametrize("code", ["_o = not _c_0", "_o = (not _c_0)"])
def test_detect_unop_not(code):
    t = _tasklet(["_c_0"], "_o", code)
    res = ConvertTaskletsToTileOps()._detect_unop(t)
    assert res is not None
    _out, _a, op = res
    assert op == "not"


def test_detect_binop_refuses_matmul():
    t = _tasklet(["_a", "_b"], "_o", "_o = _a @ _b")
    assert ConvertTaskletsToTileOps()._detect_binop(t) is None


def test_tile_unop_not_constructs_and_lowers_to_bang():
    """``TileUnop(op='not')`` is a valid node and its op-char is ``!``."""
    from dace.libraries.tileops import TileUnop
    from dace.libraries.tileops._isa_codegen import _UNOP_TO_CHAR
    u = TileUnop(name="n", widths=(8, ), op="not", kind_a="Tile")
    assert u.op == "not"
    assert _UNOP_TO_CHAR["not"] == "!"


def _state_with_mask_edge(mask_dtype):
    """Build a state with a TileBinop whose ``_mask`` connector is fed by an
    array of ``mask_dtype``."""
    from dace.libraries.tileops import TileBinop
    sdfg = dace.SDFG("m")
    sdfg.add_array("a", [8], dtype=dace.float64, transient=True)
    sdfg.add_array("b", [8], dtype=dace.float64, transient=True)
    sdfg.add_array("c", [8], dtype=dace.float64, transient=True)
    sdfg.add_array("msk", [8], dtype=mask_dtype, transient=True)
    st = sdfg.add_state("s", is_start_block=True)
    binop = TileBinop(name="bp", widths=(8, ), op="+", has_mask=True)
    st.add_node(binop)
    st.add_edge(st.add_access("a"), None, binop, "_a", dace.Memlet("a[0:8]"))
    st.add_edge(st.add_access("b"), None, binop, "_b", dace.Memlet("b[0:8]"))
    st.add_edge(st.add_access("msk"), None, binop, "_mask", dace.Memlet("msk[0:8]"))
    st.add_edge(binop, "_c", st.add_access("c"), None, dace.Memlet("c[0:8]"))
    return sdfg


def test_mask_connectors_are_bool_accepts_bool():
    from dace.transformation.passes.vectorization.utils.pass_invariants import mask_connectors_are_bool
    sdfg = _state_with_mask_edge(dace.bool_)
    assert mask_connectors_are_bool(sdfg) is None


def test_mask_connectors_are_bool_rejects_non_bool():
    from dace.transformation.passes.vectorization.utils.pass_invariants import mask_connectors_are_bool
    sdfg = _state_with_mask_edge(dace.float64)
    violation = mask_connectors_are_bool(sdfg)
    assert violation is not None
    assert "must be bool" in violation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
