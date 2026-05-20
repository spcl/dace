# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Print-paste tests for the T9 ``cute`` expansions.

Each tile-op lib node has a ``cute`` expansion that emits a Python
tasklet whose body uses ``cuda.tile.*`` primitives. These tests check
that calling the expansion produces a Python tasklet whose body parses
as valid Python and contains the expected ``cuda.tile`` call shape.

The cuTile-Python runtime is NOT executed (no GPU + cuTile install
required on CI); the tests just verify the emitted snippet for
inspection / future GPU lowering.
"""
import ast

import dace
from dace.libraries.tileops import TileBinop, TileLoad, TileMaskGen, TileStore


def _expand_cute(lib_node):
    """Return the Python body string from the lib node's ``cute`` expansion."""
    sdfg = dace.SDFG(f"cute_smoke_{lib_node.label}")
    state = sdfg.add_state("main")
    state.add_node(lib_node)
    cls = lib_node.implementations["cute"]
    tasklet = cls.expansion(lib_node, state, sdfg)
    return tasklet.code.as_string, tasklet.language


def _assert_parses_as_python(body: str) -> None:
    """Confirm ``body`` is a valid Python statement / expression."""
    ast.parse(body)


def test_tile_load_cute_emits_cuda_tile_load():
    """Unmasked TileLoad cute body calls ``cuda.tile.load(...)``."""
    body, lang = _expand_cute(TileLoad(name="L", widths=(8,)))
    _assert_parses_as_python(body)
    assert "cuda.tile.load(" in body
    assert "shape=(8,)" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_load_cute_masked_threads_mask_parameter():
    """Masked TileLoad cute body passes ``mask=__mask`` to ``cuda.tile.load``."""
    body, _ = _expand_cute(TileLoad(name="L", widths=(4, 8), has_mask=True))
    _assert_parses_as_python(body)
    assert "cuda.tile.load(" in body
    assert "mask=__mask" in body
    assert "shape=(4, 8)" in body


def test_tile_store_cute_emits_cuda_tile_store():
    """TileStore cute body calls ``cuda.tile.store(...)``."""
    body, _ = _expand_cute(TileStore(name="S", widths=(8,), has_mask=True))
    _assert_parses_as_python(body)
    assert "cuda.tile.store(" in body
    assert "mask=__mask" in body


def test_tile_binop_cute_emits_elementwise_op():
    """Tile/Tile TileBinop cute body emits the operator inline."""
    body, _ = _expand_cute(TileBinop(name="B", widths=(8,), op="+"))
    _assert_parses_as_python(body)
    assert "__rhs1 + __rhs2" in body


def test_tile_binop_cute_emits_where_when_masked():
    """Masked TileBinop cute body wraps the result in ``cuda.tile.where``."""
    body, _ = _expand_cute(TileBinop(name="B", widths=(8,), op="*", has_mask=True))
    _assert_parses_as_python(body)
    assert "cuda.tile.where(__mask," in body


def test_tile_binop_cute_symbol_operand_inlines_expr():
    """Symbol-kind RHS embeds the expression literally."""
    body, _ = _expand_cute(
        TileBinop(name="B", widths=(8,), op="+", kind_b="Symbol", expr_b="alpha")
    )
    _assert_parses_as_python(body)
    assert "alpha" in body


def test_tile_binop_cute_uses_cuda_tile_minimum_for_min():
    """``min`` op routes to ``cuda.tile.minimum``."""
    body, _ = _expand_cute(TileBinop(name="B", widths=(4, 8), op="min"))
    _assert_parses_as_python(body)
    assert "cuda.tile.minimum(" in body


def test_tile_mask_gen_cute_emits_per_dim_broadcasts():
    """TileMaskGen cute body builds per-dim ``cuda.tile.arange`` +
    broadcast + ``&`` for K=2."""
    body, _ = _expand_cute(
        TileMaskGen(name="M", widths=(4, 8), iter_vars=("i", "j"), global_ubs=("M_ub", "N_ub"))
    )
    _assert_parses_as_python(body)
    assert "cuda.tile.arange(4)" in body
    assert "cuda.tile.arange(8)" in body
    assert "cuda.tile.broadcast_to" in body
    assert " & " in body


def test_tile_mask_gen_cute_1d_no_unnecessary_broadcast_combinator():
    """K=1 emits a single broadcast — no ``&`` combinator."""
    body, _ = _expand_cute(
        TileMaskGen(name="M", widths=(8,), iter_vars=("i",), global_ubs=("N_ub",))
    )
    _assert_parses_as_python(body)
    assert "cuda.tile.arange(8)" in body
    assert "cuda.tile.broadcast_to" in body
    assert "&" not in body
