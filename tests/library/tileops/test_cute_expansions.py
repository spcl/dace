# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Print-paste tests for the T9 ``cute`` expansions.

Each tile-op lib node has a ``cute`` expansion that emits a Python
tasklet whose body uses ``cuda.tile.*`` primitives via the ``ct``
short alias. The emitted shape matches the reference cuTile kernels
in the user's ``manual_cutile_simple.py`` and ``manual_cutile_
masked.py`` documents: ``__pid<k> = ct.bid(k)`` preamble,
``ct.load(... padding_mode=ct.PaddingMode.ZERO)`` for loads, bare
element-wise op for binops (mask applied at store), and
``ct.scatter`` for masked stores with per-lane indices.

These tests check that calling the expansion produces a Python
tasklet whose body parses as valid Python and contains the expected
``ct.*`` call shape. The cuTile-Python runtime is NOT executed
(no GPU + cuTile install required on CI).

DaCe's Python tasklet pipeline parses + unparses the body so trailing
tuple commas are dropped (``(__pid0, __pid1,)`` → ``(__pid0, __pid1)``)
and binary-op rhs gets wrapped in parens (``a + b`` → ``(a + b)``); the
assertions below match the post-round-trip shape.
"""
import ast

import dace
from dace.libraries.tileops import TileBinop, TileLoad, TileMaskGen, TileStore


def _expand_cute(lib_node):
    """Return ``(body, language)`` from the lib node's ``cute`` expansion."""
    sdfg = dace.SDFG(f"cute_smoke_{lib_node.label}")
    state = sdfg.add_state("main")
    state.add_node(lib_node)
    cls = lib_node.implementations["cute"]
    tasklet = cls.expansion(lib_node, state, sdfg)
    return tasklet.code.as_string, tasklet.language


def _assert_parses_as_python(body: str) -> None:
    """Confirm ``body`` is a valid Python statement / expression."""
    ast.parse(body)


def test_tile_load_cute_emits_block_id_and_ct_load_with_padding():
    """K=1 TileLoad cute body: ``__pid0 = ct.bid(0)`` + ``ct.load(...)``."""
    body, lang = _expand_cute(TileLoad(name="L", widths=(8,)))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.load(__src, index=(__pid0,)" in body
    assert "shape=(8,)" in body
    assert "padding_mode=ct.PaddingMode.ZERO" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_load_cute_K2_emits_two_block_ids():
    """K=2 TileLoad cute body has ``__pid0`` and ``__pid1``."""
    body, _ = _expand_cute(TileLoad(name="L", widths=(4, 8)))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "__pid1 = ct.bid(1)" in body
    assert "index=(__pid0, __pid1)" in body
    assert "shape=(4, 8)" in body


def test_tile_store_cute_unmasked_emits_ct_store():
    """Unmasked TileStore: ``ct.store(__output, index=(__pid0,), tile=__src)``."""
    body, _ = _expand_cute(TileStore(name="S", widths=(8,)))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.store(__output, index=(__pid0,)" in body
    assert "tile=__src" in body
    assert "ct.scatter" not in body


def test_tile_store_cute_masked_emits_ct_scatter_with_arange_indices():
    """Masked TileStore: ``ct.scatter(__output, (__idx0,), __src, mask=__mask)``
    with per-lane indices ``__idx_k = ct.arange(W_k) + __pid_k * W_k``."""
    body, _ = _expand_cute(TileStore(name="S", widths=(8,), has_mask=True))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 8" in body
    assert "ct.scatter(__output, (__idx0,), __src, mask=__mask)" in body


def test_tile_store_cute_K2_masked_scatter_has_two_idx_tiles():
    """Masked K=2 store emits two per-lane index tiles + scatter."""
    body, _ = _expand_cute(TileStore(name="S", widths=(4, 8), has_mask=True))
    _assert_parses_as_python(body)
    assert "ct.arange(4, dtype=ct.int32)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 4" in body
    assert "__pid1 * 8" in body
    assert "ct.scatter(__output, (__idx0, __idx1), __src, mask=__mask)" in body


def test_tile_binop_cute_emits_bare_elementwise_op():
    """TileBinop cute body emits the operator inline; no ``ct.where`` wrap
    (mask is applied at the scatter store, not at the binop)."""
    body, _ = _expand_cute(TileBinop(name="B", widths=(8,), op="+"))
    _assert_parses_as_python(body)
    assert "__rhs1 + __rhs2" in body
    assert "ct.where" not in body


def test_tile_binop_cute_masked_still_bare_op():
    """Masked TileBinop does NOT wrap with ``ct.where`` — mask flows to the
    store. ``has_mask=True`` drops the ``__mask`` input on the cute body
    because the binop never reads it."""
    body, _ = _expand_cute(TileBinop(name="B", widths=(8,), op="*", has_mask=True))
    _assert_parses_as_python(body)
    assert "__rhs1 * __rhs2" in body
    assert "ct.where" not in body


def test_tile_binop_cute_symbol_operand_inlines_expr():
    """Symbol-kind RHS embeds the expression literally."""
    body, _ = _expand_cute(
        TileBinop(name="B", widths=(8,), op="+", kind_b="Symbol", expr_b="alpha")
    )
    _assert_parses_as_python(body)
    assert "alpha" in body


def test_tile_binop_cute_uses_ct_minimum_for_min():
    """``min`` op routes to ``ct.minimum``."""
    body, _ = _expand_cute(TileBinop(name="B", widths=(4, 8), op="min"))
    _assert_parses_as_python(body)
    assert "ct.minimum(__rhs1, __rhs2)" in body


def test_tile_mask_gen_cute_1d_uses_arange_and_bid():
    """K=1 mask: ``__offsets0 = ct.arange(W)`` + ``__mask0 = ... < ub``;
    output is the single per-dim mask (no ``&`` combinator)."""
    body, _ = _expand_cute(
        TileMaskGen(name="M", widths=(8,), iter_vars=("i",), global_ubs=("N_ub",))
    )
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 8" in body
    assert "N_ub" in body
    assert "__output = __mask0" in body
    assert "&" not in body


def test_tile_mask_gen_cute_K2_combines_per_dim_via_broadcast_and_amp():
    """K=2 mask combines two per-dim masks via ``broadcast_to`` and ``&``."""
    body, _ = _expand_cute(
        TileMaskGen(name="M", widths=(4, 8), iter_vars=("i", "j"), global_ubs=("M_ub", "N_ub"))
    )
    _assert_parses_as_python(body)
    assert "ct.arange(4" in body
    assert "ct.arange(8" in body
    assert "__pid0 * 4" in body
    assert "__pid1 * 8" in body
    assert "ct.broadcast_to(__mask0[:, None], (4, 8))" in body
    assert "ct.broadcast_to(__mask1[None, :], (4, 8))" in body
    assert " & " in body
