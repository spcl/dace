# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Print-paste tests for the T9 ``cutile`` expansions.

Each tile-op lib node has a ``cutile`` expansion that emits a Python
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

import pytest

import dace
from dace.libraries.tileops import (TileBinop, TileGather, TileLoad, TileMaskGen, TileMerge, TileReduce, TileStore)
from dace.libraries.tileops.nodes import tile_merge as _tile_merge_mod
from dace.libraries.tileops.nodes import tile_reduce as _tile_reduce_mod


def _expand_cutile(lib_node):
    """Return ``(body, language)`` from the lib node's ``cutile`` expansion."""
    sdfg = dace.SDFG(f"cutile_smoke_{lib_node.label}")
    state = sdfg.add_state("main")
    state.add_node(lib_node)
    cls = lib_node.implementations["cutile"]
    tasklet = cls.expansion(lib_node, state, sdfg)
    return tasklet.code.as_string, tasklet.language


def _expand_cutile_tasklet(lib_node):
    """Return the raw tasklet from the lib node's ``cutile`` expansion."""
    sdfg = dace.SDFG(f"cutile_smoke_{lib_node.label}")
    state = sdfg.add_state("main")
    state.add_node(lib_node)
    cls = lib_node.implementations["cutile"]
    return cls.expansion(lib_node, state, sdfg)


def _expand_merge_cutile_with_dtype(lib_node, out_dtype):
    """Expand a :class:`TileMerge` ``cutile`` body with a wired ``_o`` output of
    ``out_dtype`` (the fallback path reads the output dtype off the edge)."""
    sdfg = dace.SDFG(f"cutile_merge_{lib_node.label}")
    state = sdfg.add_state("main")
    n = len(lib_node.widths)
    shape = tuple(lib_node.widths) if n > 1 else (lib_node.widths[0], )
    for name in ("cond", "then", "else", "out"):
        sdfg.add_array(name, shape, out_dtype, transient=(name != "out"))
    state.add_node(lib_node)
    cond = state.add_access("cond")
    then = state.add_access("then")
    els = state.add_access("else")
    out = state.add_access("out")
    state.add_edge(cond, None, lib_node, "_cond", dace.Memlet.from_array("cond", sdfg.arrays["cond"]))
    state.add_edge(then, None, lib_node, "_t", dace.Memlet.from_array("then", sdfg.arrays["then"]))
    state.add_edge(els, None, lib_node, "_e", dace.Memlet.from_array("else", sdfg.arrays["else"]))
    state.add_edge(lib_node, "_o", out, None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    cls = lib_node.implementations["cutile"]
    tasklet = cls.expansion(lib_node, state, sdfg)
    return tasklet.code.as_string, tasklet.language


def _assert_parses_as_python(body: str) -> None:
    """Confirm ``body`` is a valid Python statement / expression."""
    ast.parse(body)


def test_tile_load_cutile_emits_block_id_and_ct_load_with_padding():
    """K=1 TileLoad cutile body: ``__pid0 = ct.bid(0)`` + ``ct.load(...)``."""
    body, lang = _expand_cutile(TileLoad(name="L", widths=(8, )))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.load(__src, index=(__pid0,)" in body
    assert "shape=(8,)" in body
    assert "padding_mode=ct.PaddingMode.ZERO" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_load_cutile_K2_emits_two_block_ids():
    """K=2 TileLoad cutile body has ``__pid0`` and ``__pid1``."""
    body, _ = _expand_cutile(TileLoad(name="L", widths=(4, 8)))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "__pid1 = ct.bid(1)" in body
    assert "index=(__pid0, __pid1)" in body
    assert "shape=(4, 8)" in body


def test_tile_load_cutile_pos_inf_pad_mode():
    """``pad_mode='POS_INF'`` selects ``ct.PaddingMode.POSITIVE_INFINITY``."""
    body, _ = _expand_cutile(TileLoad(name="L", widths=(8, ), pad_mode="POS_INF"))
    _assert_parses_as_python(body)
    assert "padding_mode=ct.PaddingMode.POSITIVE_INFINITY" in body
    assert "ct.PaddingMode.ZERO" not in body


def test_tile_load_cutile_masked_drops_dangling_mask_input():
    """``has_mask=True`` must NOT add a dead ``__mask`` input or reference it
    (L-load-nomask: ``ct.load`` has no mask; gating is deferred to the store)."""
    tasklet = _expand_cutile_tasklet(TileLoad(name="L", widths=(8, ), has_mask=True))
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "__mask" not in body
    assert "__mask" not in tasklet.in_connectors
    assert "padding_mode=ct.PaddingMode.ZERO" in body


def test_tile_store_cutile_unmasked_emits_ct_store():
    """Unmasked TileStore: ``ct.store(__output, index=(__pid0,), tile=__src)``."""
    body, _ = _expand_cutile(TileStore(name="S", widths=(8, )))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.store(__output, index=(__pid0,)" in body
    assert "tile=__src" in body
    assert "ct.scatter" not in body


def test_tile_store_cutile_masked_emits_ct_scatter_with_arange_indices():
    """Masked TileStore: ``ct.scatter(__output, (__idx0,), __src, mask=__mask)``
    with per-lane indices ``__idx_k = ct.arange(W_k) + __pid_k * W_k``."""
    body, _ = _expand_cutile(TileStore(name="S", widths=(8, ), has_mask=True))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 8" in body
    assert "ct.scatter(__output, (__idx0,), __src, mask=__mask)" in body


def test_tile_store_cutile_K2_masked_scatter_has_two_idx_tiles():
    """Masked K=2 store emits two per-lane index tiles + scatter."""
    body, _ = _expand_cutile(TileStore(name="S", widths=(4, 8), has_mask=True))
    _assert_parses_as_python(body)
    assert "ct.arange(4, dtype=ct.int32)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 4" in body
    assert "__pid1 * 8" in body
    assert "ct.scatter(__output, (__idx0, __idx1), __src, mask=__mask)" in body


def test_tile_binop_cutile_emits_bare_elementwise_op():
    """TileBinop cutile body emits the operator inline; no ``ct.where`` wrap
    (mask is applied at the scatter store, not at the binop)."""
    body, _ = _expand_cutile(TileBinop(name="B", widths=(8, ), op="+"))
    _assert_parses_as_python(body)
    assert "__rhs1 + __rhs2" in body
    assert "ct.where" not in body


def test_tile_binop_cutile_masked_still_bare_op():
    """Masked TileBinop does NOT wrap with ``ct.where`` — mask flows to the
    store. ``has_mask=True`` drops the ``__mask`` input on the cutile body
    because the binop never reads it."""
    body, _ = _expand_cutile(TileBinop(name="B", widths=(8, ), op="*", has_mask=True))
    _assert_parses_as_python(body)
    assert "__rhs1 * __rhs2" in body
    assert "ct.where" not in body


def test_tile_binop_cutile_symbol_operand_inlines_expr():
    """Symbol-kind RHS embeds the expression literally."""
    body, _ = _expand_cutile(TileBinop(name="B", widths=(8, ), op="+", kind_b="Symbol", expr_b="alpha"))
    _assert_parses_as_python(body)
    assert "alpha" in body


def test_tile_binop_cutile_uses_ct_minimum_for_min():
    """``min`` op routes to ``ct.minimum``."""
    body, _ = _expand_cutile(TileBinop(name="B", widths=(4, 8), op="min"))
    _assert_parses_as_python(body)
    assert "ct.minimum(__rhs1, __rhs2)" in body


def test_tile_mask_gen_cutile_1d_uses_arange_and_bid():
    """K=1 mask: ``__offsets0 = ct.arange(W)`` + ``__mask0 = ... < ub``;
    output is the single per-dim mask (no ``&`` combinator)."""
    body, _ = _expand_cutile(TileMaskGen(name="M", widths=(8, ), iter_vars=("i", ), global_ubs=("N_ub", )))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 8" in body
    assert "N_ub" in body
    assert "__output = __mask0" in body
    assert "&" not in body


def test_tile_mask_gen_cutile_K2_combines_per_dim_via_broadcast_and_amp():
    """K=2 mask combines two per-dim masks via ``broadcast_to`` and ``&``."""
    body, _ = _expand_cutile(TileMaskGen(name="M", widths=(4, 8), iter_vars=("i", "j"), global_ubs=("M_ub", "N_ub")))
    _assert_parses_as_python(body)
    assert "ct.arange(4" in body
    assert "ct.arange(8" in body
    assert "__pid0 * 4" in body
    assert "__pid1 * 8" in body
    assert "ct.broadcast_to(__mask0[:, None], (4, 8))" in body
    assert "ct.broadcast_to(__mask1[None, :], (4, 8))" in body
    assert " & " in body


def test_tile_gather_cutile_1d_unmasked_emits_padding_value():
    """1D unmasked gather: single index tile + ``padding_value=0`` (no mask)."""
    body, lang = _expand_cutile(TileGather(name="G", widths=(8, )))
    _assert_parses_as_python(body)
    assert "ct.gather(__src, __idx_0, padding_value=0)" in body
    assert "mask=" not in body
    assert lang == dace.dtypes.Language.Python


def test_tile_gather_cutile_1d_masked_emits_mask_and_padding_value():
    """1D masked gather: ``mask=__mask, padding_value=0``."""
    body, _ = _expand_cutile(TileGather(name="G", widths=(8, ), has_mask=True))
    _assert_parses_as_python(body)
    assert "ct.gather(__src, __idx_0, mask=__mask, padding_value=0)" in body


def test_tile_gather_cutile_2d_masked_uses_index_tuple():
    """2D-source masked gather uses the ``(__idx_0, __idx_1)`` tuple form."""
    body, _ = _expand_cutile(TileGather(name="G", widths=(8, ), source_ndim=2, has_mask=True))
    _assert_parses_as_python(body)
    assert "ct.gather(__src, (__idx_0, __idx_1), mask=__mask, padding_value=0)" in body


def test_tile_gather_cutile_pad_value_emitted():
    """A non-zero ``pad_value`` is emitted as ``padding_value=<v>`` (cuTile
    gather padding is an arbitrary scalar, so e.g. ``1`` for prod identity)."""
    body, _ = _expand_cutile(TileGather(name="G", widths=(8, ), pad_value=1))
    _assert_parses_as_python(body)
    assert "padding_value=1" in body


def test_tile_gather_cutile_nonunit_index_strides_raises():
    """Non-unit ``index_strides`` cannot be expressed by ``ct.gather`` (no
    per-lane stride) → ``NotImplementedError`` at expansion time."""
    with pytest.raises(NotImplementedError):
        _expand_cutile(TileGather(name="G", widths=(8, ), index_strides=(2, )))


def test_tile_reduce_cutile_unmasked_full_and_axis():
    """Unmasked reduction emits ``ct.sum(__src)`` / ``ct.sum(__src, axis=1)``."""
    full, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="+"))
    _assert_parses_as_python(full)
    assert "__output = ct.sum(__src)" in full

    axed, _ = _expand_cutile(TileReduce(name="R", widths=(4, 8), op="+", axis=1))
    _assert_parses_as_python(axed)
    assert "ct.sum(__src, axis=1)" in axed


def test_tile_reduce_cutile_unmasked_max_uses_ct_max():
    """``op='max'`` routes to ``ct.max``."""
    body, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="max"))
    _assert_parses_as_python(body)
    assert "__output = ct.max(__src)" in body


def test_tile_reduce_cutile_masked_sum_preselects_identity_via_where():
    """Masked ``+`` reduction (primary, ct.where assumed present on CI) must
    pre-select the identity ``0`` into masked lanes and consume ``__mask``."""
    tasklet = _expand_cutile_tasklet(TileReduce(name="R", widths=(4, 8), op="+", axis=1, has_mask=True))
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "ct.where(__mask, __src, 0)" in body
    assert "ct.sum(__masked_src, axis=1)" in body
    # The previously-dead __mask input is now genuinely consumed.
    assert "__mask" in tasklet.in_connectors


def test_tile_reduce_cutile_masked_min_preselects_pos_inf_via_where():
    """Masked ``min`` reduction pre-selects ``+inf`` into masked lanes."""
    body, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="min", has_mask=True))
    _assert_parses_as_python(body)
    assert "ct.where(__mask, __src, float('inf'))" in body
    assert "ct.min(__masked_src)" in body


def test_tile_reduce_cutile_masked_sum_fallback_blend(monkeypatch):
    """With ``ct.where`` known absent, masked ``+`` falls back to the
    arithmetic blend ``__m * __src`` reduced by ``ct.sum``."""
    monkeypatch.setattr(_tile_reduce_mod, "_CT_HAS_WHERE", False)
    body, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="+", has_mask=True))
    _assert_parses_as_python(body)
    assert "ct.where" not in body
    assert "__mask.astype(__src.dtype)" in body
    # DaCe's Python tasklet pipeline re-parens binop rhs on unparse.
    assert "ct.sum((__m * __src))" in body


def test_tile_reduce_cutile_masked_prod_fallback_blend(monkeypatch):
    """With ``ct.where`` absent, masked ``*`` blends ``1`` into inactive lanes."""
    monkeypatch.setattr(_tile_reduce_mod, "_CT_HAS_WHERE", False)
    body, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="*", has_mask=True))
    _assert_parses_as_python(body)
    # DaCe's Python tasklet pipeline re-parens binop subexpressions on unparse.
    assert "ct.prod(((__m * __src) + (1.0 - __m)))" in body


def test_tile_reduce_cutile_masked_min_without_where_raises(monkeypatch):
    """Masked ``min``/``max`` with ``ct.where`` absent raises (the ``inf*0``
    hazard makes the arithmetic blend unsafe — L-reduce-nomask)."""
    monkeypatch.setattr(_tile_reduce_mod, "_CT_HAS_WHERE", False)
    with pytest.raises(NotImplementedError):
        _expand_cutile(TileReduce(name="R", widths=(8, ), op="min", has_mask=True))
    with pytest.raises(NotImplementedError):
        _expand_cutile(TileReduce(name="R", widths=(8, ), op="max", has_mask=True))


def test_tile_merge_cutile_primary_emits_ct_where():
    """Primary (CI default) TileMerge body is ``ct.where(__cond, __then, __else)``."""
    body, lang = _expand_cutile(TileMerge(name="M", widths=(8, )))
    _assert_parses_as_python(body)
    assert body == "__output = ct.where(__cond, __then, __else)"
    assert lang == dace.dtypes.Language.Python


def test_tile_merge_cutile_fallback_arith_blend_for_int(monkeypatch):
    """With ``ct.where`` absent and an integer output dtype, TileMerge emits
    the arithmetic blend (no ``ct.where``)."""
    monkeypatch.setattr(_tile_merge_mod, "_CT_HAS_WHERE", False)
    body, _ = _expand_merge_cutile_with_dtype(TileMerge(name="M", widths=(8, )), dace.int32)
    _assert_parses_as_python(body)
    assert "ct.where" not in body
    assert "__cond.astype(__then.dtype)" in body
    assert "__m * __then" in body
    assert "* __else" in body


def test_tile_merge_cutile_fallback_float_raises(monkeypatch):
    """With ``ct.where`` absent and a float output dtype, TileMerge raises
    (``0.0 * inf = NaN`` would leak a non-finite unselected branch)."""
    monkeypatch.setattr(_tile_merge_mod, "_CT_HAS_WHERE", False)
    with pytest.raises(NotImplementedError):
        _expand_merge_cutile_with_dtype(TileMerge(name="M", widths=(8, )), dace.float64)
