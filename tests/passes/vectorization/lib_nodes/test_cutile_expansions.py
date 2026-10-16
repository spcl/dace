from dace.transformation.passes.vectorization.enums import ISA
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

The cutile expansion tasklets use the same connector names as the
library nodes themselves: ``_src``, ``_dst``, ``_a``, ``_b``, ``_c``,
``_cond``, ``_t``, ``_e``, ``_o``, ``_mask``, ``_idx_<k>`` (matching
the pure expansions).

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
from dace.libraries.tileops import (TileBinop, TileIota, TileITE, TileGather, TileMaskGen, TileMMA, TileReduce, TileScatter)


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


def _expand_cutile_with_edges(lib_node, in_arrays=None, out_arrays=None, transients=()):
    """Return ``(body, language)`` wiring actual edges + arrays.

    :param in_arrays: dict mapping input connector name to
        ``(array_name, shape, dtype)``. An edge is created from an
        :class:`AccessNode` of ``array_name`` to the lib node connector.
    :param out_arrays: dict mapping output connector name to
        ``(array_name, shape, dtype)``.
    :param transients: array names to declare as transients (e.g. a
        ``widths``-shaped tile-fill destination).
    """
    tasklet = _expand_cutile_tasklet_with_edges(lib_node, in_arrays, out_arrays, transients)
    return tasklet.code.as_string, tasklet.language


def _expand_cutile_tasklet_with_edges(lib_node, in_arrays=None, out_arrays=None, transients=()):
    """Like :func:`_expand_cutile_with_edges` but returns the raw tasklet."""
    sdfg = dace.SDFG(f"cutile_smoke_{lib_node.label}")
    state = sdfg.add_state("main")
    state.add_node(lib_node)

    for mapping, is_input in ((in_arrays or {}, True), (out_arrays or {}, False)):
        for conn, (arr_name, shape, dtype) in mapping.items():
            if arr_name not in sdfg.arrays:
                sdfg.add_array(arr_name, shape, dtype, transient=arr_name in transients)
            acc = state.add_access(arr_name)
            mem = dace.Memlet.from_array(arr_name, sdfg.arrays[arr_name])
            if is_input:
                state.add_edge(acc, None, lib_node, conn, mem)
            else:
                state.add_edge(lib_node, conn, acc, None, mem)

    cls = lib_node.implementations["cutile"]
    return cls.expansion(lib_node, state, sdfg)


def _assert_parses_as_python(body: str) -> None:
    """Confirm ``body`` is a valid Python statement / expression."""
    ast.parse(body)


def test_tile_load_cutile_emits_block_id_and_ct_load_with_padding():
    """K=1 TileGather cutile body: ``__pid0 = ct.bid(0)`` + ``ct.load(...)``."""
    body, lang = _expand_cutile_with_edges(
        TileGather(name="L", widths=(8, )),
        in_arrays={"_src": ("src", (100, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.load(_src, index=(__pid0,)" in body
    assert "shape=(8,)" in body
    assert "padding_mode=ct.PaddingMode.ZERO" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_load_cutile_K2_emits_two_block_ids():
    """K=2 TileGather cutile body has ``__pid0`` and ``__pid1``."""
    body, _ = _expand_cutile_with_edges(
        TileGather(name="L", widths=(4, 8)),
        in_arrays={"_src": ("src", (100, 200), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "__pid1 = ct.bid(1)" in body
    assert "index=(__pid0, __pid1)" in body
    assert "shape=(4, 8)" in body


def test_tile_load_cutile_pos_inf_pad_mode():
    """``pad_mode='POS_INF'`` selects ``ct.PaddingMode.POS_INF``."""
    body, _ = _expand_cutile_with_edges(
        TileGather(name="L", widths=(8, ), pad_mode="POS_INF"),
        in_arrays={"_src": ("src", (100, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "padding_mode=ct.PaddingMode.POS_INF" in body
    assert "ct.PaddingMode.ZERO" not in body


def test_tile_load_cutile_masked_uses_ct_where():
    """``has_mask=True`` wraps the loaded tile with
    ``ct.where(_mask, loaded, pad_value)`` to gate which lanes are valid.
    ``_mask`` IS declared as an input connector."""
    tasklet = _expand_cutile_tasklet_with_edges(
        TileGather(name="L", widths=(8, ), has_mask=True),
        in_arrays={"_src": ("src", (100, ), dace.float32)},
    )
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "ct.where(_mask" in body
    assert "_mask" in tasklet.in_connectors
    assert "ct.load(_src" in body
    assert "padding_mode=ct.PaddingMode.ZERO" in body


def test_tile_store_cutile_unmasked_emits_ct_store():
    """Unmasked TileScatter: ``ct.store(_dst, index=(__pid0,), tile=_src)``."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(8, )),
        in_arrays={"_src": ("src_tile", (8, ), dace.float32)},
        out_arrays={"_dst": ("dst", (100, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.store(_dst, index=(__pid0,)" in body
    assert "tile=_src" in body
    assert "ct.scatter" not in body


def test_tile_store_cutile_masked_emits_ct_scatter_with_arange_indices():
    """Masked TileScatter: ``ct.scatter(_dst, (__idx0,), _src, mask=_mask)``
    with per-lane indices ``__idx_k = ct.arange(W_k) + __pid_k * W_k``."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(8, ), has_mask=True),
        in_arrays={"_src": ("src_tile", (8, ), dace.float32)},
        out_arrays={"_dst": ("dst", (100, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 8" in body
    assert "ct.scatter(_dst, (__idx0,), _src, mask=_mask)" in body


def test_tile_store_cutile_K2_masked_scatter_has_two_idx_tiles():
    """Masked K=2 store emits two per-lane index tiles + scatter. The
    index tiles are broadcast to the full tile shape (cuTile scatter
    indices must broadcast against the scattered tile, matching the
    ``ct.gather`` index convention on the load side)."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(4, 8), has_mask=True),
        in_arrays={"_src": ("src_tile", (4, 8), dace.float32)},
        out_arrays={"_dst": ("dst", (100, 200), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "ct.arange(4, dtype=ct.int32)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "__pid0 * 4" in body
    assert "__pid1 * 8" in body
    assert "[:, None], (4, 8))" in body
    assert "[None, :], (4, 8))" in body
    assert "ct.scatter(_dst, (__idx0, __idx1), _src, mask=_mask)" in body


def test_tile_store_cutile_strided_emits_scatter_with_scaled_indices():
    """Non-unit ``dim_strides`` ⇒ no aligned block tile: the store lowers
    to ``ct.scatter`` with ``(ct.arange(W) + __pid0 * W) * coeff`` indices
    (mirror of the load's strided ``ct.gather`` path), even unmasked."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(8, ), dim_strides=(2, )),
        in_arrays={"_src": ("src_tile", (8, ), dace.float32)},
        out_arrays={"_dst": ("dst", (100, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "ct.store" not in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "* 2" in body
    assert "ct.scatter(_dst, (__idx0,), _src)" in body
    assert "mask=" not in body


def test_tile_store_cutile_strided_masked_scatter_combines_mask_and_stride():
    """Strided + masked store: one ``ct.scatter`` carries both the scaled
    per-lane indices and ``mask=_mask``."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(8, ), dim_strides=(2, ), has_mask=True),
        in_arrays={"_src": ("src_tile", (8, ), dace.float32)},
        out_arrays={"_dst": ("dst", (100, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "* 2" in body
    assert "ct.scatter(_dst, (__idx0,), _src, mask=_mask)" in body


def test_tile_store_cutile_transposed_dst_dims_permutes_tile():
    """Out-of-order ``dst_dims`` (transposed store) permutes the tile into
    array-dim order via ``ct.permute`` before the aligned ``ct.store``;
    the index tuple follows array-dim order (``__pid1`` first)."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(4, 8), dst_dims=(1, 0)),
        in_arrays={"_src": ("src_tile", (4, 8), dace.float32)},
        out_arrays={"_dst": ("dst", (100, 200), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "ct.permute(_src, axes=(1, 0))" in body
    assert "ct.store(_dst, index=(__pid1, __pid0), tile=__tile)" in body


def test_tile_store_cutile_unused_dst_dim_expands_rank_and_indexes_zero():
    """A destination with more dims than the tile (K=1 tile into a 2-D
    array, ``dst_dims=(1,)``) inserts a singleton axis (``_src[None, :]``)
    and pins the unused array dim's index to 0."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(8, ), dst_dims=(1, )),
        in_arrays={"_src": ("src_tile", (8, ), dace.float32)},
        out_arrays={"_dst": ("dst", (100, 200), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "_src[None, :]" in body
    assert "ct.store(_dst, index=(0, __pid0), tile=__tile)" in body


def test_tile_store_cutile_scalar_src_broadcasts_item():
    """``src_kind='Scalar'`` from a length-1 global array loads the value
    and broadcasts it to the tile shape before the store (mirror of the
    load's Scalar path)."""
    body, _ = _expand_cutile_with_edges(
        TileScatter(name="S", widths=(8, ), src_kind="Scalar"),
        in_arrays={"_src": ("sval", (1, ), dace.float32)},
        out_arrays={"_dst": ("dst", (100, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "ct.broadcast_to(ct.load(_src" in body
    assert ".item()" in body
    assert "tile=__tile" in body


def test_tile_store_cutile_symbol_fill_to_tile_transient_is_assignment():
    """``src_kind='Symbol'`` writing a ``widths``-shaped transient is the
    const-fill idiom: a plain dtype-typed ``ct.full`` assignment (tiles are
    SSA values in cuTile; a bare literal would materialize the wrong dtype),
    NOT a ``ct.store``. No ``_src`` input is declared."""
    tasklet = _expand_cutile_tasklet_with_edges(
        TileScatter(name="S", widths=(8, ), src_kind="Symbol", src_expr="alpha"),
        out_arrays={"_dst": ("tile_c", (8, ), dace.float32)},
        transients=("tile_c", ),
    )
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "_src" not in tasklet.in_connectors
    assert "ct.store" not in body
    assert "ct.scatter" not in body
    assert "ct.full((8,), alpha, ct.float32)" in body
    assert body.startswith("_dst = ")


def test_tile_store_cutile_symbol_fill_masked_blends_with_where():
    """A masked tile-transient fill blends inactive lanes to 0 via
    ``ct.where(_mask, <broadcast>, 0)`` (mirror of the load's mask blend)."""
    tasklet = _expand_cutile_tasklet_with_edges(
        TileScatter(name="S", widths=(8, ), src_kind="Symbol", src_expr="3.14", has_mask=True),
        out_arrays={"_dst": ("tile_c", (8, ), dace.float32)},
        transients=("tile_c", ),
    )
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "_mask" in tasklet.in_connectors
    assert "ct.where(_mask, ct.full((8,), 3.14, ct.float32), 0)" in body


def test_tile_store_cutile_symbol_to_global_broadcasts_then_stores():
    """``src_kind='Symbol'`` writing a non-transient global array emits a
    broadcast tile + aligned ``ct.store`` (no ``_src`` input)."""
    tasklet = _expand_cutile_tasklet_with_edges(
        TileScatter(name="S", widths=(8, ), src_kind="Symbol", src_expr="alpha"),
        out_arrays={"_dst": ("dst", (100, ), dace.float32)},
    )
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "_src" not in tasklet.in_connectors
    assert "__tile = ct.full((8,), alpha, ct.float32)" in body
    assert "ct.store(_dst, index=(__pid0,), tile=__tile)" in body


def test_tile_binop_cutile_emits_bare_elementwise_op():
    """TileBinop cutile body emits the operator inline; no ``ct.where`` wrap
    (mask is applied at the scatter store, not at the binop)."""
    body, _ = _expand_cutile(TileBinop(name="B", widths=(8, ), op="+"))
    _assert_parses_as_python(body)
    assert "_a + _b" in body
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
    assert "ct.minimum(_a, _b)" in body


def test_tile_mask_gen_cutile_1d_uses_arange_and_bid():
    """K=1 mask: ``__offsets0 = ct.arange(W)`` + ``__mask0 = offsets + iter_var < ub``;
    output is the single per-dim mask (no ``&`` combinator). The mask base is the
    iter var itself (bound to ``start + __pid*step`` by the cuTile codegen), NOT a
    reconstructed ``__pid0 * W`` — the latter miscomputes for nonzero-begin ranges."""
    body, _ = _expand_cutile(TileMaskGen(name="M", widths=(8, ), iter_vars=("i", ), global_ubs=("N_ub", )))
    _assert_parses_as_python(body)
    assert "__pid0 = ct.bid(0)" in body
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert "(__offsets0 + i) < N_ub" in body
    assert "__pid0 *" not in body
    assert "_o = __mask0" in body
    assert "&" not in body


def test_tile_mask_gen_cutile_K2_combines_per_dim_via_broadcast_and_amp():
    """K=2 mask combines two per-dim masks via ``broadcast_to`` and ``&``;
    each per-dim mask is based on its iter var (see the K=1 test)."""
    body, _ = _expand_cutile(TileMaskGen(name="M", widths=(4, 8), iter_vars=("i", "j"), global_ubs=("M_ub", "N_ub")))
    _assert_parses_as_python(body)
    assert "ct.arange(4" in body
    assert "ct.arange(8" in body
    assert "(__offsets0 + i) < M_ub" in body
    assert "(__offsets1 + j) < N_ub" in body
    assert "ct.broadcast_to(__mask0[:, None], (4, 8))" in body
    assert "ct.broadcast_to(__mask1[None, :], (4, 8))" in body
    assert " & " in body


# ============================================================
# TileIota cutile expansion
# ============================================================


def test_tile_iota_cutile_k1_affine_emits_arange_and_expr():
    """K=1 affine iota ``i + __l0``: body defines ``__l0 = ct.arange(8, ...)``
    and evaluates the expr in-line. DaCe's Python tasklet round-trip may
    re-parenthesise binary ops (``a + b`` -> ``(a + b)``)."""
    body, lang = _expand_cutile(TileIota(name="I", widths=(8, ), expr="i + __l0"))
    _assert_parses_as_python(body)
    assert "ct.bid" not in body  # TileIota has no block-ID preamble
    assert "__l0 = ct.arange(8, dtype=ct.int32)" in body
    # DaCe may re-paren: ``_dst = (i + __l0)``
    assert "i + __l0" in body
    assert "_dst" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_iota_cutile_k1_strided_expr():
    """K=1 strided iota ``i + 2 * __l0``: the stride factor is embedded
    in the expression and broadcasts element-wise over the arange."""
    body, lang = _expand_cutile(TileIota(name="I", widths=(4, ), expr="i + 2 * __l0"))
    _assert_parses_as_python(body)
    assert "__l0 = ct.arange(4, dtype=ct.int32)" in body
    assert "i + 2 * __l0" in body or "i + (2 * __l0)" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_iota_cutile_k2_broadcasts_lane_arrays():
    """K=2 iota ``i + __l0 * 4 + __l1``: each lane array is broadcast to
    the full ``(2, 4)`` tile shape before the expression."""
    body, lang = _expand_cutile(TileIota(name="I", widths=(2, 4), expr="i + __l0 * 4 + __l1"))
    _assert_parses_as_python(body)
    assert "ct.bid" not in body  # TileIota has no block-ID preamble
    assert "ct.broadcast_to(ct.arange(2, dtype=ct.int32)[:, None], (2, 4))" in body
    assert "ct.broadcast_to(ct.arange(4, dtype=ct.int32)[None, :], (2, 4))" in body
    # Expression may be re-parenthesised by DaCe's Python tasklet pipeline.
    assert "__l0 * 4" in body
    assert "__l1" in body
    assert "_dst" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_iota_cutile_k3_broadcasts_three_dims():
    """K=3 iota: three lane arrays are broadcast to (2, 4, 8)."""
    body, lang = _expand_cutile(TileIota(name="I", widths=(2, 4, 8), expr="__l0 * 32 + __l1 * 8 + __l2"))
    _assert_parses_as_python(body)
    assert "ct.bid" not in body  # TileIota has no block-ID preamble
    assert "ct.broadcast_to(ct.arange(2, dtype=ct.int32)[:, None, None], (2, 4, 8))" in body
    assert "ct.broadcast_to(ct.arange(4, dtype=ct.int32)[None, :, None], (2, 4, 8))" in body
    assert "ct.broadcast_to(ct.arange(8, dtype=ct.int32)[None, None, :], (2, 4, 8))" in body
    assert "__l0 * 32" in body
    assert "__l1 * 8" in body
    assert "__l2" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_iota_cutile_dtype_derived_from_dst_descriptor():
    """The arange dtype follows the ``_dst`` descriptor (int64 transient ->
    ``ct.int64``), so descriptor and runtime tile dtype agree."""
    body, lang = _expand_cutile_with_edges(
        TileIota(name="I64", widths=(8, ), expr="i + __l0"),
        out_arrays={"_dst": ("dst64", (8, ), dace.int64)},
        transients=("dst64", ),
    )
    _assert_parses_as_python(body)
    assert "__l0 = ct.arange(8, dtype=ct.int64)" in body
    assert "ct.int32" not in body
    assert lang == dace.dtypes.Language.Python


def test_tile_iota_cutile_dtype_derived_k2_int64():
    """K=2: both broadcast lane arrays carry the ``_dst`` dtype."""
    body, _ = _expand_cutile_with_edges(
        TileIota(name="I64_k2", widths=(2, 4), expr="__l0 * 4 + __l1"),
        out_arrays={"_dst": ("dst64_k2", (2, 4), dace.int64)},
        transients=("dst64_k2", ),
    )
    _assert_parses_as_python(body)
    assert "ct.broadcast_to(ct.arange(2, dtype=ct.int64)[:, None], (2, 4))" in body
    assert "ct.broadcast_to(ct.arange(4, dtype=ct.int64)[None, :], (2, 4))" in body


def test_tile_iota_cutile_dtype_falls_back_to_int32_without_descriptor():
    """Without a wired ``_dst`` descriptor the historical int32 default holds."""
    body, _ = _expand_cutile(TileIota(name="I_fallback", widths=(8, ), expr="__l0"))
    assert "ct.arange(8, dtype=ct.int32)" in body


@pytest.mark.gpu
def test_tile_iota_int64_lane_tile_gpu_numeric():
    """E2E: ``C[i] = A[i] * i`` materializes an int64 lane-id tile via
    TileIota; the generated arange must be int64 and match NumPy."""
    import numpy as np

    from dace.transformation.passes.vectorization import VectorizeCuTile

    @dace.program
    def tile_iota_int64_scale(A: dace.float64[100], C: dace.float64[100]):
        for i in dace.map[0:100]:
            C[i] = A[i] * i

    sdfg = tile_iota_int64_scale.to_sdfg()
    VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
    code = next(co for co in sdfg.generate_code() if co.name == sdfg.name).code
    assert "ct.arange(8, dtype=ct.int64)" in code  # the lane-id tile
    rng = np.random.default_rng(0)
    A = rng.random(100)
    C = np.zeros(100)
    sdfg(A=A, C=C)
    np.testing.assert_allclose(C, A * np.arange(100))


def test_tile_iota_cutile_degenerate_single_lane_substitutes_zeros():
    """All-ones widths (degenerate): lane vars replaced with 0, no arange."""
    body, lang = _expand_cutile(TileIota(name="I", widths=(1, ), expr="i + __l0"))
    _assert_parses_as_python(body)
    # DaCe may re-paren: ``_dst = (i + 0)``
    assert "i + 0" in body
    assert "_dst" in body
    assert "ct.arange" not in body
    assert "ct.bid" not in body
    assert lang == dace.dtypes.Language.Python


def test_tile_iota_cutile_degenerate_k2_single_lane():
    """K=2 with widths (1, 1): both lane vars replaced with 0."""
    body, lang = _expand_cutile(TileIota(name="I", widths=(1, 1), expr="i + __l0 + __l1"))
    _assert_parses_as_python(body)
    # DaCe may re-paren: ``_dst = ((i + 0) + 0)``
    assert "i + 0" in body
    assert "_dst" in body
    assert "ct.arange" not in body
    assert lang == dace.dtypes.Language.Python


def test_tile_iota_cutile_with_extra_input_idx():
    """Extra input ``_idx`` in the expression: the connector is declared and
    the expression uses array indexing over the arange tile."""
    node = TileIota(name="I", widths=(8, ), expr="_idx[__l0]", extra_inputs=("_idx", ))
    tasklet = _expand_cutile_tasklet(node)
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "_idx" in tasklet.in_connectors
    assert "__l0 = ct.arange(8, dtype=ct.int32)" in body
    assert "_idx[__l0]" in body


def test_tile_iota_cutile_with_extra_input_src():
    """Extra input ``_src`` used in a flat-offset expression: the connector
    is declared and the indexing is element-wise cuTile gather."""
    node = TileIota(name="I", widths=(4, ), expr="_src[__l0 * 2]", extra_inputs=("_src", ))
    tasklet = _expand_cutile_tasklet(node)
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "_src" in tasklet.in_connectors
    assert "_src[__l0 * 2]" in body or "_src[((__l0 * 2))]" in body or "_src[(__l0 * 2)]" in body


def test_tile_iota_cutile_degenerate_single_lane_with_idx_rewrites_indexing():
    """Degenerate single-lane with ``_idx[__l0]``: after substitution,
    ``_idx[0]`` becomes the bare connector name ``_idx``."""
    node = TileIota(name="I", widths=(1, ), expr="_idx[__l0]", extra_inputs=("_idx", ))
    tasklet = _expand_cutile_tasklet(node)
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "_dst = _idx" in body
    assert "_idx[0]" not in body
    assert "_idx" in tasklet.in_connectors


def test_tile_iota_cutile_multiple_extra_inputs():
    """Multiple extra inputs: all are declared as connectors."""
    node = TileIota(name="I", widths=(8, ), expr="_a + _b[__l0]", extra_inputs=("_a", "_b"))
    tasklet = _expand_cutile_tasklet(node)
    body = tasklet.code.as_string
    _assert_parses_as_python(body)
    assert "_a" in tasklet.in_connectors
    assert "_b" in tasklet.in_connectors
    # DaCe may re-paren: ``_dst = (_a + _b[__l0])``
    assert "_a" in body
    assert "_b[__l0]" in body


def test_tile_iota_cutile_implementations_dict_has_cutile():
    """``TileIota.implementations`` exposes ``'cutile'``."""
    assert "cutile" in TileIota.implementations
    assert "pure" in TileIota.implementations


def test_tile_iota_cutile_target_isa_property_default():
    """``TileIota.target_isa`` defaults to ``'SCALAR'`` and is stampable."""
    node = TileIota(name="I", widths=(8, ), expr="__l0")
    assert node.target_isa == ISA.SCALAR
    node.target_isa = ISA.CUTILE
    assert node.target_isa == ISA.CUTILE


# ============================================================
# TileReduce cuTile expansion (bug fix verification)
# ============================================================


def test_tile_reduce_cutile_sum_unmasked():
    """K=1, op='+', unmasked: emits ct.sum(_src, axis=None)."""
    body, lang = _expand_cutile(TileReduce(name="R", widths=(8, ), op="+"))
    _assert_parses_as_python(body)
    assert "ct.sum(_src" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_reduce_cutile_sum_masked():
    """K=1, op='+', masked: applies ct.where before reduction."""
    body, _ = _expand_cutile_with_edges(
        TileReduce(name="R", widths=(8, ), op="+", has_mask=True),
        in_arrays={
            "_src": ("src", (8, ), dace.float32),
            "_mask": ("mask", (8, ), dace.bool_),
        },
        out_arrays={"_dst": ("dst", (1, ), dace.float32)},
    )
    _assert_parses_as_python(body)
    assert "ct.where(_mask" in body
    assert "ct.sum" in body


def test_tile_reduce_cutile_prod():
    """op='*' emits ct.prod."""
    body, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="*"))
    _assert_parses_as_python(body)
    assert "ct.prod(" in body


def test_tile_reduce_cutile_min():
    """op='min' emits ct.min."""
    body, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="min"))
    _assert_parses_as_python(body)
    assert "ct.min(" in body


def test_tile_reduce_cutile_max():
    """op='max' emits ct.max."""
    body, _ = _expand_cutile(TileReduce(name="R", widths=(8, ), op="max"))
    _assert_parses_as_python(body)
    assert "ct.max(" in body


def test_tile_reduce_cutile_k2_axis0():
    """K=2, axis=0: verify axis appears in emitted reduction."""
    body, _ = _expand_cutile(TileReduce(name="R", widths=(4, 8), op="+", axis=0))
    _assert_parses_as_python(body)
    assert "axis=0" in body


def test_tile_reduce_cutile_k2_axis1():
    """K=2, axis=1: verify axis=1."""
    body, _ = _expand_cutile(TileReduce(name="R", widths=(4, 8), op="+", axis=1))
    _assert_parses_as_python(body)
    assert "axis=1" in body


def test_tile_reduce_cutile_inputs_include_mask_when_masked():
    """Verify _mask is in tasklet inputs when has_mask=True."""
    tasklet = _expand_cutile_tasklet_with_edges(
        TileReduce(name="R", widths=(8, ), op="+", has_mask=True),
        in_arrays={
            "_src": ("src", (8, ), dace.float32),
            "_mask": ("mask", (8, ), dace.bool_),
        },
        out_arrays={"_dst": ("dst", (1, ), dace.float32)},
    )
    assert "_mask" in tasklet.in_connectors
    assert "_src" in tasklet.in_connectors


# ============================================================
# TileBinop ** operator
# ============================================================


def test_tile_binop_cutile_power_operator():
    """op='**' emits Python power expression."""
    body, lang = _expand_cutile(TileBinop(name="P", widths=(8, ), op="**"))
    _assert_parses_as_python(body)
    assert "**" in body
    assert lang == dace.dtypes.Language.Python


# ============================================================
# TileITE cuTile expansion
# ============================================================


def _ite_arrays(widths, kind_mask="Tile", kind_t="Tile", kind_e="Tile"):
    """Build ``(in_arrays, out_arrays)`` dicts for a TileITE expansion test."""
    shape = tuple(widths)
    ins = {}
    if kind_mask in ("Tile", "Scalar"):
        ins["_mask"] = ("mask", shape, dace.bool_)
    if kind_t in ("Tile", "Scalar"):
        ins["_t"] = ("t", shape, dace.float32)
    if kind_e in ("Tile", "Scalar"):
        ins["_e"] = ("e", shape, dace.float32)
    outs = {"_o": ("o", shape, dace.float32)}
    return ins, outs


def test_tile_ite_cutile_all_tile():
    """All-Tile operands: basic ct.where(_mask, _t, _e)."""
    node = TileITE(name="ITE", widths=(8, ))
    ins, outs = _ite_arrays((8, ))
    body, lang = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.where(_mask, _t, _e)" in body
    assert lang == dace.dtypes.Language.Python


def test_tile_ite_cutile_symbol_then():
    """Symbol then-arm: inline expression in ct.where."""
    node = TileITE(name="ITE", widths=(8, ), kind_t="Symbol", expr_t="0.0")
    ins, outs = _ite_arrays((8, ), kind_t="Symbol")
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.where(_mask, 0.0, _e)" in body


def test_tile_ite_cutile_symbol_else():
    """Symbol else-arm: inline expression."""
    node = TileITE(name="ITE", widths=(8, ), kind_e="Symbol", expr_e="1.0")
    ins, outs = _ite_arrays((8, ), kind_e="Symbol")
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.where(_mask, _t, 1.0)" in body


def test_tile_ite_cutile_all_symbol():
    """All-Symbol operands: all three inlined, no connectors."""
    node = TileITE(name="ITE",
                   widths=(8, ),
                   kind_mask="Symbol",
                   expr_mask="True",
                   kind_t="Symbol",
                   expr_t="1",
                   kind_e="Symbol",
                   expr_e="0")
    ins, outs = _ite_arrays((8, ), kind_mask="Symbol", kind_t="Symbol", kind_e="Symbol")
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.where(True, 1, 0)" in body


def test_tile_ite_cutile_mixed_kinds():
    """Mixed: mask=Tile, then=Symbol, else=Tile -- verify connectors."""
    node = TileITE(name="ITE", widths=(8, ), kind_t="Symbol", expr_t="42")
    ins, outs = _ite_arrays((8, ), kind_t="Symbol")
    tasklet = _expand_cutile_tasklet_with_edges(node, in_arrays=ins, out_arrays=outs)
    assert "_mask" in tasklet.in_connectors  # Tile mask
    assert "_t" not in tasklet.in_connectors  # Symbol -- no connector
    assert "_e" in tasklet.in_connectors  # Tile else


def test_tile_ite_cutile_scalar_mask():
    """Scalar mask: connector present but value is passed through."""
    node = TileITE(name="ITE", widths=(8, ), kind_mask="Scalar")
    ins, outs = _ite_arrays((8, ), kind_mask="Scalar")
    tasklet = _expand_cutile_tasklet_with_edges(node, in_arrays=ins, out_arrays=outs)
    assert "_mask" in tasklet.in_connectors


def test_tile_ite_cutile_k2():
    """K=2 tile: body is valid Python."""
    node = TileITE(name="ITE", widths=(4, 8))
    ins, outs = _ite_arrays((4, 8))
    body, lang = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.where" in body
    assert lang == dace.dtypes.Language.Python


# ============================================================
# TileMMA cuTile expansion
# ============================================================


def _mma_arrays(M, K_inner, N, beta):
    """Build in/out array dicts for TileMMA expansion."""
    ins = {
        "_a": ("a", (M, K_inner), dace.float32),
        "_b": ("b", (K_inner, N), dace.float32),
    }
    if beta != 0:
        ins["_cin"] = ("cin", (M, N), dace.float32)
    outs = {"_c": ("c", (M, N), dace.float32)}
    return ins, outs


def test_tile_mma_cutile_alpha1_beta0():
    """alpha=1, beta=0: overwrite, no _cin."""
    node = TileMMA(name="MMA", widths=(16, 8, 16), alpha=1, beta=0)
    ins, outs = _mma_arrays(16, 8, 16, 0)
    body, lang = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "_c = ct.mma(_a, _b)" in body
    assert "_cin" not in body
    assert lang == dace.dtypes.Language.Python


def test_tile_mma_cutile_alpha1_beta1():
    """alpha=1, beta=1: accumulate via ct.mma's third arg."""
    node = TileMMA(name="MMA", widths=(16, 8, 16), alpha=1, beta=1)
    ins, outs = _mma_arrays(16, 8, 16, 1)
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "_c = ct.mma(_a, _b, _cin)" in body


def test_tile_mma_cutile_alpha2_beta0():
    """alpha=2, beta=0: scaled overwrite."""
    node = TileMMA(name="MMA", widths=(16, 8, 16), alpha=2, beta=0)
    ins, outs = _mma_arrays(16, 8, 16, 0)
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "2 * ct.mma(_a, _b)" in body


def test_tile_mma_cutile_alpha1_beta2():
    """alpha=1, beta=2: scaled accumulate."""
    node = TileMMA(name="MMA", widths=(16, 8, 16), alpha=1, beta=2)
    ins, outs = _mma_arrays(16, 8, 16, 2)
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.mma(_a, _b)" in body
    assert "2 * _cin" in body


def test_tile_mma_cutile_general():
    """General: alpha=3, beta=2."""
    node = TileMMA(name="MMA", widths=(16, 8, 16), alpha=3, beta=2)
    ins, outs = _mma_arrays(16, 8, 16, 2)
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "3 * ct.mma(_a, _b)" in body
    assert "2 * _cin" in body


def test_tile_mma_cutile_connectors_beta0():
    """beta=0: _cin NOT in inputs."""
    node = TileMMA(name="MMA", widths=(16, 8, 16), alpha=1, beta=0)
    ins, outs = _mma_arrays(16, 8, 16, 0)
    tasklet = _expand_cutile_tasklet_with_edges(node, in_arrays=ins, out_arrays=outs)
    assert "_cin" not in tasklet.in_connectors


def test_tile_mma_cutile_connectors_beta1():
    """beta=1: _cin IS in inputs."""
    node = TileMMA(name="MMA", widths=(16, 8, 16), alpha=1, beta=1)
    ins, outs = _mma_arrays(16, 8, 16, 1)
    tasklet = _expand_cutile_tasklet_with_edges(node, in_arrays=ins, out_arrays=outs)
    assert "_cin" in tasklet.in_connectors


# ============================================================
# TileGather gather_dims cuTile expansion
# ============================================================


def test_tile_load_cutile_gather_1d():
    """1-D source, gather_dims=(0,): emit ct.gather with _idx_0."""
    node = TileGather(name="L", widths=(8, ), gather_dims=(0, ))
    ins = {
        "_src": ("src", (64, ), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    body, lang = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    _assert_parses_as_python(body)
    assert "ct.gather" in body
    assert "_idx_0" in body
    assert "ct.load" not in body
    assert lang == dace.dtypes.Language.Python


def test_tile_load_cutile_gather_2d():
    """2-D source, gather_dims=(0,1): both dims gathered."""
    node = TileGather(name="L", widths=(8, ), gather_dims=(0, 1))
    ins = {
        "_src": ("src", (64, 64), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
        "_idx_1": ("idx1", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    _assert_parses_as_python(body)
    assert "ct.gather" in body
    assert "_idx_0" in body
    assert "_idx_1" in body


def test_tile_load_cutile_gather_partial():
    """2-D source, gather_dims=(0,): dim 0 gathered, dim 1 structured."""
    node = TileGather(name="L", widths=(8, ), gather_dims=(0, ), src_dims=(1, ))
    ins = {
        "_src": ("src", (64, 64), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    _assert_parses_as_python(body)
    assert "ct.gather" in body
    assert "_idx_0" in body
    assert "ct.arange" in body  # structured contribution for dim 1


def test_tile_load_cutile_gather_masked():
    """gather + has_mask: mask passed to ct.gather via mask= kwarg."""
    node = TileGather(name="L", widths=(8, ), gather_dims=(0, ), has_mask=True)
    ins = {
        "_src": ("src", (64, ), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
        "_mask": ("mask", (8, ), dace.bool_),
    }
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    _assert_parses_as_python(body)
    assert "mask=_mask" in body
    # On the gather_dims path, mask is passed to ct.gather directly, NOT ct.where
    assert "ct.where" not in body


def test_tile_load_cutile_gather_idx_connectors():
    """_idx_{d} are in tasklet input connectors."""
    node = TileGather(name="L", widths=(8, ), gather_dims=(0, 1))
    ins = {
        "_src": ("src", (64, 64), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
        "_idx_1": ("idx1", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    tasklet = _expand_cutile_tasklet_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    assert "_idx_0" in tasklet.in_connectors
    assert "_idx_1" in tasklet.in_connectors


# ============================================================
# TileGather replicate cuTile expansion
# ============================================================


def test_tile_load_cutile_replicate_factor_2():
    """replicate_factor=(2,): forces ct.gather with // 2 in index."""
    node = TileGather(name="L", widths=(8, ), replicate_factor_per_dim=(2, ))
    ins = {"_src": ("src", (64, ), dace.float32)}
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    _assert_parses_as_python(body)
    assert "ct.gather" in body  # NOT ct.load -- replicate forces gather
    assert "// 2" in body
    assert "ct.load" not in body


def test_tile_load_cutile_replicate_no_replicate_uses_load():
    """No replicate (all 1s): should use ct.load, not ct.gather."""
    node = TileGather(name="L", widths=(8, ), replicate_factor_per_dim=(1, ))
    ins = {"_src": ("src", (64, ), dace.float32)}
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    _assert_parses_as_python(body)
    assert "ct.load" in body  # Should use the aligned load path


# ============================================================
# TileGather NEG_INF padding
# ============================================================


def test_tile_load_cutile_neg_inf_pad_mode():
    """NEG_INF padding mode: emits ct.PaddingMode.NEG_INF."""
    node = TileGather(name="L", widths=(8, ), pad_mode="NEG_INF")
    ins = {"_src": ("src", (64, ), dace.float32)}
    outs = {"_dst": ("dst", (8, ), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs, transients=("dst", ))
    _assert_parses_as_python(body)
    assert "NEG_INF" in body


# ============================================================
# TileScatter gather_dims cuTile expansion
# ============================================================


def test_tile_store_cutile_gather_1d():
    """1-D dest, gather_dims=(0,): emit ct.scatter with _idx_0."""
    node = TileScatter(name="S", widths=(8, ), gather_dims=(0, ))
    ins = {
        "_src": ("src", (8, ), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (64, ), dace.float32)}
    body, lang = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.scatter" in body
    assert "_idx_0" in body
    assert "ct.store" not in body
    assert lang == dace.dtypes.Language.Python


def test_tile_store_cutile_gather_2d():
    """2-D dest, gather_dims=(0,1): both dims scattered."""
    node = TileScatter(name="S", widths=(8, ), gather_dims=(0, 1))
    ins = {
        "_src": ("src", (8, ), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
        "_idx_1": ("idx1", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (64, 64), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "ct.scatter" in body
    assert "_idx_0" in body
    assert "_idx_1" in body


def test_tile_store_cutile_gather_masked():
    """gather + has_mask: verify mask=_mask on ct.scatter."""
    node = TileScatter(name="S", widths=(8, ), gather_dims=(0, ), has_mask=True)
    ins = {
        "_src": ("src", (8, ), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
        "_mask": ("mask", (8, ), dace.bool_),
    }
    outs = {"_dst": ("dst", (64, ), dace.float32)}
    body, _ = _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)
    _assert_parses_as_python(body)
    assert "mask=_mask" in body


def test_tile_store_cutile_gather_idx_connectors():
    """_idx_{d} in tasklet input connectors."""
    node = TileScatter(name="S", widths=(8, ), gather_dims=(0, ))
    ins = {
        "_src": ("src", (8, ), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (64, ), dace.float32)}
    tasklet = _expand_cutile_tasklet_with_edges(node, in_arrays=ins, out_arrays=outs)
    assert "_idx_0" in tasklet.in_connectors


# ============================================================
# TileScatter WCR guard
# ============================================================


def test_tile_store_cutile_wcr_raises():
    """WCR set on TileScatter: cuTile expansion raises NotImplementedError."""
    node = TileScatter(name="S", widths=(8, ), gather_dims=(0, ), dim_strides=(0, ), wcr="lambda a, b: a + b")
    ins = {
        "_src": ("src", (8, ), dace.float32),
        "_idx_0": ("idx0", (8, ), dace.int32),
    }
    outs = {"_dst": ("dst", (64, ), dace.float32)}
    with pytest.raises(NotImplementedError, match="WCR"):
        _expand_cutile_with_edges(node, in_arrays=ins, out_arrays=outs)


# ============================================================
# Pipeline registration
# ============================================================


def test_cutile_lowering_tile_node_types_includes_ite_and_mma():
    """_tile_node_types() includes TileITE and TileMMA."""
    from dace.transformation.passes.vectorization.cutile_lowering import _tile_node_types
    types = _tile_node_types()
    type_names = {t.__name__ for t in types}
    assert "TileITE" in type_names
    assert "TileMMA" in type_names
