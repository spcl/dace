# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`materialise_per_lane_index_tile` (design section 3.8).

Covers the single-dim dependency case end-to-end: hand-built SDFG, the
helper mints the transient + emits the materialisation tasklet, the SDFG
compiles, runs, and produces the expected per-lane indices.
"""
import numpy as np

import dace
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.widen_accesses import (
    materialise_per_lane_index_tile, )


def test_helper_mints_transient_and_tasklet():
    """The helper adds the transient + tasklet + edge to the state."""
    sdfg = dace.SDFG("mint_fixture")
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state, name_hint="idx_t", gather_expr="i", tile_iter_vars="i", tile_widths=8)
    # Transient is in arrays + has the expected shape / dtype.
    desc = sdfg.arrays[name]
    assert tuple(desc.shape) == (8, )
    assert desc.dtype == dace.int64
    assert desc.transient
    # The state has one tasklet + one AccessNode.
    tasklets = [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]
    ans = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)]
    assert len(tasklets) == 1
    assert len(ans) == 1


def test_e2e_identity_lane_index():
    """``gather_expr="i"`` with ``tile_iter_vars="i"`` materialises ``[0, 1, ..., W-1]``."""
    sdfg = dace.SDFG("identity_lane_index")
    sdfg.add_array("Out", (8, ), dace.int64, transient=False)
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state, name_hint="idx_t", gather_expr="i", tile_iter_vars="i", tile_widths=8)
    inner_an = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode))
    out_an = state.add_access("Out")
    state.add_edge(inner_an, None, out_an, None, Memlet(f"{name}[0:8]"))
    sdfg.validate()
    out = np.zeros(8, dtype=np.int64)
    sdfg(Out=out)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.int64))


def test_e2e_affine_lane_index():
    """``gather_expr="2*i + 1"`` -- per-lane evaluation of affine in the iter-var."""
    sdfg = dace.SDFG("affine_lane_index")
    sdfg.add_array("Out", (8, ), dace.int64, transient=False)
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state,
                                           name_hint="idx_t",
                                           gather_expr="2*i + 1",
                                           tile_iter_vars="i",
                                           tile_widths=8)
    inner_an = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode))
    out_an = state.add_access("Out")
    state.add_edge(inner_an, None, out_an, None, Memlet(f"{name}[0:8]"))
    sdfg.validate()
    out = np.zeros(8, dtype=np.int64)
    sdfg(Out=out)
    expected = np.array([2 * i + 1 for i in range(8)], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_helper_supports_int32_dtype():
    """``idx_dtype=dace.int32`` produces an int32-typed transient."""
    sdfg = dace.SDFG("int32_index")
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state,
                                           name_hint="idx_t",
                                           gather_expr="i",
                                           tile_iter_vars="i",
                                           tile_widths=4,
                                           idx_dtype=dace.int32)
    assert sdfg.arrays[name].dtype == dace.int32


def test_widen_accesses_returns_none_on_empty_sdfg():
    """No tile-tagged maps -> no widening / gather materialisation -> ``None``.

    Folds the prior ``PreparePerLaneIndices`` Pass-level test into the
    unified ``WidenAccesses`` per user direction 2026-06-11.
    """
    from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    assert WidenAccesses(widths=(8, )).apply_pass(sdfg, {}) is None


def test_widen_accesses_materialises_index_tile_for_gather_access():
    """``a[idx[ii]]`` (1D gather) -> WidenAccesses mints one per-lane index tile.

    Folds the prior ``PreparePerLaneIndices`` walker test into the unified
    ``WidenAccesses`` (per user direction 2026-06-11).
    """
    from dace.memlet import Memlet as _Memlet
    from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses

    sdfg = dace.SDFG("walker_gather_fixture")
    sdfg.add_array("A", (32, ), dace.float64, transient=False)
    sdfg.add_array("idx", (32, ), dace.int64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body_nsdfg")
    inner.add_array("A", (32, ), dace.float64, transient=False)
    inner.add_array("idx", (32, ), dace.int64, transient=False)
    inner.add_array("out_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    t_inner = instate.add_access("out_t")
    tasklet = instate.add_tasklet("ld", {"_a"}, {"_o"}, "_o = _a")
    # Gather: A[idx[ii]]. Per-tile subset on A is [idx[ii]:idx[ii]+1] -- GATHER on dim 0.
    from dace.subsets import Range as _Range
    from dace.symbolic import pystr_to_symbolic as _to_sym
    instate.add_edge(a_inner, None, tasklet, "_a",
                     _Memlet(data="A", subset=_Range([(_to_sym("idx[ii]"), _to_sym("idx[ii]"), 1)])))
    instate.add_edge(tasklet, "_o", t_inner, None, _Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "idx"}, set(), symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    idx_outer = state.add_access("idx")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=_Memlet("A[0:32]"))
    state.add_memlet_path(idx_outer, me, nsdfg, dst_conn="idx", memlet=_Memlet("idx[0:32]"))
    state.add_nedge(nsdfg, mx, _Memlet())

    before_int_arrays = sum(1 for d in inner.arrays.values()
                            if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64)
    WidenAccesses(widths=(8, )).apply_pass(sdfg, {})
    after_int_arrays = sum(1 for d in inner.arrays.values()
                           if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64)
    assert after_int_arrays == before_int_arrays + 1, "expected one new int64 index transient"


def test_e2e_2d_lane_index():
    """Full K=2 dep: ``2*i + j`` materialises ``(W_i, W_j)`` -- no trailing ONE.

    Per user direction 2026-06-10 (cuTile contract): the idx tile shape matches
    the output tile rank. When both iter-vars appear in the gather expression,
    both dims are lane-dep -- shape has no ONE markers.
    """
    sdfg = dace.SDFG("two_dim_lane_index")
    W_i, W_j = 4, 8
    sdfg.add_array("Out", (W_i, W_j), dace.int64, transient=False)
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state,
                                           name_hint="idx_2d",
                                           gather_expr="2*i + j",
                                           tile_iter_vars=("i", "j"),
                                           tile_widths=(W_i, W_j))
    inner_an = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode))
    out_an = state.add_access("Out")
    state.add_edge(inner_an, None, out_an, None, Memlet(f"{name}[0:{W_i}, 0:{W_j}]"))
    sdfg.validate()
    out = np.zeros((W_i, W_j), dtype=np.int64)
    sdfg(Out=out)
    expected = np.array([[2 * i + j for j in range(W_j)] for i in range(W_i)], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_e2e_3d_lane_index():
    """Full K=3 dep: ``i + j + k`` materialises ``(W_i, W_j, W_k)`` -- no trailing ONE."""
    sdfg = dace.SDFG("three_dim_lane_index")
    W_i, W_j, W_k = 2, 4, 8
    sdfg.add_array("Out", (W_i, W_j, W_k), dace.int64, transient=False)
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state,
                                           name_hint="idx_3d",
                                           gather_expr="i + j + k",
                                           tile_iter_vars=("i", "j", "k"),
                                           tile_widths=(W_i, W_j, W_k))
    inner_an = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode))
    out_an = state.add_access("Out")
    state.add_edge(inner_an, None, out_an, None, Memlet(f"{name}[0:{W_i}, 0:{W_j}, 0:{W_k}]"))
    sdfg.validate()
    out = np.zeros((W_i, W_j, W_k), dtype=np.int64)
    sdfg(Out=out)
    expected = np.zeros((W_i, W_j, W_k), dtype=np.int64)
    for i in range(W_i):
        for j in range(W_j):
            for k in range(W_k):
                expected[i, j, k] = i + j + k
    np.testing.assert_array_equal(out, expected)


def test_e2e_k1_short_form_str_int_args_still_work():
    """K=1 short form (str + int) still works; emits ``(W,)`` shape (no ONE)."""
    sdfg = dace.SDFG("k1_short_form")
    W = 8
    sdfg.add_array("Out", (W, ), dace.int64, transient=False)
    state = sdfg.add_state("s")
    # K=1 caller passes a single string + single int -- the helper normalises.
    name = materialise_per_lane_index_tile(state, name_hint="idx_t", gather_expr="i", tile_iter_vars="i", tile_widths=W)
    inner_an = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode))
    out_an = state.add_access("Out")
    state.add_edge(inner_an, None, out_an, None, Memlet(f"{name}[0:{W}]"))
    sdfg.validate()
    out = np.zeros(W, dtype=np.int64)
    sdfg(Out=out)
    np.testing.assert_array_equal(out, np.arange(W, dtype=np.int64))


def test_k2_partial_k_dep_dim_marker_per_position():
    """K=2 partial-K_dep gather: shape is K-D with ONE in the non-dep position.

    Per user direction 2026-06-10:
    * ``idx[i]`` on (i,j) tile -> shape ``(W_0, ONE)``.
    * ``idx[j]`` on (i,j) tile -> shape ``(ONE, W_1)``.
    * ``idx[i, j]`` on (i,j) tile -> shape ``(W_0, W_1)``.

    No trailing ONE; the rank matches the output tile rank exactly.
    """
    from dace.symbolic import ONE
    sdfg = dace.SDFG("k2_partial_kdep")
    W_i, W_j = 4, 8
    state = sdfg.add_state("s")

    # Dep on i only -> (W_i, ONE)
    name_i = materialise_per_lane_index_tile(state,
                                             name_hint="idx_i_only",
                                             gather_expr="i",
                                             tile_iter_vars=("i", "j"),
                                             tile_widths=(W_i, W_j))
    desc_i = sdfg.arrays[name_i]
    assert len(desc_i.shape) == 2, f"expected 2-D, got {desc_i.shape}"
    assert int(desc_i.shape[0]) == W_i
    assert ONE in desc_i.shape[1].free_symbols, f"expected ONE in dim 1, got {desc_i.shape}"

    # Dep on j only -> (ONE, W_j)
    name_j = materialise_per_lane_index_tile(state,
                                             name_hint="idx_j_only",
                                             gather_expr="j",
                                             tile_iter_vars=("i", "j"),
                                             tile_widths=(W_i, W_j))
    desc_j = sdfg.arrays[name_j]
    assert len(desc_j.shape) == 2, f"expected 2-D, got {desc_j.shape}"
    assert ONE in desc_j.shape[0].free_symbols, f"expected ONE in dim 0, got {desc_j.shape}"
    assert int(desc_j.shape[1]) == W_j

    # Dep on both -> (W_i, W_j), no ONE.
    name_full = materialise_per_lane_index_tile(state,
                                                name_hint="idx_full",
                                                gather_expr="i + j",
                                                tile_iter_vars=("i", "j"),
                                                tile_widths=(W_i, W_j))
    desc_full = sdfg.arrays[name_full]
    assert len(desc_full.shape) == 2
    assert int(desc_full.shape[0]) == W_i and int(desc_full.shape[1]) == W_j
    import sympy
    for dim in desc_full.shape:
        assert not (isinstance(dim, sympy.Basic) and ONE in dim.free_symbols), \
            f"unexpected ONE in {desc_full.shape}"


def test_materialise_indirect_lane_dep_treated_as_full_dep():
    """Indirect lane-dep (per-lane symbol like ``__sym``) -> conservatively
    full-dep shape.

    The walker is the dispatch point that decides CONSTANT vs GATHER (it has
    inner_sdfg context to walk interstate edges). The materialiser trusts the
    caller: if the expression has free symbols but none of them are direct
    iter-vars, assume full-dep on every dim.
    """
    sdfg = dace.SDFG("indirect_lane_dep")
    sdfg.add_symbol("__sym_x", dace.int64)
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state,
                                           name_hint="idx_indirect",
                                           gather_expr="__sym_x",
                                           tile_iter_vars=("i", ),
                                           tile_widths=(8, ))
    desc = sdfg.arrays[name]
    # Full-dep fallback -> shape (W,) (no ONE).
    assert tuple(int(s) for s in desc.shape) == (8, )


def test_refuse_misaligned_iter_vars_and_widths():
    """Tuple lengths must match."""
    sdfg = dace.SDFG("mismatch")
    state = sdfg.add_state("s")
    import pytest as _pt
    with _pt.raises(ValueError, match="must align"):
        materialise_per_lane_index_tile(state,
                                        name_hint="bad",
                                        gather_expr="i + j",
                                        tile_iter_vars=("i", "j"),
                                        tile_widths=(8, ))
