# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`materialise_per_lane_index_tile` (design section 3.8).

Covers the single-dim dependency case end-to-end: hand-built SDFG, the
helper mints the transient + emits the materialisation tasklet, the SDFG
compiles, runs, and produces the expected per-lane indices.
"""
import numpy as np

import dace
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.prepare_per_lane_indices import (
    PreparePerLaneIndices,
    materialise_per_lane_index_tile,
)


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


def test_pass_stub_is_noop():
    """The pass class itself is a stub that returns None (no rewrites)."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    result = PreparePerLaneIndices().apply_pass(sdfg, {})
    assert result is None


def test_e2e_2d_lane_index():
    """Multi-dim dependency: ``2*i + j`` materialises a ``(W_i, W_j)`` index tile."""
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
    """K=3 dependency: ``i + j + k`` materialises a ``(W_i, W_j, W_k)`` index tile."""
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
    """K=1 short form (str + int) still works for backwards compatibility."""
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
