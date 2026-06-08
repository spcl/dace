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
    name = materialise_per_lane_index_tile(state, name_hint="idx_t", gather_expr="i", tile_iter_var="i", tile_width=8)
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
    """``gather_expr="i"`` with ``tile_iter_var="i"`` materialises ``[0, 1, ..., W-1]``."""
    sdfg = dace.SDFG("identity_lane_index")
    sdfg.add_array("Out", (8, ), dace.int64, transient=False)
    state = sdfg.add_state("s")
    name = materialise_per_lane_index_tile(state, name_hint="idx_t", gather_expr="i", tile_iter_var="i", tile_width=8)
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
                                           tile_iter_var="i",
                                           tile_width=8)
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
                                           tile_iter_var="i",
                                           tile_width=4,
                                           idx_dtype=dace.int32)
    assert sdfg.arrays[name].dtype == dace.int32


def test_pass_stub_is_noop():
    """The pass class itself is a stub that returns None (no rewrites)."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    result = PreparePerLaneIndices().apply_pass(sdfg, {})
    assert result is None
