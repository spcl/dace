# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`stage_constant_access` (design section 3.1 + section 3.6)."""
import dace
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.stage_inside_body import (
    StageInsideBody,
    stage_constant_access,
)


def _build_state_with_an(shape=(16, ), dtype=dace.float64, name="A"):
    sdfg = dace.SDFG("stage_const_fixture")
    sdfg.add_array(name, shape, dtype, transient=False)
    state = sdfg.add_state("s")
    an = state.add_access(name)
    return sdfg, state, an


def test_helper_mints_scalar_transient_and_an_to_an_edge():
    """The helper adds a Scalar transient + a direct AN -> AN edge."""
    sdfg, state, an = _build_state_with_an()
    name = stage_constant_access(state, an, name_hint="bridge")
    desc = sdfg.arrays[name]
    assert isinstance(desc, dace.data.Scalar)
    assert desc.transient
    assert desc.dtype == dace.float64
    bridge_ans = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == name]
    assert len(bridge_ans) == 1
    bridge_an = bridge_ans[0]
    # The edge is direct AN(an) -> AN(bridge); no tasklet between.
    edges = [e for e in state.edges() if e.src is an and e.dst is bridge_an]
    assert len(edges) == 1
    assert edges[0].data is not None


def test_helper_unique_names_when_called_twice():
    """Two calls with the same name_hint produce distinct transients."""
    sdfg, state, an = _build_state_with_an()
    name_a = stage_constant_access(state, an, name_hint="bridge")
    name_b = stage_constant_access(state, an, name_hint="bridge")
    assert name_a != name_b
    assert name_a in sdfg.arrays and name_b in sdfg.arrays


def test_helper_preserves_source_dtype():
    """The Scalar transient's dtype matches the source array's element dtype."""
    sdfg, state, an = _build_state_with_an(dtype=dace.int32)
    name = stage_constant_access(state, an)
    assert sdfg.arrays[name].dtype == dace.int32


def test_pass_stub_is_noop():
    """The Pass class itself is a stub returning None."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    assert StageInsideBody().apply_pass(sdfg, {}) is None


# ---- stage_tile_access (G7 step 2) ----------------------------------------

from dace.libraries.tileops import TileLoad
from dace.transformation.passes.vectorization.stage_inside_body import stage_tile_access


def test_tile_helper_mints_tile_transient_and_tileload_node():
    """The helper adds a (widths,)-shaped Array transient + a TileLoad node + wires both edges."""
    widths = (4, 8)
    sdfg = dace.SDFG("stage_tile_fixture")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    state = sdfg.add_state("s")
    an = state.add_access("A")
    src_mem = Memlet(f"A[i:i+{widths[0]}, j:j+{widths[1]}]")
    name, load = stage_tile_access(state, an, widths=widths, src_subset=src_mem, name_hint="t_bridge")
    desc = sdfg.arrays[name]
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == widths
    assert desc.transient
    assert desc.dtype == dace.float64
    # TileLoad inserted.
    assert isinstance(load, TileLoad)
    assert tuple(load.widths) == widths
    # Both edges wired: an -> load._src, load._dst -> bridge_an.
    src_edges = [e for e in state.edges() if e.src is an and e.dst is load and e.dst_conn == "_src"]
    dst_edges = [e for e in state.edges() if e.src is load and e.src_conn == "_dst"]
    assert len(src_edges) == 1 and len(dst_edges) == 1


def test_tile_helper_forwards_dim_strides_and_replicate():
    """Per-dim properties forward to the TileLoad node verbatim."""
    widths = (4, 8)
    sdfg = dace.SDFG("forward_props")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    state = sdfg.add_state("s")
    an = state.add_access("A")
    name, load = stage_tile_access(state,
                                   an,
                                   widths=widths,
                                   src_subset=Memlet("A[i:i+8, j:j+16]"),
                                   dim_strides=(2, 1),
                                   replicate_factor_per_dim=(1, 1),
                                   src_dims=(0, 1))
    assert tuple(load.dim_strides) == (2, 1)
    assert tuple(load.replicate_factor_per_dim) == (1, 1)
    assert tuple(load.src_dims) == (0, 1)


def test_tile_helper_uniquifies_transient_names():
    """Repeated calls with the same name_hint produce distinct transients."""
    widths = (4, 8)
    sdfg = dace.SDFG("unique_t")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    state = sdfg.add_state("s")
    an = state.add_access("A")
    name_a, _ = stage_tile_access(state, an, widths=widths, src_subset=Memlet(f"A[i:i+4, j:j+8]"))
    name_b, _ = stage_tile_access(state, an, widths=widths, src_subset=Memlet(f"A[i:i+4, j:j+8]"))
    assert name_a != name_b


def test_tile_helper_preserves_source_dtype():
    """Bridge transient's dtype matches the source."""
    widths = (4, 8)
    sdfg = dace.SDFG("dtype_check")
    sdfg.add_array("A", (16, 32), dace.int64, transient=False)
    state = sdfg.add_state("s")
    an = state.add_access("A")
    name, _ = stage_tile_access(state, an, widths=widths, src_subset=Memlet("A[i:i+4, j:j+8]"))
    assert sdfg.arrays[name].dtype == dace.int64


# ---- stage_gather_access (G7 step 3) -------------------------------------

import pytest
from dace.transformation.passes.vectorization.stage_inside_body import stage_gather_access


def _add_idx(sdfg, name, shape):
    sdfg.add_array(name, shape, dace.int64, transient=True)
    return name


def test_gather_helper_emits_tileload_with_gather_dims_and_wires_idx_connectors():
    """gather_dims=(0,) -> TileLoad with `_idx_0` wired from a (W_0,) index tile."""
    widths = (4, 8)
    sdfg = dace.SDFG("gather_partial")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    _add_idx(sdfg, "Idx0", (4, ))
    state = sdfg.add_state("s")
    an = state.add_access("A")
    idx_an = state.add_access("Idx0")
    name, load = stage_gather_access(state,
                                     an,
                                     widths=widths,
                                     src_subset=Memlet("A[0:16, j:j+8]"),
                                     gather_dims=(0, ),
                                     idx_sources={0: idx_an})
    assert tuple(load.gather_dims) == (0, )
    assert "_idx_0" in load.in_connectors
    # _idx_0 edge wired from Idx0 access.
    idx_edges = [e for e in state.edges() if e.src is idx_an and e.dst is load and e.dst_conn == "_idx_0"]
    assert len(idx_edges) == 1


def test_gather_helper_supports_multiple_gather_dims_with_distinct_shapes():
    """ICON pattern -- two gather dims with shape (W_i,) each."""
    widths = (4, 8, 16)  # K=3
    sdfg = dace.SDFG("icon_pattern")
    sdfg.add_array("A", (32, 32, 64), dace.float64, transient=False)
    _add_idx(sdfg, "Idx0", (4, ))
    _add_idx(sdfg, "Idx2", (4, ))
    state = sdfg.add_state("s")
    an = state.add_access("A")
    idx0 = state.add_access("Idx0")
    idx2 = state.add_access("Idx2")
    name, load = stage_gather_access(state,
                                     an,
                                     widths=widths,
                                     src_subset=Memlet("A[0:32, j:j+8, 0:64]"),
                                     gather_dims=(0, 2),
                                     idx_sources={
                                         0: idx0,
                                         2: idx2
                                     })
    assert tuple(load.gather_dims) == (0, 2)
    assert "_idx_0" in load.in_connectors
    assert "_idx_2" in load.in_connectors
    assert "_idx_1" not in load.in_connectors


def test_gather_helper_refuses_idx_sources_mismatch():
    """idx_sources keys must match gather_dims."""
    widths = (4, 8)
    sdfg = dace.SDFG("mismatch")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    _add_idx(sdfg, "Idx0", (4, ))
    state = sdfg.add_state("s")
    an = state.add_access("A")
    idx_an = state.add_access("Idx0")
    with pytest.raises(ValueError, match="must match"):
        stage_gather_access(state,
                            an,
                            widths=widths,
                            src_subset=Memlet("A[0:16, j:j+8]"),
                            gather_dims=(0, 1),
                            idx_sources={0: idx_an})
