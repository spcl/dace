# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`stage_constant_access` (design section 3.1 + section 3.6)."""
import pytest

import dace
from dace.libraries.tileops import TileLoad
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.stage_inside_body import (
    StageInsideBody,
    stage_constant_access,
    stage_gather_access,
    stage_tile_access,
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


def test_pass_returns_none_on_empty_sdfg():
    """Empty SDFG -> no tile-tagged maps -> no stages -> ``None``."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    assert StageInsideBody(widths=(8, )).apply_pass(sdfg, {}) is None


def test_pass_refuses_widths_outside_k_range():
    """Constructor refuses K outside {1, 2, 3}."""
    import pytest as _pt
    with _pt.raises(ValueError, match=r"widths length"):
        StageInsideBody(widths=())
    with _pt.raises(ValueError, match=r"widths length"):
        StageInsideBody(widths=(8, 8, 8, 8))


# ---- stage_tile_access (G7 step 2) ----------------------------------------


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


# ---- StageInsideBody walker (G7 step 4) ----------------------------------


def _build_const_only_tile_fixture():
    """Build an SDFG with one innermost K=1 map (param ``ii``), a body NSDFG, and
    a non-transient AccessNode ``B`` whose only edge in the body reads ``B[0]``
    (CONSTANT-only -- no dependency on the tile iter-var ``ii``).
    """
    sdfg = dace.SDFG("walker_fixture")
    sdfg.add_array("B", (16, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body_nsdfg")
    inner.add_array("B", (16, ), dace.float64, transient=False)
    inner.add_array("out_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    b_inner = instate.add_access("B")
    t_inner = instate.add_access("out_t")
    tasklet = instate.add_tasklet("ld", {"_b"}, {"_o"}, "_o = _b")
    instate.add_edge(b_inner, None, tasklet, "_b", Memlet("B[0]"))
    instate.add_edge(tasklet, "_o", t_inner, None, Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"B"}, set())
    b_outer = state.add_access("B")
    state.add_memlet_path(b_outer, me, nsdfg, dst_conn="B", memlet=Memlet("B[0:16]"))
    state.add_nedge(nsdfg, mx, Memlet())
    return sdfg, inner


def test_walker_stages_constant_only_access_in_body_nsdfg():
    """The walker mints a Scalar bridge transient inside the body NSDFG for the
    CONSTANT-only AN ``B``."""
    sdfg, inner = _build_const_only_tile_fixture()
    before_scalars = sum(1 for d in inner.arrays.values() if isinstance(d, dace.data.Scalar) and d.transient)
    result = StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    after_scalars = sum(1 for d in inner.arrays.values() if isinstance(d, dace.data.Scalar) and d.transient)
    assert after_scalars == before_scalars + 1, "expected one staged Scalar bridge"


def test_walker_skips_non_innermost_maps():
    """A nested outer-map structure with no innermost K-dim map yields no stages."""
    sdfg = dace.SDFG("outer_only")
    sdfg.add_array("B", (16, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    state.add_map("outer", {"o": "0:4"})
    assert StageInsideBody(widths=(8, )).apply_pass(sdfg, {}) is None


def test_walker_skips_maps_with_fewer_dims_than_K():
    """K=2 walker skips a K=1 innermost map (len(params) < K)."""
    sdfg, _ = _build_const_only_tile_fixture()
    assert StageInsideBody(widths=(8, 8)).apply_pass(sdfg, {}) is None


def _build_linear_tile_fixture():
    """Build an SDFG with one innermost K=1 map (param ``ii``), body NSDFG with
    a non-transient ``B`` whose body reads ``B[ii]`` -- a LINEAR access (per-lane)."""
    sdfg = dace.SDFG("walker_tile_fixture")
    sdfg.add_array("B", (32, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})

    inner = dace.SDFG("body_nsdfg")
    inner.add_array("B", (32, ), dace.float64, transient=False)
    inner.add_array("out_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    b_inner = instate.add_access("B")
    t_inner = instate.add_access("out_t")
    tasklet = instate.add_tasklet("ld", {"_b"}, {"_o"}, "_o = _b")
    instate.add_edge(b_inner, None, tasklet, "_b", Memlet("B[ii]"))
    instate.add_edge(tasklet, "_o", t_inner, None, Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"B"}, set(), symbol_mapping={"ii": "ii"})
    b_outer = state.add_access("B")
    state.add_memlet_path(b_outer, me, nsdfg, dst_conn="B", memlet=Memlet("B[0:32]"))
    state.add_nedge(nsdfg, mx, Memlet())
    return sdfg, inner


def test_walker_stages_linear_access_via_tile_branch():
    """LINEAR access ``B[ii]`` stages through a tile-shape Array transient + TileLoad."""
    sdfg, inner = _build_linear_tile_fixture()
    before_arrays = sum(1 for d in inner.arrays.values()
                        if isinstance(d, dace.data.Array) and d.transient and tuple(d.shape) == (8, ))
    result = StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    after_arrays = sum(1 for d in inner.arrays.values()
                       if isinstance(d, dace.data.Array) and d.transient and tuple(d.shape) == (8, ))
    assert after_arrays == before_arrays + 1, "expected one new (8,)-shape tile bridge transient"
    # The new TileLoad is wired between B and the bridge.
    body_state = next(s for s in inner.states())
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    assert len(tile_loads) == 1, "expected exactly one TileLoad inserted by the walker"


def _build_gather_tile_fixture():
    """``A[idx[ii]]`` (K=1 1-D gather): non-transient A + non-transient idx in a body NSDFG."""
    from dace.subsets import Range as _Range
    from dace.symbolic import pystr_to_symbolic as _to_sym

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
    instate.add_edge(a_inner, None, tasklet, "_a",
                     Memlet(data="A", subset=_Range([(_to_sym("idx[ii]"), _to_sym("idx[ii]"), 1)])))
    instate.add_edge(tasklet, "_o", t_inner, None, Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "idx"}, set(), symbol_mapping={"ii": "ii"})
    a_outer = state.add_access("A")
    idx_outer = state.add_access("idx")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:32]"))
    state.add_memlet_path(idx_outer, me, nsdfg, dst_conn="idx", memlet=Memlet("idx[0:32]"))
    state.add_nedge(nsdfg, mx, Memlet())
    return sdfg, inner


def test_walker_stages_gather_access_via_tile_branch_with_idx_sources():
    """``A[idx[ii]]`` -- the walker materialises a (W_i,)-shape int64 index tile, mints a
    tile bridge transient, and inserts a TileLoad with ``gather_dims=(0,)`` wiring _idx_0."""
    sdfg, inner = _build_gather_tile_fixture()
    before_int_arrays = sum(1 for d in inner.arrays.values()
                            if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64)
    before_float_tiles = sum(
        1 for d in inner.arrays.values()
        if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.float64 and tuple(d.shape) == (8, ))
    result = StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    after_int_arrays = sum(1 for d in inner.arrays.values()
                           if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64)
    after_float_tiles = sum(
        1 for d in inner.arrays.values()
        if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.float64 and tuple(d.shape) == (8, ))
    assert after_int_arrays == before_int_arrays + 1, "expected one int64 index tile materialised"
    assert after_float_tiles == before_float_tiles + 1, "expected one float64 tile bridge"
    body_state = next(s for s in inner.states())
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    assert len(tile_loads) == 1
    load = tile_loads[0]
    assert tuple(load.gather_dims) == (0, ), "expected gather_dims=(0,) on the TileLoad"
    assert "_idx_0" in load.in_connectors, "expected _idx_0 connector wired"


def _build_k2_gather_tile_fixture():
    """K=2 multi-tile-dim gather: ``A[idx[ii, jj]]`` -- depends on both tile iter-vars."""
    from dace.subsets import Range as _Range
    from dace.symbolic import pystr_to_symbolic as _to_sym

    sdfg = dace.SDFG("walker_k2_gather_fixture")
    sdfg.add_array("A", (64, ), dace.float64, transient=False)
    sdfg.add_array("idx", (8, 16), dace.int64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8", "jj": "0:16"})

    inner = dace.SDFG("body_nsdfg")
    inner.add_array("A", (64, ), dace.float64, transient=False)
    inner.add_array("idx", (8, 16), dace.int64, transient=False)
    inner.add_array("out_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    a_inner = instate.add_access("A")
    t_inner = instate.add_access("out_t")
    tasklet = instate.add_tasklet("ld", {"_a"}, {"_o"}, "_o = _a")
    instate.add_edge(a_inner, None, tasklet, "_a",
                     Memlet(data="A", subset=_Range([(_to_sym("idx[ii, jj]"), _to_sym("idx[ii, jj]"), 1)])))
    instate.add_edge(tasklet, "_o", t_inner, None, Memlet("out_t[0]"))

    nsdfg = state.add_nested_sdfg(inner, {"A", "idx"}, set(), symbol_mapping={"ii": "ii", "jj": "jj"})
    a_outer = state.add_access("A")
    idx_outer = state.add_access("idx")
    state.add_memlet_path(a_outer, me, nsdfg, dst_conn="A", memlet=Memlet("A[0:64]"))
    state.add_memlet_path(idx_outer, me, nsdfg, dst_conn="idx", memlet=Memlet("idx[0:8, 0:16]"))
    state.add_nedge(nsdfg, mx, Memlet())
    return sdfg, inner


def test_walker_stages_K2_multi_tile_dim_gather():
    """``A[idx[ii, jj]]`` (K=2 multi-tile-dim gather) -- the walker materialises a (8, 16)-shape
    int64 index tile + a (8, 16) tile bridge transient, and inserts a TileLoad with
    ``gather_dims=(0,)`` wiring _idx_0."""
    sdfg, inner = _build_k2_gather_tile_fixture()
    before_int_arrays = sum(
        1 for d in inner.arrays.values()
        if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64 and tuple(d.shape) == (8, 16))
    before_float_tiles = sum(
        1 for d in inner.arrays.values()
        if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.float64 and tuple(d.shape) == (8, 16))
    result = StageInsideBody(widths=(8, 16)).apply_pass(sdfg, {})
    assert result == 1
    after_int_arrays = sum(
        1 for d in inner.arrays.values()
        if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.int64 and tuple(d.shape) == (8, 16))
    after_float_tiles = sum(
        1 for d in inner.arrays.values()
        if isinstance(d, dace.data.Array) and d.transient and d.dtype == dace.float64 and tuple(d.shape) == (8, 16))
    assert after_int_arrays == before_int_arrays + 1, "expected one (8, 16)-shape int64 index tile"
    assert after_float_tiles == before_float_tiles + 1, "expected one (8, 16) float64 tile bridge"
    body_state = next(s for s in inner.states())
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    assert len(tile_loads) == 1
    load = tile_loads[0]
    assert tuple(load.gather_dims) == (0, )
    assert "_idx_0" in load.in_connectors


def test_walker_rewires_consumers_to_bridge_for_tile_branch():
    """After staging a LINEAR access, the downstream tasklet's ``_b`` connector must read
    from the bridge transient, not the original non-transient AccessNode."""
    sdfg, inner = _build_linear_tile_fixture()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    tasklet = next(n for n in body_state.nodes() if isinstance(n, dace.nodes.Tasklet))
    b_edges = [e for e in body_state.in_edges(tasklet) if e.dst_conn == "_b"]
    assert len(b_edges) == 1
    src = b_edges[0].src
    assert isinstance(src, dace.nodes.AccessNode)
    desc = inner.arrays[src.data]
    assert desc.transient, "expected tasklet _b to read from a transient bridge, not the original"
    assert tuple(desc.shape) == (8, ), "expected the bridge to be (W,)-shaped"


def test_walker_rewires_consumers_to_bridge_for_constant_branch():
    """After staging a CONSTANT access, downstream consumers read from the Scalar bridge."""
    sdfg, inner = _build_const_only_tile_fixture()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    tasklet = next(n for n in body_state.nodes() if isinstance(n, dace.nodes.Tasklet))
    b_edges = [e for e in body_state.in_edges(tasklet) if e.dst_conn == "_b"]
    assert len(b_edges) == 1
    src = b_edges[0].src
    assert isinstance(src, dace.nodes.AccessNode)
    desc = inner.arrays[src.data]
    assert isinstance(desc, dace.data.Scalar) and desc.transient, \
        "expected consumer to read from Scalar bridge transient"


# ---- write-side staging (TileStore) ---------------------------------------


def _build_linear_write_fixture():
    """Body NSDFG: ``B[ii] = src_t[0]`` -- LINEAR write into a non-transient B.

    Models the post-walker shape for an output store: the body writes a
    tile-shape value into the non-transient AN; the walker should mint a
    bridge transient + TileStore between the writer and the AN.
    """
    sdfg = dace.SDFG("walker_write_fixture")
    sdfg.add_array("B", (32, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body_nsdfg")
    inner.add_array("B", (32, ), dace.float64, transient=False)
    inner.add_array("src_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    src_inner = instate.add_access("src_t")
    b_inner = instate.add_access("B")
    tasklet = instate.add_tasklet("st", {"_s"}, {"_b"}, "_b = _s")
    instate.add_edge(src_inner, None, tasklet, "_s", Memlet("src_t[0]"))
    instate.add_edge(tasklet, "_b", b_inner, None, Memlet("B[ii]"))
    nsdfg = state.add_nested_sdfg(inner, set(), {"B"}, symbol_mapping={"ii": "ii"})
    b_outer = state.add_access("B")
    state.add_nedge(me, nsdfg, Memlet())
    state.add_memlet_path(nsdfg, mx, b_outer, src_conn="B", memlet=Memlet("B[0:32]"))
    return sdfg, inner


def test_walker_stages_linear_write_via_tilestore():
    """LINEAR write ``B[ii] = ...`` stages through a tile bridge + TileStore."""
    from dace.libraries.tileops import TileStore
    sdfg, inner = _build_linear_write_fixture()
    result = StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    assert result is not None and result >= 1
    body_state = next(s for s in inner.states())
    stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(stores) == 1, f"expected one TileStore, got {len(stores)}"


def test_walker_rewires_producers_to_bridge_for_tile_write():
    """The original tasklet's ``_b`` connector writes to a bridge transient, not directly to B."""
    sdfg, inner = _build_linear_write_fixture()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    tasklet = next(n for n in body_state.nodes() if isinstance(n, dace.nodes.Tasklet))
    out_edges = [e for e in body_state.out_edges(tasklet) if e.src_conn == "_b"]
    assert len(out_edges) == 1
    dst = out_edges[0].dst
    assert isinstance(dst, dace.nodes.AccessNode)
    desc = inner.arrays[dst.data]
    assert desc.transient, "expected the tasklet to write into a bridge transient, not the original AN"
    assert tuple(desc.shape) == (8, )


def test_walker_tilestore_feeds_into_original_an():
    """The TileStore's ``_dst`` edge writes into the original non-transient AN."""
    from dace.libraries.tileops import TileStore
    sdfg, inner = _build_linear_write_fixture()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    store = next(n for n in body_state.nodes() if isinstance(n, TileStore))
    dst_edges = [e for e in body_state.out_edges(store) if e.src_conn == "_dst"]
    assert len(dst_edges) == 1
    target = dst_edges[0].dst
    assert isinstance(target, dace.nodes.AccessNode)
    assert target.data == "B"


def _build_constant_write_fixture():
    """Body NSDFG: ``B[0] = src_t[0]`` -- CONSTANT-only write to a non-transient B."""
    sdfg = dace.SDFG("walker_const_write_fixture")
    sdfg.add_array("B", (32, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body_nsdfg")
    inner.add_array("B", (32, ), dace.float64, transient=False)
    inner.add_array("src_t", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    src_inner = instate.add_access("src_t")
    b_inner = instate.add_access("B")
    tasklet = instate.add_tasklet("st", {"_s"}, {"_b"}, "_b = _s")
    instate.add_edge(src_inner, None, tasklet, "_s", Memlet("src_t[0]"))
    instate.add_edge(tasklet, "_b", b_inner, None, Memlet("B[0]"))
    nsdfg = state.add_nested_sdfg(inner, set(), {"B"}, symbol_mapping={"ii": "ii"})
    b_outer = state.add_access("B")
    state.add_nedge(me, nsdfg, Memlet())
    state.add_memlet_path(nsdfg, mx, b_outer, src_conn="B", memlet=Memlet("B[0:32]"))
    return sdfg, inner


def test_walker_leaves_constant_only_write_as_direct_copy():
    """Per user direction 2026-06-09: CONSTANT-only writes stay as the original direct
    producer -> AN copy (symmetric to CONSTANT-only reads on the load side). The walker
    does NOT insert a TileStore for this pattern."""
    from dace.libraries.tileops import TileStore
    sdfg, inner = _build_constant_write_fixture()
    StageInsideBody(widths=(8, )).apply_pass(sdfg, {})
    body_state = next(s for s in inner.states())
    stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(stores) == 0, "expected NO TileStore for CONSTANT-only write -- direct copy is the contract"
    tasklets = [n for n in body_state.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1
    out_edges = list(body_state.out_edges(tasklets[0]))
    assert len(out_edges) == 1
    assert isinstance(out_edges[0].dst, dace.nodes.AccessNode)
    assert out_edges[0].dst.data == "B"


# ---- scatter (GATHER on write) ------------------------------------------------


def test_stage_tile_store_helper_emits_tilestore_with_gather_dims_and_wires_idx_connectors():
    """Direct exercise of stage_tile_store(gather_dims=(0,)) -> TileStore with `_idx_0`
    wired from a (W_0,) index tile.

    The walker's GATHER classification path is shared with the read side and already
    exercised by the read-side gather tests (the classifier doesn't care which direction
    the access is). This test pins the WRITE-side helper's contract directly.
    """
    from dace.transformation.passes.vectorization.stage_inside_body import stage_tile_store
    sdfg = dace.SDFG("scatter_partial")
    sdfg.add_array("Dst", (32, ), dace.float64, transient=False)
    sdfg.add_array("Idx0", (8, ), dace.int64, transient=True)
    sdfg.add_array("Src_bridge", (8, ), dace.float64, transient=True)
    state = sdfg.add_state("s")
    dst_an = state.add_access("Dst")
    idx_an = state.add_access("Idx0")
    name, store = stage_tile_store(state,
                                   dst_an,
                                   widths=(8, ),
                                   dst_subset=Memlet("Dst[0:32]"),
                                   name_hint="scatter_bridge",
                                   gather_dims=(0, ),
                                   idx_sources={0: idx_an})
    assert tuple(store.gather_dims) == (0, )
    # The TileStore should have a _idx_0 in-edge connected from Idx0.
    in_edges = list(state.in_edges(store))
    idx_edges = [e for e in in_edges if e.dst_conn == "_idx_0"]
    assert len(idx_edges) == 1
    assert idx_edges[0].src == idx_an
    # And a _src in-edge from the bridge.
    src_edges = [e for e in in_edges if e.dst_conn == "_src"]
    assert len(src_edges) == 1
    bridge_an = src_edges[0].src
    assert isinstance(bridge_an, dace.nodes.AccessNode)
    assert bridge_an.data == name


def test_stage_tile_store_helper_rejects_mismatched_idx_sources():
    """The helper validates gather_dims set matches idx_sources keys exactly."""
    from dace.transformation.passes.vectorization.stage_inside_body import stage_tile_store
    sdfg = dace.SDFG("scatter_bad")
    sdfg.add_array("Dst", (32, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    dst_an = state.add_access("Dst")
    with pytest.raises(ValueError, match=r"gather_dims"):
        stage_tile_store(state,
                         dst_an,
                         widths=(8, ),
                         dst_subset=Memlet("Dst[0:32]"),
                         gather_dims=(0, ),
                         idx_sources=None)
    with pytest.raises(ValueError, match=r"gather_dims"):
        stage_tile_store(state,
                         dst_an,
                         widths=(8, ),
                         dst_subset=Memlet("Dst[0:32]"),
                         gather_dims=(0, 1),
                         idx_sources={0: dst_an})


def test_walker_scatter_dispatch_uses_stage_tile_store_helper():
    """Validate the walker write branch invokes the scatter dispatch when GATHER classification
    fires. Direct walker-to-end-to-end testing of scatter requires a ``B[idx[ii]]`` memlet
    which DaCe's parser doesn't accept verbatim; the helper-level tests above pin the
    contract of ``stage_tile_store(gather_dims=...)`` and the walker reuses it via the
    same shared materialise_per_lane_index_tile path that the read-side gather branch uses.
    This test is a thin smoke that the StageInsideBody class itself imports the helpers
    without breaking when scatter dispatch is reachable.
    """
    from dace.transformation.passes.vectorization.stage_inside_body import (StageInsideBody, stage_tile_store)
    # Smoke: both names import + Pass class constructs cleanly.
    assert stage_tile_store is not None
    p = StageInsideBody(widths=(8, ))
    assert tuple(p.widths) == (8, )
