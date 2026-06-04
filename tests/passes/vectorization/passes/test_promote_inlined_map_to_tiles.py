# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the slice-1 helpers of :class:`PromoteInlinedMapToTiles`.

Each test builds a small SDFG with a tile-tagged Map by hand and calls
:func:`widen_body_scalars_to_tile` directly so the descriptor + memlet
rewrite can be asserted in isolation. End-to-end apply_pass tests
land in slice 4 with the full ``MarkTileDims`` -> orchestrator plumbing.

The widened SDFG is intentionally not runnable (tasklet bodies still
read scalar connectors). Slice 2 closes the loop with the tasklet ->
:class:`TileBinop` / :class:`TileUnop` rewrite.
"""
import dace
from dace import data, nodes, subsets
from dace.memlet import Memlet

from dace.transformation.passes.vectorization.promote_inlined_map_to_tiles import (
    _collect_body_scalar_transients,
    _is_scalar_shaped,
    widen_body_scalars_to_tile,
)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


def _spec(iter_vars=("i", "j"), widths=(4, 8), global_ubs=("M", "N")):
    """Build a minimal :class:`TileDimSpec`."""
    return TileDimSpec(iter_vars=tuple(iter_vars), widths=tuple(widths), global_ubs=tuple(global_ubs))


def _build_map_with_scalar_body():
    """Build: outer Map(i, j) with a body that contains:
       - global Array ``A`` -> Tasklet -> transient Scalar ``t`` -> Tasklet -> global Array ``B``.
    The body-scoped scalar ``t`` is the widening target.
    """
    sdfg = dace.SDFG("scalar_body_fixture")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    sdfg.add_array("B", (16, 32), dace.float64, transient=False)
    sdfg.add_scalar("t", dace.float64, transient=True)
    state = sdfg.add_state("s")
    me, mx = state.add_map("m", {"i": "0:16:4", "j": "0:32:8"})
    a = state.add_access("A")
    b = state.add_access("B")
    t = state.add_access("t")
    t1 = state.add_tasklet("read", {"_in"}, {"_out"}, "_out = _in")
    t2 = state.add_tasklet("write", {"_in"}, {"_out"}, "_out = _in")
    state.add_memlet_path(a, me, t1, dst_conn="_in", memlet=Memlet("A[i, j]"))
    state.add_edge(t1, "_out", t, None, Memlet("t[0]"))
    state.add_edge(t, None, t2, "_in", Memlet("t[0]"))
    state.add_memlet_path(t2, mx, b, src_conn="_out", memlet=Memlet("B[i, j]"))
    return sdfg, state, me, t


def test_is_scalar_shaped_recognises_scalar_and_length_one_arrays():
    sdfg = dace.SDFG("probe")
    sdfg.add_scalar("s", dace.float64, transient=True)
    sdfg.add_array("a1", (1, ), dace.float64, transient=True)
    sdfg.add_array("a11", (1, 1), dace.float64, transient=True)
    sdfg.add_array("big", (4, 8), dace.float64, transient=True)
    assert _is_scalar_shaped(sdfg.arrays["s"])
    assert _is_scalar_shaped(sdfg.arrays["a1"])
    assert _is_scalar_shaped(sdfg.arrays["a11"])
    assert not _is_scalar_shaped(sdfg.arrays["big"])


def test_collect_body_scalar_transients_finds_in_scope_only():
    """A transient whose AN is inside the map scope is a candidate; one whose
    AN is in another state is NOT (widening it would change a non-tile read)."""
    sdfg, state, me, _ = _build_map_with_scalar_body()
    # Add a second state that ALSO references ``t`` -- this disqualifies it.
    s2 = sdfg.add_state_after(state, "other")
    s2.add_access("t")

    scope_nodes = set(state.scope_subgraph(me).nodes())
    found = _collect_body_scalar_transients(state, me, scope_nodes)
    assert "t" not in found, "t referenced outside the scope must not be widened"


def test_widen_body_scalars_rewrites_descriptor_and_memlets():
    """``t`` widens to ``Array(shape=widths)``; the two memlets that read /
    write ``t`` get their subsets rewritten to the full tile region."""
    sdfg, state, me, t_node = _build_map_with_scalar_body()
    spec = _spec()
    counts = widen_body_scalars_to_tile(state, me, spec)
    assert "t" in counts
    # Two ``t``-touching memlets were rewritten (write from t1, read by t2).
    assert counts["t"] == 2
    # Descriptor is now an Array of shape spec.widths.
    widened = sdfg.arrays["t"]
    assert isinstance(widened, data.Array)
    assert tuple(widened.shape) == spec.widths
    # The Memlets on the t-touching edges are full-tile subsets.
    expected = subsets.Range([(0, w - 1, 1) for w in spec.widths])
    t_edges = [e for e in state.edges() if e.data is not None and e.data.data == "t"]
    assert len(t_edges) == 2
    for e in t_edges:
        assert e.data.subset == expected


def test_global_arrays_untouched():
    """Non-transient (global) arrays must not be widened by this pass."""
    sdfg, state, me, _ = _build_map_with_scalar_body()
    orig_a = tuple(sdfg.arrays["A"].shape)
    orig_b = tuple(sdfg.arrays["B"].shape)
    widen_body_scalars_to_tile(state, me, _spec())
    assert tuple(sdfg.arrays["A"].shape) == orig_a
    assert tuple(sdfg.arrays["B"].shape) == orig_b


def test_no_op_when_no_body_scalar():
    """A Map whose scope has no scalar transient leaves the SDFG unchanged."""
    sdfg = dace.SDFG("no_scalar_fixture")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    sdfg.add_array("B", (16, 32), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("m", {"i": "0:16:4", "j": "0:32:8"})
    a = state.add_access("A")
    b = state.add_access("B")
    tlet = state.add_tasklet("t", {"_in"}, {"_out"}, "_out = _in")
    state.add_memlet_path(a, me, tlet, dst_conn="_in", memlet=Memlet("A[i, j]"))
    state.add_memlet_path(tlet, mx, b, src_conn="_out", memlet=Memlet("B[i, j]"))
    counts = widen_body_scalars_to_tile(state, me, _spec())
    assert counts == {}
