# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`WidenScalarsToTiles`.

Builds tile-tagged Map body NSDFGs by hand and verifies the widening
pass:

  * Widens transient ``Scalar`` descriptors to ``Array(widths)``.
  * Widens transient ``(1,)`` ``Array`` descriptors (Python frontend
    artifacts) to ``Array(widths)`` -- per user direction 2026-06-10.
  * Skips loop-invariant (CONSTANT-classified) Scalars.
  * Updates every memlet referencing a widened name.
"""
import dace
from dace import data
from dace.transformation.passes.vectorization.widen_scalars_to_tiles import WidenScalarsToTiles


def _build_tagged_body(scalar_kind: str, subset: str, with_constant: bool = False):
    """Build a minimal SDFG with a tile-tagged Map enclosing a body NSDFG
    that contains one transient (Scalar or (1,)-Array) accessed with
    ``subset``.

    :param scalar_kind: ``"scalar"`` or ``"array1"``.
    :param subset: Memlet subset string for the transient (e.g. ``"0"``).
    :param with_constant: When True, the body also contains a loop-invariant
        Scalar that the pass must leave alone.
    """
    sdfg = dace.SDFG("test")
    N = dace.symbol("N")
    sdfg.add_array("A", shape=(N, ), dtype=dace.float64, transient=False)
    sdfg.add_array("B", shape=(N, ), dtype=dace.float64, transient=False)
    state = sdfg.add_state("outer")
    me, mx = state.add_map("outer", {"i": "0:N:8"})
    # Tag the map as tile-tagged using TileDimSpec-equivalent attribute.
    # The pass uses ``is_innermost_map`` + ``len(params) >= K``; with K=1 and
    # one param, the predicate passes regardless of an explicit tag.
    # Build the body NSDFG.
    inner = dace.SDFG("inner")
    inner.add_array("A", shape=(N, ), dtype=dace.float64, transient=False)
    inner.add_array("B", shape=(N, ), dtype=dace.float64, transient=False)
    if scalar_kind == "scalar":
        inner.add_scalar("t", dtype=dace.float64, transient=True)
    else:
        inner.add_array("t", shape=(1, ), dtype=dace.float64, transient=True)
    if with_constant:
        inner.add_scalar("k", dtype=dace.float64, transient=True)
    inner_state = inner.add_state("body")
    a_in = inner_state.add_read("A")
    t_an = inner_state.add_access("t")
    b_out = inner_state.add_write("B")
    tlet1 = inner_state.add_tasklet("read_a", {"_in"}, {"_out"}, "_out = _in * 2.0")
    tlet2 = inner_state.add_tasklet("write_b", {"_in"}, {"_out"}, "_out = _in + 1.0")
    inner_state.add_edge(a_in, None, tlet1, "_in", dace.Memlet(f"A[i]"))
    inner_state.add_edge(tlet1, "_out", t_an, None, dace.Memlet(f"t[{subset}]"))
    inner_state.add_edge(t_an, None, tlet2, "_in", dace.Memlet(f"t[{subset}]"))
    inner_state.add_edge(tlet2, "_out", b_out, None, dace.Memlet(f"B[i]"))
    if with_constant:
        k_an = inner_state.add_access("k")
        # CONSTANT-classified subset: just ``0`` (no iter-var ``i``).
        const_tlet = inner_state.add_tasklet("const", {}, {"_out"}, "_out = 3.14")
        inner_state.add_edge(const_tlet, "_out", k_an, None, dace.Memlet("k[0]"))
        # Wire k into tlet2 so it isn't dead-code.
        tlet2.add_in_connector("_k")
        inner_state.add_edge(k_an, None, tlet2, "_k", dace.Memlet("k[0]"))
        tlet2.code.as_string = "_out = _in + _k"
    nsdfg = state.add_nested_sdfg(inner, {"A"}, {"B"}, {"i": "i", "N": "N"})
    state.add_memlet_path(state.add_read("A"), me, nsdfg, dst_conn="A", memlet=dace.Memlet("A[0:N]"))
    state.add_memlet_path(nsdfg, mx, state.add_write("B"), src_conn="B", memlet=dace.Memlet("B[0:N]"))
    return sdfg, inner


def test_widens_transient_scalar_to_tile():
    """A transient ``Scalar`` carrying lane-dep data widens to ``Array(W,)``."""
    sdfg, inner = _build_tagged_body(scalar_kind="scalar", subset="0")
    assert isinstance(inner.arrays["t"], data.Scalar)
    result = WidenScalarsToTiles(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    new_desc = inner.arrays["t"]
    assert isinstance(new_desc, data.Array)
    assert tuple(new_desc.shape) == (8, )
    # Every memlet for ``t`` now uses [0:8].
    body_state = next(iter(inner.states()))
    for edge in body_state.edges():
        if edge.data is None or edge.data.data != "t":
            continue
        assert str(edge.data.subset) == "0:8", f"expected [0:8], got {edge.data.subset}"


def test_widens_length_one_array_to_tile():
    """A ``(1,)`` Array transient widens to ``Array(W,)`` (user 2026-06-10)."""
    sdfg, inner = _build_tagged_body(scalar_kind="array1", subset="0")
    assert isinstance(inner.arrays["t"], data.Array)
    assert tuple(inner.arrays["t"].shape) == (1, )
    result = WidenScalarsToTiles(widths=(8, )).apply_pass(sdfg, {})
    assert result == 1
    new_desc = inner.arrays["t"]
    assert isinstance(new_desc, data.Array)
    assert tuple(new_desc.shape) == (8, )
    body_state = next(iter(inner.states()))
    for edge in body_state.edges():
        if edge.data is None or edge.data.data != "t":
            continue
        assert str(edge.data.subset) == "0:8"


def test_widens_every_transient_scalar_in_body():
    """Per staging-first design: every transient Scalar / (1,) Array in a
    tile-tagged body widens. Loop-invariant data never becomes a transient
    Scalar after :class:`StageGlobalArrayThroughScalars` (it stays as a
    direct non-transient read), so the pass doesn't need a CONSTANT filter.
    Widening one extra Scalar is safe -- every lane just holds the same value.
    """
    sdfg, inner = _build_tagged_body(scalar_kind="scalar", subset="0", with_constant=True)
    result = WidenScalarsToTiles(widths=(8, )).apply_pass(sdfg, {})
    # Both ``t`` and ``k`` widen (no loop-invariant filter in this pass).
    assert result == 2
    assert tuple(inner.arrays["t"].shape) == (8, )
    assert tuple(inner.arrays["k"].shape) == (8, )


def test_noop_when_no_tile_tagged_map():
    """SDFG with no tile-eligible body NSDFG -> pass is a no-op."""
    sdfg = dace.SDFG("flat")
    sdfg.add_array("A", shape=(8, ), dtype=dace.float64, transient=False)
    state = sdfg.add_state()
    state.add_read("A")
    result = WidenScalarsToTiles(widths=(8, )).apply_pass(sdfg, {})
    assert result is None


def test_widths_validation():
    """Invalid widths length is refused at construction."""
    import pytest
    with pytest.raises(ValueError, match="widths length"):
        WidenScalarsToTiles(widths=())
    with pytest.raises(ValueError, match="widths length"):
        WidenScalarsToTiles(widths=(8, 8, 8, 8))
