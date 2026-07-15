# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra unit tests for :mod:`dace.transformation.layout.block_aware_map_tiling`.

These complement ``block_aware_map_tiling_test.py`` (which covers the 2D perfect-match /
int_floor-free case). Here we exercise, at the level of the pass itself:

* the ``apply_pass`` count and the rank filter (only maps whose param count matches
  ``tile_sizes`` are tiled) plus the ``modifies`` / ``should_reapply`` contract,
* a 1D perfect tile-then-block that emits neither ``int_floor`` nor ``Mod`` on the compute access,
* the un-tiled counterpart that *does* keep exact ``int_floor``/``Mod`` (the general block lowering),
* three simultaneously-blocked dimensions,
* a tile factor that does not divide the extent (remainder-guarded inner map), and
* idempotence when the tile size equals the map extent (a structural no-op).

Each compiled case is checked bit-exact against a NumPy oracle.
"""
import copy

import numpy

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.layout.block_aware_map_tiling import BlockAwareMapTiling
from dace.transformation.layout.split_dimensions import SplitDimensions

M = dace.symbol("M")
P = dace.symbol("P")
Q = dace.symbol("Q")
R = dace.symbol("R")


@dace.program
def vadd(A: dace.float64[M], B: dace.float64[M]):
    for i in dace.map[0:M] @ dace.ScheduleType.Sequential:
        B[i] = A[i] * 2.0


@dace.program
def add3(A: dace.float64[P, Q, R], B: dace.float64[P, Q, R], C: dace.float64[P, Q, R]):
    for i, j, k in dace.map[0:P, 0:Q, 0:R] @ dace.ScheduleType.Sequential:
        C[i, j, k] = A[i, j, k] + B[i, j, k]


@dace.program
def two_maps(A: dace.float64[M, M], C: dace.float64[M, M], v: dace.float64[M], w: dace.float64[M]):
    for i, j in dace.map[0:M, 0:M] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] * 2.0
    for i in dace.map[0:M] @ dace.ScheduleType.Sequential:
        w[i] = v[i] + 1.0


@dace.program
def vadd64(A: dace.float64[64], B: dace.float64[64]):
    for i in dace.map[0:64] @ dace.ScheduleType.Sequential:
        B[i] = A[i] + 1.0


def collect_map_entries(sdfg):
    return [n for state in sdfg.all_states() for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]


def collect_tasklet_subsets(sdfg, arr):
    """String forms of the per-iteration compute accesses to ``arr`` (edges touching a Tasklet).

    The map-boundary memlet (AccessNode <-> MapEntry) legitimately over-approximates with
    ``int_floor``; only the compute access decides whether a block match was perfect, so we
    restrict to Tasklet-incident edges -- mirroring the existing test's helper."""
    out = []
    for state in sdfg.all_states():
        for e in state.edges():
            data = e.data
            if data is None or data.data != arr or data.subset is None:
                continue
            if isinstance(e.src, dace.nodes.Tasklet) or isinstance(e.dst, dace.nodes.Tasklet):
                out.append(str(data.subset))
    return out


def test_apply_pass_tiles_only_matching_rank():
    """The pass tiles exactly the top-level maps whose param count equals ``len(tile_sizes)``:
    the 2D map is tiled (count == 1) while the sibling 1D map is left untouched."""
    sdfg = two_maps.to_sdfg()

    pass_obj = BlockAwareMapTiling(tile_sizes=(4, 4))

    # Contract of the pass wrapper (independent of any graph).
    mods = pass_obj.modifies()
    assert bool(mods & ppl.Modifies.Nodes)
    assert bool(mods & ppl.Modifies.Memlets)
    assert bool(mods & ppl.Modifies.Scopes)
    assert pass_obj.should_reapply(mods) is False

    count = pass_obj.apply_pass(sdfg, {})
    assert count == 1, "only the single 2D top-level map should have been tiled"
    sdfg.validate()

    maps = collect_map_entries(sdfg)
    # Tiling the 2D map yields an outer+inner pair; the 1D map stays a single map -> 3 total.
    assert len(maps) == 3

    # The rank-1 map was NOT tiled: its parameters and range are unchanged.
    one_param = [n for n in maps if len(n.map.params) == 1]
    assert len(one_param) == 1
    assert one_param[0].map.params == ["i"]
    assert str(one_param[0].map.range) == "0:M"

    # Exactly one outer tile map was introduced, stepping by the tile factor.
    tile_maps = [n for n in maps if len(n.map.params) == 2 and all(p.startswith("tile") for p in n.map.params)]
    assert len(tile_maps) == 1
    assert str(tile_maps[0].map.range) == "0:M:4, 0:M:4"


def test_perfect_match_emits_no_modulo_or_int_floor():
    """Tiling the flat 1D map by the block factor first makes SplitDimensions take its
    perfect-match path: the compute access is ``tile/b, i - tile`` with no ``int_floor``/``Mod``."""
    original = vadd.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "vadd_perfect"
    count = BlockAwareMapTiling(tile_sizes=(16, ), divides_evenly=True).apply_pass(sdfg, {})
    assert count == 1
    sdfg.validate()

    SplitDimensions(split_map={"A": ([True], [16])}).apply_pass(sdfg, {})
    sdfg.validate()

    subsets = collect_tasklet_subsets(sdfg, "A")
    assert subsets, "expected at least one compute access to A"
    for s in subsets:
        assert "int_floor" not in s, f"perfect match must not emit int_floor, got {s}"
        assert "Mod" not in s, f"perfect match must not emit a modulo, got {s}"

    _M = 16 * 5
    numpy.random.seed(0)
    A = numpy.random.rand(_M)
    B0 = numpy.zeros(_M)
    B1 = numpy.zeros(_M)
    original(A=A.copy(), B=B0, M=_M)
    A2 = A.reshape(_M // 16, 16).copy()  # physically blocked [M/16, 16]
    sdfg(A=A2, B=B1, M=_M)
    assert numpy.allclose(B1, B0)


def test_untiled_block_keeps_int_floor_and_modulo():
    """Without tiling, SplitDimensions cannot match a block: the compute access keeps the exact
    ``int_floor(i, b)`` tile index and ``Mod(i, b)`` offset -- and still lowers bit-exact."""
    original = vadd.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "vadd_untiled"
    SplitDimensions(split_map={"A": ([True], [16])}).apply_pass(sdfg, {})
    sdfg.validate()

    subsets = collect_tasklet_subsets(sdfg, "A")
    assert subsets, "expected at least one compute access to A"
    joined = " | ".join(subsets)
    assert "int_floor" in joined, f"un-tiled block must keep int_floor, got {joined}"
    assert "Mod" in joined, f"un-tiled block must keep a modulo, got {joined}"

    _M = 16 * 5
    numpy.random.seed(1)
    A = numpy.random.rand(_M)
    B0 = numpy.zeros(_M)
    B1 = numpy.zeros(_M)
    original(A=A.copy(), B=B0, M=_M)
    A2 = A.reshape(_M // 16, 16).copy()
    sdfg(A=A2, B=B1, M=_M)
    assert numpy.allclose(B1, B0)


def test_multiple_blocked_dims_perfect_match():
    """Three simultaneously-blocked dimensions: every compute access is a clean tile/offset pair
    (no ``int_floor``/``Mod`` on any of the three dims) and the result is bit-exact."""
    original = add3.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "add3_perfect"
    count = BlockAwareMapTiling(tile_sizes=(4, 8, 2), divides_evenly=True).apply_pass(sdfg, {})
    assert count == 1
    sdfg.validate()

    split_map = {
        "A": ([True, True, True], [4, 8, 2]),
        "B": ([True, True, True], [4, 8, 2]),
    }
    SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()

    for arr in ("A", "B"):
        subsets = collect_tasklet_subsets(sdfg, arr)
        assert subsets, f"expected a compute access to {arr}"
        for s in subsets:
            assert "int_floor" not in s, f"{arr} perfect match emitted int_floor: {s}"
            assert "Mod" not in s, f"{arr} perfect match emitted a modulo: {s}"

    _P, _Q, _R = 4 * 3, 8 * 2, 2 * 5
    numpy.random.seed(2)
    A = numpy.random.rand(_P, _Q, _R)
    B = numpy.random.rand(_P, _Q, _R)
    C0 = numpy.zeros((_P, _Q, _R))
    C1 = numpy.zeros((_P, _Q, _R))
    original(A=A.copy(), B=B.copy(), C=C0, P=_P, Q=_Q, R=_R)
    # Physically block each dim and move the block factors to the trailing axes.
    A2 = A.reshape(_P // 4, 4, _Q // 8, 8, _R // 2, 2).transpose(0, 2, 4, 1, 3, 5).copy()
    B2 = B.reshape(_P // 4, 4, _Q // 8, 8, _R // 2, 2).transpose(0, 2, 4, 1, 3, 5).copy()
    sdfg(A=A2, B=B2, C=C1, P=_P, Q=_Q, R=_R)
    assert numpy.allclose(C1, C0)


def test_non_dividing_factor_uses_remainder_guard():
    """A tile factor that does not divide the extent (``divides_evenly=False``) yields a
    remainder-guarded inner map (``Min(...)``) over a clean ``0:M:b`` outer map, still bit-exact."""
    original = vadd.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "vadd_remainder"
    count = BlockAwareMapTiling(tile_sizes=(8, ), divides_evenly=False).apply_pass(sdfg, {})
    assert count == 1
    sdfg.validate()

    maps = collect_map_entries(sdfg)
    outer = [n for n in maps if n.map.params == ["tile_i"]]
    inner = [n for n in maps if n.map.params == ["i"]]
    assert len(outer) == 1 and len(inner) == 1
    assert str(outer[0].map.range) == "0:M:8"
    assert "Min" in str(inner[0].map.range), "non-dividing factor needs a remainder guard on the inner map"

    _M = 20  # 20 % 8 != 0 -> a partial trailing tile
    numpy.random.seed(3)
    A = numpy.random.rand(_M)
    B0 = numpy.zeros(_M)
    B1 = numpy.zeros(_M)
    original(A=A.copy(), B=B0, M=_M)
    sdfg(A=A.copy(), B=B1, M=_M)
    assert numpy.allclose(B1, B0)


def test_idempotent_when_tile_equals_extent():
    """When the tile size equals the (concrete) map extent, MapTiling skips the dimension, so the
    pass is a structural no-op -- applying it twice leaves a single unchanged ``0:64`` map."""
    original = vadd64.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "vadd64_idem"
    pass_obj = BlockAwareMapTiling(tile_sizes=(64, ))

    first = pass_obj.apply_pass(sdfg, {})
    assert first == 1
    maps = collect_map_entries(sdfg)
    assert len(maps) == 1 and maps[0].map.params == ["i"]
    assert str(maps[0].map.range) == "0:64"

    second = pass_obj.apply_pass(sdfg, {})
    assert second == 1
    maps = collect_map_entries(sdfg)
    assert len(maps) == 1 and maps[0].map.params == ["i"]
    assert str(maps[0].map.range) == "0:64", "tile == extent must not add nesting on re-application"
    sdfg.validate()

    numpy.random.seed(4)
    A = numpy.random.rand(64)
    B0 = numpy.zeros(64)
    B1 = numpy.zeros(64)
    original(A=A.copy(), B=B0)
    sdfg(A=A.copy(), B=B1)
    assert numpy.allclose(B1, B0)


if __name__ == "__main__":
    test_apply_pass_tiles_only_matching_rank()
    test_perfect_match_emits_no_modulo_or_int_floor()
    test_untiled_block_keeps_int_floor_and_modulo()
    test_multiple_blocked_dims_perfect_match()
    test_non_dividing_factor_uses_remainder_guard()
    test_idempotent_when_tile_equals_extent()
    print("block-aware map tiling extra tests PASS")