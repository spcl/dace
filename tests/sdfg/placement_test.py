# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import itertools

import pytest
import sympy

import dace
from dace import placement


def brute_force(place: placement.Placement, shape: tuple[int, ...]) -> dict[tuple[int, int], tuple[int, ...]]:
    """(node, offset) -> indices for every element; asserts no two elements collide."""
    seen = {}
    for idx in itertools.product(*(range(n) for n in shape)):
        key = (int(place.node_of(idx)), int(place.offset_of(idx, shape)))
        assert key not in seen, f"{idx} and {seen[key]} both map to {key}"
        seen[key] = idx
    return seen


def test_trace_normalizes_floor_division():
    B, P = dace.symbol("B"), dace.symbol("P")
    lam = placement.SymbolicLambda(lambda i: (i // B) % P)
    assert [p.name for p in lam.params] == ["i"]
    assert lam(13).subs({B: 4, P: 2}) == 1
    assert placement.SymbolicLambda(lambda i, j: (i, j % 3))(4, 7) == (4, 1)


def test_trace_accepts_its_parameters_after_the_symbol_cache_dropped_them():
    """A parameter rebuilt after sympy's cache evicted the traced one is equal, not identical, and still accepted."""

    def body(i):
        sympy.core.cache.clear_cache()
        return (i // 4) % 2

    assert placement.SymbolicLambda(body)(13) == 1


@pytest.mark.parametrize(
    "body",
    [
        lambda i: i / 2,
        lambda i: i < 4,
        lambda i: min(i, 4),
        lambda i: i * 0.5,
        lambda i: dace.symbolic.int_ceil(i, 4),
    ],
)
def test_trace_refuses_non_block_cyclic(body):
    with pytest.raises(ValueError):
        placement.SymbolicLambda(body)


@pytest.mark.parametrize(
    "nodes, body, shape",
    [
        (4, lambda i: i % 4, (10,)),
        (3, lambda i: (i // 2) % 3, (17,)),
        (4, lambda i: i // 3, (12,)),
        (6, lambda i, j: 3 * ((i // 2) % 2) + (j // 4) % 3, (7, 13)),
        (4, lambda i, j: (j // 2) % 4, (3, 9)),
        (1, lambda i, j: 0, (2, 3)),
        # Page-cyclic: 2x3 pages over a 3x2 page grid (last pages partial), dealt to 4 nodes
        (4, lambda i, j: (2 * (i // 2) + j // 3) % 4, (5, 6)),
        (3, lambda i, j: ((i // 2) + 3 * (j // 2)) % 3, (6, 7)),
    ],
)
def test_offsets_match_brute_force(nodes, body, shape):
    place = placement.Placement(nodes, placement.SymbolicLambda(body))
    local = place.local_shape(shape)
    capacity = 1
    for extent in local:
        capacity *= int(extent)
    place.check(shape, capacity)
    pairs = brute_force(place, shape)
    assert {node for node, offset in pairs} == set(range(nodes))
    assert all(0 <= offset < capacity for node, offset in pairs)


def test_symbolic_block_cyclic_offsets():
    B, P, N = dace.symbol("B"), dace.symbol("P"), dace.symbol("N")
    place = placement.Placement(P, placement.SymbolicLambda(lambda i: (i // B) % P))
    assert place.dims[0].block == B and place.dims[0].count == P
    values = {B: 3, P: 4}
    for i in range(50):
        assert place.offset_of((i,), (N,)).subs(values) == (i // 12) * 3 + i % 3
    place.check((N,), 9, symbols={"B": 3, "P": 4, "N": 30})


def test_check_rejects_bad_maps():
    # Both dimensions select the node with stride 1: elements collide
    clash = placement.Placement(4, placement.SymbolicLambda(lambda i, j: i % 2 + j % 2))
    with pytest.raises(ValueError, match="bijection"):
        clash.check((4, 4), 16)
    # Node ids 0..3 on a 2-node memory
    with pytest.raises(ValueError, match="node ids"):
        placement.Placement(2, placement.SymbolicLambda(lambda i: i % 4)).check((8,), 4)
    # Share of 4 elements does not fit 3
    with pytest.raises(ValueError, match="capacity"):
        placement.Placement(2, placement.SymbolicLambda(lambda i: i % 2)).check((8,), 3)
    # Pages i + j collide
    with pytest.raises(ValueError, match="bijection"):
        placement.Placement(4, placement.SymbolicLambda(lambda i, j: (i + j) % 4)).check((4, 4), 4)
    # Neither per dimension nor page-cyclic
    with pytest.raises(ValueError):
        placement.Placement(4, placement.SymbolicLambda(lambda i, j: (i * j) % 4))
    with pytest.raises(ValueError):
        placement.Placement(4, placement.SymbolicLambda(lambda i, j: (i // 2 + j % 2) % 4))


def test_tensix_interleaved_tiles():
    # Tile page t = (i // 32) * T + j // 32 on bank t % NB, T tiles per row
    T, NB, M, N = (dace.symbol(name) for name in "T NB M N".split())
    place = placement.Placement(NB, placement.SymbolicLambda(lambda i, j: ((i // 32) * T + j // 32) % NB))
    assert place.pages == placement.PagePlacement((32, 32), (T, 1), NB)
    values = {T: 3, NB: 4}
    assert place.node_of((40, 70)).subs(values) == (1 * 3 + 2) % 4
    # Page 5 is the second page of bank 1: offset one page plus row 8, column 6 within the tile
    assert place.offset_of((40, 70), (M, N)).subs({**values, M: 64, N: 96}) == 1024 + 8 * 32 + 6
    place.check((M, N), 2 * 1024, symbols={"T": 3, "NB": 4, "M": 64, "N": 96})
    with pytest.raises(ValueError, match="bijection"):
        place.check((M, N), 2 * 1024, symbols={"T": 2, "NB": 4, "M": 64, "N": 96})


def test_array_placement_survives_json_and_clone():
    B = dace.symbol("B")
    sdfg = dace.SDFG("placement_roundtrip")
    sdfg.add_symbol("B", dace.int64)
    sdfg.add_array("A", [64, 32], dace.float32)
    place = placement.Placement(6, placement.SymbolicLambda(lambda i, j: 3 * ((i // B) % 2) + (j // 4) % 3))
    sdfg.arrays["A"].placement = place
    sdfg.add_array("C", [8], dace.float32)

    loaded = dace.SDFG.from_json(sdfg.to_json())
    got = loaded.arrays["A"].placement
    assert got.nodes == 6
    assert got.index_map.exprs == place.index_map.exprs
    assert got.dims == place.dims
    pages = placement.Placement(4, placement.SymbolicLambda(lambda i, j: ((i // 32) * 2 + j // 32) % 4))
    sdfg.arrays["C"].placement = None
    sdfg.arrays["A"].placement = pages
    assert dace.SDFG.from_json(sdfg.to_json()).arrays["A"].placement.pages == pages.pages
    sdfg.arrays["A"].placement = place
    assert loaded.arrays["C"].placement is None
    for desc in (sdfg.arrays["A"].clone(), copy.deepcopy(sdfg.arrays["A"])):
        assert desc.placement.index_map.exprs == place.index_map.exprs


if __name__ == "__main__":
    test_trace_normalizes_floor_division()
    test_trace_accepts_its_parameters_after_the_symbol_cache_dropped_them()
    test_trace_refuses_non_block_cyclic(lambda i: i / 2)
    test_trace_refuses_non_block_cyclic(lambda i: i < 4)
    test_trace_refuses_non_block_cyclic(lambda i: min(i, 4))
    test_trace_refuses_non_block_cyclic(lambda i: i * 0.5)
    test_trace_refuses_non_block_cyclic(lambda i: dace.symbolic.int_ceil(i, 4))
    test_offsets_match_brute_force(4, lambda i: i % 4, (10,))
    test_offsets_match_brute_force(3, lambda i: (i // 2) % 3, (17,))
    test_offsets_match_brute_force(4, lambda i: i // 3, (12,))
    test_offsets_match_brute_force(6, lambda i, j: 3 * ((i // 2) % 2) + (j // 4) % 3, (7, 13))
    test_offsets_match_brute_force(4, lambda i, j: (j // 2) % 4, (3, 9))
    test_offsets_match_brute_force(1, lambda i, j: 0, (2, 3))
    test_offsets_match_brute_force(4, lambda i, j: (2 * (i // 2) + j // 3) % 4, (5, 6))
    test_offsets_match_brute_force(3, lambda i, j: ((i // 2) + 3 * (j // 2)) % 3, (6, 7))
    test_symbolic_block_cyclic_offsets()
    test_check_rejects_bad_maps()
    test_tensix_interleaved_tiles()
    test_array_placement_survives_json_and_clone()
