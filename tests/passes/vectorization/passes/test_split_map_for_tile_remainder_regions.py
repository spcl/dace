# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""G5 regression tests for :class:`SplitMapForTileRemainder` (design section 8.2).

Verifies the K-boundary peel produces exactly K+1 regions (1 interior +
K boundary slabs) for every K in {1, 2, 3} with non-divisible bounds.
This is the load-bearing region-count invariant; a Cartesian split
would produce 2^K regions which is wrong (section 8.2 algorithm).
"""
import dace
from dace.sdfg.nodes import MapEntry
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (
    SplitMapForTileRemainder,
    TILE_MAIN_MARKER,
)


def _count_map_regions(sdfg, kernel_label):
    """Count interior + boundary MapEntry nodes whose label starts with kernel_label.

    Interior = the one (if any) MapEntry whose label ends with TILE_MAIN_MARKER. All other
    matching MapEntries are boundary slabs (in masked mode they keep their original label
    so we can't filter by suffix; in scalar / tile_k1 modes they get explicit markers).
    """
    total = 0
    interior = 0
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, MapEntry):
            continue
        lbl = node.map.label
        if not lbl.startswith(kernel_label):
            continue
        total += 1
        if lbl.endswith(TILE_MAIN_MARKER):
            interior += 1
    boundary = total - interior
    return interior, boundary


def _build_kernel(K, bounds, widths, kernel_label="k"):
    """Build a hand-rolled SDFG with one innermost K-dim map at the given bounds.

    The map's body is a no-op tasklet so the split pass has something to peel.
    """
    sdfg = dace.SDFG(f"split_K{K}")
    sdfg.add_array("A", bounds, dace.float64, transient=False)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    map_dims = {f"d{p}": f"0:{bounds[p]}" for p in range(K)}
    me, mx = state.add_map(kernel_label, map_dims)
    tasklet = state.add_tasklet("body", set(), {"_out"}, "_out = 0.0")
    subset = ", ".join(f"d{p}" for p in range(K))
    state.add_memlet_path(me, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(tasklet, mx, a, src_conn="_out", memlet=dace.Memlet(f"A[{subset}]"))
    return sdfg, state, me


def test_K1_non_divisible_produces_2_regions():
    """K=1 with N % W != 0 -> 1 interior + 1 boundary = 2 regions."""
    sdfg, state, me = _build_kernel(K=1, bounds=(17, ), widths=(4, ))
    SplitMapForTileRemainder(widths=(4, ), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 1
    assert interior + boundary == 2


def test_K1_divisible_stays_one_region():
    """K=1 with N % W == 0 -> just the interior; no boundary (provably divisible)."""
    sdfg, state, me = _build_kernel(K=1, bounds=(16, ), widths=(4, ))
    SplitMapForTileRemainder(widths=(4, ), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 0


def test_K2_both_non_divisible_produces_3_regions():
    """K=2 with non-divisible bounds on both dims -> 1 interior + 2 boundary = 3 regions
    (NOT 4 = 2^K, which would be a Cartesian corner split)."""
    sdfg, state, me = _build_kernel(K=2, bounds=(17, 13), widths=(4, 8))
    SplitMapForTileRemainder(widths=(4, 8), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 2
    assert interior + boundary == 3


def test_K2_one_dim_divisible_one_not_produces_2_regions():
    """K=2 with one dim divisible, one not -> 1 interior + 1 boundary = 2 regions."""
    sdfg, state, me = _build_kernel(K=2, bounds=(16, 13), widths=(4, 8))
    SplitMapForTileRemainder(widths=(4, 8), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 1


def test_K3_all_non_divisible_produces_4_regions():
    """K=3 with non-divisible bounds on all 3 dims -> 1 interior + 3 boundary = 4 regions
    (NOT 8 = 2^K)."""
    sdfg, state, me = _build_kernel(K=3, bounds=(17, 13, 11), widths=(4, 8, 4))
    SplitMapForTileRemainder(widths=(4, 8, 4), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 3
    assert interior + boundary == 4


def test_K3_all_divisible_stays_one_region():
    """K=3 with all bounds divisible -> 1 interior, no boundaries."""
    sdfg, state, me = _build_kernel(K=3, bounds=(16, 16, 8), widths=(4, 8, 4))
    SplitMapForTileRemainder(widths=(4, 8, 4), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 0
