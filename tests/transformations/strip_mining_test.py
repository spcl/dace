# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow.strip_mining import StripMining

import numpy as np


def test_strip_mining():
    """
    Test a simple example example where Stripmining works
    """
    # 1. The program
    sdfg = dace.SDFG("assign")
    sdfg.add_array("A", (64,), dtype=dace.uint32)
    sdfg.add_array("B", (1,), dtype=dace.uint32)
    state = sdfg.add_state("main")

    # inputs
    A = state.add_access("A")
    B = state.add_access("B")

    # kernel map
    map_entry, map_exit = state.add_map("map", dict(i="0:64"))

    # Assign tasklet
    tasklet = state.add_tasklet(
        "assign", inputs=dict(), outputs={"_out"}, code="_out = 1;", language=dace.dtypes.Language.CPP
    )

    # Write first 1 to B[0] then B[0] to A[i]
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, "_out", B, None, dace.Memlet("B[0]"))
    state.add_edge(B, None, map_exit, "IN_A", dace.Memlet("[0] -> A[i]"))
    state.add_edge(map_exit, "OUT_A", A, None, dace.Memlet("A[0:64]", volume=64))
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")

    # 2. Apply StripMining
    stripmine = StripMining()
    stripmine.map_entry = map_entry
    stripmine.dim_idx = 0
    stripmine.new_dim_prefix = "block"
    stripmine.tile_size = 32
    stripmine.tile_stride = 32
    stripmine.apply(state, sdfg)

    # 3. Run with example input and check correctness
    A_np = np.zeros([64], dtype=np.uint32)
    B_np = np.zeros([1], dtype=np.uint32)
    sdfg(A=A_np, B=B_np)
    assert np.all(A_np == 1), f"A should be all ones. but got {A_np}"


def strided_tiled_sdfg(step: int, tile_size: int):
    """``A[i] = 1`` over ``0:64:step``, strip-mined by ``tile_size`` iterations."""
    sdfg = dace.SDFG(f"strided_assign_{step}")
    sdfg.add_array("A", (64,), dtype=dace.uint32)
    state = sdfg.add_state("main")
    A = state.add_access("A")
    map_entry, map_exit = state.add_map("map", dict(i=f"0:64:{step}"))
    tasklet = state.add_tasklet("assign", inputs=dict(), outputs={"_out"}, code="_out = 1")
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, "_out", map_exit, "IN_A", dace.Memlet("A[i]"))
    state.add_edge(map_exit, "OUT_A", A, None, dace.Memlet("A[0:64]"))
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")

    stripmine = StripMining()
    stripmine.map_entry = map_entry
    stripmine.dim_idx = 0
    stripmine.new_dim_prefix = "block"
    stripmine.tile_size = tile_size
    stripmine.tile_stride = tile_size
    stripmine.apply(state, sdfg)
    return sdfg, state, map_entry


def test_strided_tile_extent_is_scaled_by_the_map_step():
    """A tile of ``tile_size`` iterations spans ``tile_size * step`` indices.

    ``Range.size()`` reads the ``SymExpr`` over-approximation, so an approximation that omits the
    step reports ``tile_size / step`` iterations per tile.
    """
    tile_size = 8
    for step in (1, 2, 4):
        _, state, map_entry = strided_tiled_sdfg(step, tile_size)
        # ``map_entry`` is the inner (tile) map after strip-mining; its parent is the tile-number map.
        assert map_entry.map.range.size() == [tile_size], (
            f"step={step}: tile holds {map_entry.map.range.size()} iterations, expected {tile_size}"
        )
        assert state.entry_node(map_entry).map.range[0][2] == tile_size * step, (
            f"step={step}: tile-number map strides {state.entry_node(map_entry).map.range[0][2]}"
        )


def test_strided_tile_assigns_every_visited_element():
    """The corrected approximation must not change which elements the strip-mined map writes."""
    sdfg, _, _ = strided_tiled_sdfg(2, 8)
    A_np = np.zeros([64], dtype=np.uint32)
    sdfg(A=A_np)
    expected = np.zeros([64], dtype=np.uint32)
    expected[0:64:2] = 1
    assert np.array_equal(A_np, expected), f"strided tiling wrote {np.flatnonzero(A_np)}"


if __name__ == '__main__':
    test_strip_mining()
    test_strided_tile_extent_is_scaled_by_the_map_step()
    test_strided_tile_assigns_every_visited_element()
