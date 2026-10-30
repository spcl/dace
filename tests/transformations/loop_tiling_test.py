# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopTiling strip-mines a sequential loop into a tile loop holding a point loop, and UntileLoops takes it back."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopTiling
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

N = dace.symbol("N")


@dace.program
def prefix_sum(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + b[i]


@dace.program
def every_other(a: dace.float64[N]):
    for i in range(0, N, 2):
        a[i] = 2.0 * a[i]


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def run_prefix_sum(sdfg: dace.SDFG, n: int) -> np.ndarray:
    a, b = np.zeros(n), np.arange(n, dtype=np.float64)
    sdfg(a=a, b=b, N=n)
    return a


def expected_prefix_sum(n: int) -> np.ndarray:
    a, b = np.zeros(n), np.arange(n, dtype=np.float64)
    for i in range(1, n):
        a[i] = a[i - 1] + b[i]
    return a


def test_a_tiled_loop_is_a_tile_loop_holding_a_point_loop_that_computes_the_same():
    sdfg = prefix_sum.to_sdfg(simplify=True)
    (loop,) = loops(sdfg)

    LoopTiling.apply_to(sdfg, options={"tile_size": 8}, loop=loop)

    (tiles,) = [r for r in loops(sdfg) if r.parent_graph is sdfg]
    (points,) = [r for r in loops(sdfg) if r.parent_graph is tiles]
    assert loop_stride(tiles) == 8 and loop_stride(points) == 1
    sdfg.validate()
    for n in (20, 17, 8):  # a partial last tile, an odd size, one whole tile
        assert np.array_equal(run_prefix_sum(sdfg, n), expected_prefix_sum(n))


def test_untiling_a_tiled_loop_gives_one_loop_back():
    sdfg = prefix_sum.to_sdfg(simplify=True)
    (loop,) = loops(sdfg)
    tiles = LoopTiling.apply_to(sdfg, options={"tile_size": 8}, loop=loop)

    assert UntileLoops().untile_loop(tiles, sdfg)

    assert len(loops(sdfg)) == 1
    sdfg.validate()
    assert np.array_equal(run_prefix_sum(sdfg, 20), expected_prefix_sum(20))


def test_a_loop_with_a_stride_is_not_tiled():
    sdfg = every_other.to_sdfg(simplify=True)
    (loop,) = loops(sdfg)

    assert not LoopTiling.can_be_applied_to(sdfg, options={"tile_size": 8}, loop=loop)


def loop_stride(loop: LoopRegion) -> int:
    from dace.transformation.passes.analysis import loop_analysis

    return int(loop_analysis.get_loop_stride(loop))


if __name__ == "__main__":
    test_a_tiled_loop_is_a_tile_loop_holding_a_point_loop_that_computes_the_same()
    test_untiling_a_tiled_loop_gives_one_loop_back()
    test_a_loop_with_a_stride_is_not_tiled()
